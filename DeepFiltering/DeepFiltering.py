import torch
import torch.nn.functional as F

def spec_pad(spec, df_order, df_lookahead, dim):
    # Pad the spectrogram on the specified dimension. You might need to define an 
    # appropriate padding scheme based on your application context.
    # Example padding: (0, 0, df_order, df_lookahead) assumes padding on the time axis (2D input).
    return F.pad(spec, (0, 0, df_order, df_lookahead))

def forward_real_unfold(spec: torch.Tensor, coefs: torch.Tensor, df_bins: int, df_order: int) -> torch.Tensor:
    # It is assumed that:
    # spec is a Tensor of shape [B, 1, T, F, 2], where F is the number of frequency bins and '2' is for the real and imaginary parts.
    # coefs is a Tensor of shape [B, T, O, F, 2], where O is the df_order.

    # Pad the spectrogram
    padded = spec_pad(spec[..., :df_bins, :], df_order, 0, dim=-3)

    # Use unfold to create overlapping frames
    # The size of the unfolding window is df_order, and the step size is 1 for a slide of one step at a time.
    padded = padded.unfold(dimension=2, size=df_order, step=1)  # [B, 1, T-new, df_order, F, 2]

    # Rearrange dimensions to align time steps with coefficients
    padded = padded.permute(0, 2, 3, 1, 4, 5)  # [B, T-new, df_order, 1, F, 2]

    # Perform the multiplication and sum across the df_order dimension
    # separating real and imaginary parts into last dimension, and then multiply
    # with coefficients of the same shape [B, T, O, F, 2]
    result = torch.zeros_like(padded)
    real = padded[..., 0] * coefs[..., 0]  # Multiplies corresponding real parts
    imag = padded[..., 1] * coefs[..., 1]  # Multiplies imaginary*imaginary (but remember to subtract since i^2 = -1)
    result[..., 0] = real.sum(dim=2) - imag.sum(dim=2)   # Sum up across the df_order and subtract imaginary parts
    real = padded[..., 0] * coefs[..., 1]  # Multiplies real*imaginary
    imag = padded[..., 1] * coefs[..., 0]  # Multiplies imaginary*real
    result[..., 1] = real.sum(dim=2) + imag.sum(dim=2)   # Sum up across the df_order and add up

    # Squeeze to remove the one-length dimension and return result
    return result.squeeze(3)  # [B, T-new, F, 2] 
if __name__ == "__main__":
    # Example usage:
    # Creating dummy spectrogram and coefficient tensors
    B, T, F, O = 1, 100, 50, 5  # Batch size, Time steps, Frequency bins, df order
    df_bins = F  # Assuming entire frequency range is being processed
    spec = torch.randn(B, 1, T, F, 2)
    coefs = torch.randn(B, T, O, F, 2)

    # Call the forward function and print the result
    result = forward_real_unfold(spec, coefs, df_bins, O)
    print(result.shape)  # Expected output shape: [B, T-new, F, 2]
