import torch
import torch.nn as nn

class TACModule(nn.Module):
    """
    Transform-Average-Concatenate (TAC) module implementation.

    args:
        input_size: int, dimension of the input feature.
        hidden_size: int, dimension of the transform and average hidden representations.
        num_mics: int or None, number of microphones (channels).
                   If None, a fixed geometry array (average across all channels) is used.
    """
    def __init__(self, input_size, hidden_size, num_mics=None):
        super(TACModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_mics = num_mics
        
        self.transform = nn.Linear(input_size, hidden_size*3)
        self.activate_transform = nn.PReLU()
        self.average = nn.Linear(hidden_size*3, hidden_size*3)
        self.activate_average = nn.PReLU()
        self.concat = nn.Linear(hidden_size*6, input_size)
        self.activate_concat = nn.PReLU()

    def forward(self, input):
        # input shape: batch, time_frames, channels, features
        batch_size, time_frames, channels, features = input.shape
        # TAC for cross-channel communication
        # Transform
        ch_input = input.view(-1, features)
        ch_transformed = self.transform(ch_input)
        ch_transformed = self.activate_transform(ch_transformed)
        ch_transformed = ch_transformed.view(batch_size, time_frames, channels, -1)
        
        # Average (across channels)
        if self.num_mics is None or self.num_mics == channels:
            # use all channels
            ch_mean = ch_transformed.mean(dim=2)
        else:
            # only consider valid channels up to self.num_mics for each batch element
            ch_mean_list = [ch_transformed[b,:,:self.num_mics].mean(dim=1) for b in range(batch_size)]
            ch_mean = torch.stack(ch_mean_list, dim=0)
        
        ch_mean = self.average(ch_mean)
        ch_mean = self.activate_average(ch_mean)
        
        # Expand for concatenation
        ch_mean = ch_mean.unsqueeze(2).expand_as(ch_transformed)
        
        # Concatenate
        ch_concatenated = torch.cat([ch_transformed, ch_mean], dim=-1)
        ch_concatenated = self.concat(ch_concatenated.view(-1, ch_concatenated.shape[-1]))
        ch_concatenated = self.activate_concat(ch_concatenated)
        ch_concatenated = ch_concatenated.view(batch_size, time_frames, channels, -1)

        return ch_concatenated


if __name__ == "__main__":
    # Example usage
    # input_size = features from previous module output.
    # hidden_size = TAC module internal hidden representation size.
    # num_mics = number of microphones/channels considered for averaging. None if fixed geometry.
    input_size = 256
    hidden_size = 512
    num_mics = 4  # None for a fixed geometry array.

    # Assuming input tensor shape is (batch, time_frames, channels, features)
    input_tensor = torch.rand(8, 120, num_mics, input_size)  # Example input

    tac_module = TACModule(input_size, hidden_size, num_mics=num_mics)
    output = tac_module(input_tensor)
    print(output.shape)  # Should output (batch, time_frames, channels, input_size)
