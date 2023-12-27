

mean, std = self.sde.marginal_prob(x, t, y)
z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
sigmas = std[:, None, None, None]
perturbed_data = mean + sigmas * z




