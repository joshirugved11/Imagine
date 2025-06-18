import torch
import torch.nn.functional as F

class GaussianDiffusion:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)

    def noise_images(self, x0, t):
        noise = torch.randn_like(x0)
        alpha = 1 - self.betas[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha) * x0 + torch.sqrt(1 - alpha) * noise
        return xt, noise

    def train_step(self, x0, t):
        xt, noise = self.noise_images(x0, t)
        pred_noise = self.model(xt)
        loss = F.mse_loss(pred_noise, noise)
        return loss
