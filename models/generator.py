import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, image_channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Fully connected layers to upscale latent vector to a feature map
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64 * 64 * image_channels),
            nn.Tanh()  # Output a tensor between -1 and 1 (normalized)
        )

    def forward(self, z):
        return self.model(z).view(-1, 3, 64, 64)
