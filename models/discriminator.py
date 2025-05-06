import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Flatten the image to a 1D vector
            nn.Flatten(),
            nn.Linear(64 * 64 * image_channels, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a value between 0 and 1
        )

    def forward(self, img):
        return self.model(img)
