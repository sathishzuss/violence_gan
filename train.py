import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from dataloader import FrameDataset
from config import Config
from utils.visualizer import plot_loss

# ---------- Function for Training the GAN ----------
def train_gan(dataloader, generator, discriminator, g_opt, d_opt, criterion, device, epochs=Config.EPOCHS):
    # Loss lists for plotting
    d_losses, g_losses = [], []

    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)

            # Train Discriminator (Real and Fake)
            z = torch.randn(imgs.size(0), Config.LATENT_DIM).to(device)
            fake_imgs = generator(z).detach()
            real_labels = torch.ones(imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1).to(device)

            # Real loss
            d_loss_real = criterion(discriminator(imgs), real_labels)
            # Fake loss
            d_loss_fake = criterion(discriminator(fake_imgs), fake_labels)
            d_loss = d_loss_real + d_loss_fake

            # Update Discriminator
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train Generator (Fake images that are classified as real)
            z = torch.randn(imgs.size(0), Config.LATENT_DIM).to(device)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), real_labels)

            # Update Generator
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            # Track losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            # Print progress every 100 batches
            if i % 100 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Plot loss after each epoch
        plot_loss(d_losses, g_losses, epoch)

        # Save models periodically
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"output/generator_epoch{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"output/discriminator_epoch{epoch+1}.pth")
        
    print(f"Training finished! Models saved in output folder.")

# ---------- Main ----------
if __name__ == "__main__":
    # Initialize models, optimizers, and loss function
    device = torch.device(Config.DEVICE)
    
    generator = Generator(latent_dim=Config.LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    
    g_opt = optim.Adam(generator.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Load dataset and dataloader
    dataset = FrameDataset(Config.TRAIN_FRAME_DIR, image_size=Config.IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Train the GAN
    print("[*] Starting GAN training...")
    train_gan(dataloader, generator, discriminator, g_opt, d_opt, criterion, device, epochs=Config.EPOCHS)
