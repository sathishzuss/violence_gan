import matplotlib.pyplot as plt

# Plot the discriminator and generator losses during training
def plot_loss(d_losses, g_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title(f"GAN Training Losses (Epoch {epoch+1})")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"output/loss_epoch_{epoch+1}.png")
    plt.close()
