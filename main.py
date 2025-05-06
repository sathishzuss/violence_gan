import os
import argparse
from train import train_gan
from detect import detect_violence
from models.generator import Generator
from models.discriminator import Discriminator
from config import Config
from torch.utils.data import DataLoader
from dataloader import FrameDataset
import torch
import torch.optim as optim
import torch.nn as nn
from inference_pipeline import InferencePipeline
from utils.visualizer import render_boxes

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Violence Detection using GAN")
    parser.add_argument('--mode', choices=['train', 'detect'], required=True, 
                        help="Choose between training or detecting violence.")
    parser.add_argument('--video', type=str, help="Path/URL/Device ID of the video stream for detection.")
    return parser.parse_args()

def train():
    """
    Function to train the GAN for violence detection.
    """
    # Initialize device and models
    device = torch.device(Config.DEVICE)
    generator = Generator(latent_dim=Config.LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers and loss function
    g_opt = optim.Adam(generator.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Load dataset and dataloader
    dataset = FrameDataset(Config.TRAIN_FRAME_DIR, image_size=Config.IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Start training the GAN
    print("[*] Starting GAN training...")
    train_gan(dataloader, generator, discriminator, g_opt, d_opt, criterion, device, epochs=Config.EPOCHS)

def detect(video_path):
    """
    Function to detect violence in a video.
    """
    # ---------- Parse command-line argument for video ----------
    parser = argparse.ArgumentParser(description="Violence detection from video stream.")
    parser.add_argument('--video', required=True, help='Path/URL/Device ID of the video stream')
    args = parser.parse_args()

    # ---------- Create the inference pipeline ----------
    pipeline = InferencePipeline.init(
        model_id="violence-detection-p4qev/2",  # Pre-trained model identifier
        video_reference=args.video,  # Use the passed video argument (path, URL, or device ID)
        on_prediction=render_boxes,  # This function will render boxes for detected objects
        api_key="pObFI3fotnvviosDZqfo"  # API key for authentication
    )

    # ---------- Start the pipeline and wait for it to finish ----------
    print(f"[*] Detecting violence in {args.video}...")
    pipeline.start()  # Start processing the video stream
    pipeline.join()  # Wait until the pipeline is completed

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'detect':
        if not args.video:
            print("Error: Video path must be provided for detection.")
        else:
            detect(args.video)
