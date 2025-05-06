import torch
import cv2
import os
from models.generator import Generator
from config import Config
from utils.video_utils import extract_frames_from_video
from utils.visualizer import plot_loss

# ---------- Function for Anomaly Detection ----------
def detect_violence(video_path, generator, threshold=Config.ANOMALY_THRESHOLD, device=Config.DEVICE):
    """
    Detect violence in a video by comparing reconstruction error
    from the Generator model.
    """
    # Ensure output directories exist
    output_dir = os.path.join(Config.OUTPUT_DIR, os.path.basename(video_path).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames from the video
    frame_dir = os.path.join(output_dir, "frames")
    extract_frames_from_video(video_path, frame_dir, frame_rate=Config.FRAME_RATE)

    # Iterate through frames, process with Generator, and check reconstruction error
    violent_frames = []
    for frame_file in os.listdir(frame_dir):
        if frame_file.endswith(".jpg"):
            frame_path = os.path.join(frame_dir, frame_file)

            # Load and prepare image
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img).float().to(device) / 255.0  # Normalize to [0, 1]
            img = img.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, 3, H, W)

            # Generate reconstructed image
            z = torch.randn(1, Config.LATENT_DIM).to(device)
            reconstructed_img = generator(z)

            # Calculate reconstruction error (mean squared error)
            mse_loss = torch.mean((reconstructed_img - img) ** 2)

            # Compare to threshold and save violent frame if necessary
            if mse_loss.item() > threshold:
                violent_frames.append((frame_path, mse_loss.item()))
                cv2.imwrite(os.path.join(output_dir, f"violent_{frame_file}"), cv2.cvtColor(img.permute(0, 2, 3, 1)[0].cpu().numpy(), cv2.COLOR_RGB2BGR))

    return violent_frames


# ---------- Main ----------
if __name__ == "__main__":
    # Initialize device and load trained model
    device = torch.device(Config.DEVICE)
    generator = Generator(latent_dim=Config.LATENT_DIM).to(device)
    generator.load_state_dict(torch.load("output/generator_epoch50.pth"))
    generator.eval()  # Set to evaluation mode

    # Define the path for the video to analyze
    video_path = "data/videos/test/sample_violence_video.mp4"

    # Detect violence in the video
    print("[*] Detecting violence in the video...")
    violent_frames = detect_violence(video_path, generator, threshold=Config.ANOMALY_THRESHOLD, device=device)

    # Display and log the results
    if violent_frames:
        print(f"Violence detected in {len(violent_frames)} frames!")
        for frame, loss in violent_frames:
            print(f"Frame {frame} - Loss: {loss:.4f}")
    else:
        print("No violence detected in the video.")
