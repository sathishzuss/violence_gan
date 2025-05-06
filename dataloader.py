import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FrameDataset(Dataset):
    def __init__(self, frame_dir, image_size=64):
        self.image_paths = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)
