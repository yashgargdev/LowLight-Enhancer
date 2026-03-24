import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pytorch_msssim import ssim

import torch.nn.functional as F

from model import LowLightNet

def total_variation_loss(img):
    """Penalize high-frequency noise (static) to create smoother images"""
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

def color_constancy_loss(enhanced, original):
    """Prevent the model from changing the overall color/hue of the image"""
    mean_enhanced = torch.mean(enhanced, dim=(2, 3)) # Average color per channel
    mean_original = torch.mean(original, dim=(2, 3))
    
    # Cosine similarity between the average RGB vectors
    mean_enhanced = F.normalize(mean_enhanced, p=2, dim=1)
    mean_original = F.normalize(mean_original, p=2, dim=1)
    
    cosine_sim = torch.sum(mean_enhanced * mean_original, dim=1)
    return torch.mean(1 - cosine_sim) # 0 if perfectly matching colors

class PairedImageDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.image_files = [f for f in sorted(os.listdir(low_dir)) if not f.startswith(".")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        low_path = os.path.join(self.low_dir, img_name)
        
        # Determine the corresponding high image name
        # Some datasets use "normal" instead of "low"
        if img_name.startswith("low"):
            high_img_name = img_name.replace("low", "normal", 1)
        else:
            high_img_name = img_name
            
        high_path = os.path.join(self.high_dir, high_img_name)
        
        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")
        
        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
            
        return low_img, high_img

def main():
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Hyperparameters
    batch_size = 16
    epochs = 100
    learning_rate = 1e-3
    model_save_path = "lowlight_model.pth"
    
    # Dataset transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Ensure dataset directories exist
    low_dir = os.path.join("dataset", "low")
    high_dir = os.path.join("dataset", "high")
    
    if not os.path.exists(low_dir) or not os.path.exists(high_dir):
        print(f"Dataset directories not found. Please create '{low_dir}' and '{high_dir}' and place paired images.")
        # Create dummy directories so the script doesn't completely break
        os.makedirs(low_dir, exist_ok=True)
        os.makedirs(high_dir, exist_ok=True)
        return

    # Create dataset and loader
    dataset = PairedImageDataset(low_dir, high_dir, transform=transform)
    
    if len(dataset) == 0:
        print("No images found in the dataset directories. Please add images to train.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss and optimizer
    model = LowLightNet().to(device)
    
    # Load existing weights if they exist (Fine-tuning)
    if os.path.exists(model_save_path):
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f"Loaded existing weights from {model_save_path}. Continuing training...")
        except Exception as e:
            print(f"Could not load existing weights: {e}. Starting from scratch.")
            
    criterion_l1 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for low_imgs, high_imgs in dataloader:
            low_imgs = low_imgs.to(device)
            high_imgs = high_imgs.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(low_imgs)
            
            # Compute separate losses
            loss_l1 = criterion_l1(outputs, high_imgs)
            loss_ssim = 1 - ssim(outputs, high_imgs, data_range=1.0, size_average=True)
            loss_tv = total_variation_loss(outputs)
            loss_color = color_constancy_loss(outputs, high_imgs)
            
            # Combine losses (weighted)
            # L1 and SSIM control the overall structural enhancement
            # TV loss smooths the grain (weight = 0.05)
            # Color loss prevents weird tints (weight = 0.5)
            loss = loss_l1 + (0.5 * loss_ssim) + (0.05 * loss_tv) + (0.5 * loss_color)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    # Save the trained model checkpoint
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
