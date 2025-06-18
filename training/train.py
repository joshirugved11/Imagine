import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from models.unet import SimpleUNet
from models.diffusion import GaussianDiffusion
from tqdm import tqdm

class FlatImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = FlatImageFolder("data/processed", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleUNet().to(device)
    diffusion = GaussianDiffusion(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(5):
        for images in tqdm(dataloader):
            images = images.to(device)
            t = torch.randint(0, diffusion.timesteps, (images.size(0),)).to(device)

            loss = diffusion.train_step(images, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"✅ Epoch {epoch} Loss: {loss.item():.4f}")

         # Save checkpoint
        save_path = f"checkpoints/model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train()
