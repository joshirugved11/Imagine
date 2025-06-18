import torch
from models.unet import SimpleUNet
from torchvision.utils import save_image

def generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("checkpoints\model_epoch_0.pt"))  # Example
    model.eval()

    noise = torch.randn(16, 3, 64, 64).to(device)
    with torch.no_grad():
        generated = model(noise)
    
    save_image(generated, "outputs/generated.png")
    print("✅ Images generated!")

if __name__ == "__main__":
    generate()
