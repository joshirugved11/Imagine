# generate_samples.py
import torch
from torchvision.utils import save_image
from models.unet import SimpleTextConditionedUNet
from models.diffusion import GaussianDiffusion
from transformers import CLIPTokenizer, CLIPTextModel
import os

def generate(prompt="a misty lake surrounded by trees"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        text_emb = text_encoder(**inputs).last_hidden_state.mean(dim=1)

    model = SimpleTextConditionedUNet().to(device)
    model_path = "checkpoints/model_epoch_9.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = GaussianDiffusion(model)

    with torch.no_grad():
        generated = diffusion.sample(batch_size=4, shape=(3, 64, 64), device=device, text_emb=text_emb)

    os.makedirs("outputs", exist_ok=True)
    save_image((generated + 1) / 2, "outputs/generated.png")
    print("✅ Image generated for prompt:", prompt)

if __name__ == "__main__":
    generate()
