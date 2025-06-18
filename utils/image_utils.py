import torchvision.transforms as transforms
from PIL import Image

def load_and_preprocess(image_path, size=(64, 64)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img)
