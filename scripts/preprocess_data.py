from PIL import Image
import os

def preprocess(input_dir, output_dir, size=(64, 64)):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if not os.path.isfile(img_path): continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size)
            out_path = os.path.join(output_dir, img_name)
            img.save(out_path)
            count += 1
        except Exception as e:
            print(f"❌ Skipped {img_path}: {e}")
            continue

    print(f"✅ Preprocessing complete. {count} images saved to '{output_dir}'.")

if __name__ == "__main__":
    preprocess("data/raw", "data/processed")
