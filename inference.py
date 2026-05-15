"""
Inference script for DR Grading.
Usage: python inference.py --image path/to/image.png --checkpoint checkpoints/best_model.pth --coral
"""
import argparse
import torch
from PIL import Image

from src.model import build_model, load_checkpoint
from src.dataset import get_transforms
from src.utils import ben_graham_enhance, predict_grade_from_coral

GRADE_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}


def predict(image_path, checkpoint_path, coral=False, ben_graham=True, device="cpu"):
    # pretrained=False: backbone weights come from the checkpoint, not ImageNet download
    model = build_model(coral=coral, pretrained=False)
    model = load_checkpoint(model, checkpoint_path, device)

    img = Image.open(image_path).convert("RGB")
    if ben_graham:
        img = ben_graham_enhance(img)

    transform = get_transforms(augment=False)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        if coral:
            grade = predict_grade_from_coral(output).item()
        else:
            grade = output.argmax(dim=1).item()

    return grade, GRADE_LABELS[grade]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--coral", action="store_true")
    parser.add_argument("--no-ben-graham", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grade, label = predict(
        args.image, args.checkpoint,
        coral=args.coral,
        ben_graham=not args.no_ben_graham,
        device=device
    )
    print(f"Predicted Grade : {grade}")
    print(f"Severity        : {label}")


if __name__ == "__main__":
    main()