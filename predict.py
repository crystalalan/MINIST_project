"""
推理代码 - 对单张图片进行预测并可视化结果
"""

import os
import sys
import torch
import matplotlib
matplotlib.use("TkAgg" if sys.platform != "linux" or "DISPLAY" in os.environ else "Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from config import DEVICE, MODEL_SAVE_PATH, RESULT_DIR
from models.net import MNISTNet


def load_model():
    model = MNISTNet().to(DEVICE)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    else:
        print(f"Model not found at {MODEL_SAVE_PATH}, please run train.py first.")
        sys.exit(1)
    model.eval()
    return model


def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    image = Image.open(image_path)
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    return pred, confidence, image


def visualize_prediction(image_path, pred, confidence, save_path=None):
    img = Image.open(image_path).convert("L")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].text(
        0.5, 0.5,
        f"Prediction: {pred}\nConfidence: {confidence:.2%}",
        fontsize=20, ha="center", va="center",
        transform=axes[1].transAxes,
    )
    axes[1].axis("off")
    axes[1].set_title("Result")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()

IMAGE_PATH = "D:/pythonAI/Day05/MNIST_project/data/eval_images/7/45.png"

def main():
    image_path = IMAGE_PATH.strip().strip('"')
    if not image_path or not os.path.exists(image_path):
        print(f"图片未找到: {image_path}")
        sys.exit(1)

    model = load_model()
    pred, confidence, _ = predict_image(model, image_path)
    print(f"预测数字: {pred} (置信度: {confidence:.2%})")

    basename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(RESULT_DIR, f"predict_{basename}.png")
    visualize_prediction(image_path, pred, confidence, save_path=save_path)


if __name__ == "__main__":
    main()
