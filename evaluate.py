"""
评测代码 - 对评测集中的图片进行批量评测，保存结果并可视化

支持两种图片组织方式：
1. 按文件夹分类（推荐）：data/eval_images/0/xxx.png, data/eval_images/1/xxx.png, ...
   目录名 0-9 作为真实标签，自动计算准确率
2. 平铺模式：所有图片直接放在 data/eval_images/ 下，仅推理不计算准确率
"""

import os
import sys
import torch
import matplotlib
matplotlib.use("TkAgg" if sys.platform != "linux" or "DISPLAY" in os.environ else "Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from config import DEVICE, MODEL_SAVE_PATH, EVAL_IMAGE_DIR, EVAL_RESULT_PATH, RESULT_DIR
from models.net import MNISTNet


def load_model():
    model = MNISTNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def get_eval_images():
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    images = []

    subdirs = [
        d for d in os.listdir(EVAL_IMAGE_DIR)
        if os.path.isdir(os.path.join(EVAL_IMAGE_DIR, d)) and d.isdigit()
    ]

    if subdirs:
        for label_dir in sorted(subdirs, key=int):
            dir_path = os.path.join(EVAL_IMAGE_DIR, label_dir)
            for fname in sorted(os.listdir(dir_path)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in supported:
                    images.append((os.path.join(dir_path, fname), int(label_dir)))
    else:
        for fname in sorted(os.listdir(EVAL_IMAGE_DIR)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported:
                images.append((os.path.join(EVAL_IMAGE_DIR, fname), None))

    return images


def predict_batch(model, image_paths):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    results = []
    images = []

    for path, label in image_paths:
        image = Image.open(path)
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        results.append({
            "filename": os.path.basename(path),
            "label": label,
            "prediction": pred,
            "confidence": confidence,
            "probabilities": probs.cpu().numpy()[0].tolist(),
        })
        images.append((image, label, pred, confidence))

    return results, images


def save_results(results):
    with open(EVAL_RESULT_PATH, "w") as f:
        for i, r in enumerate(results, 1):
            filename = r["filename"]
            label = r["label"]
            pred = r["prediction"]
            if label is not None:
                status = "正确" if label == pred else "错误"
                f.write(f"真值：{label}  预测：{pred}  {status}\n")
            else:
                f.write(f"{filename}  预测：{pred}\n")

        labeled = [r for r in results if r["label"] is not None]
        if labeled:
            correct = sum(1 for r in labeled if r["label"] == r["prediction"])
            accuracy = correct / len(labeled) * 100
            f.write(f"\n准确率： {accuracy:.0f}%\n")

    print(f"Results saved to {EVAL_RESULT_PATH}")


def visualize_results(results, images, save_path=None):
    n = len(images)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (img, label, pred, conf) in enumerate(images):
        orig_img = img.resize((28, 28))
        axes[idx].imshow(orig_img, cmap="gray")

        if label is not None:
            status = "V" if label == pred else "X"
            title = f"True:{label} Pred:{pred} {status}\n{conf:.2%}"
            color = "green" if label == pred else "red"
        else:
            title = f"Pred: {pred}\n{conf:.2%}"
            color = "black"

        axes[idx].set_title(title, fontsize=8, color=color)
        axes[idx].axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Evaluation Results (V=Correct, X=Wrong)", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found at {MODEL_SAVE_PATH}, please run train.py first.")
        return

    image_paths = get_eval_images()
    if not image_paths:
        print(f"No images found in {EVAL_IMAGE_DIR}")
        print("Supported modes:")
        print("  1. data/eval_images/0/xxx.png, data/eval_images/1/xxx.png, ...")
        print("  2. data/eval_images/xxx.png (flat, no label)")
        return

    labeled = [p for p in image_paths if p[1] is not None]
    print(f"Found {len(image_paths)} images for evaluation"
          f"{f' ({len(labeled)} with ground truth label)' if labeled else ''}.")

    model = load_model()
    results, images = predict_batch(model, image_paths)

    save_results(results)

    labeled_results = [r for r in results if r["label"] is not None]
    if labeled_results:
        correct = sum(1 for r in labeled_results if r["label"] == r["prediction"])
        accuracy = correct / len(labeled_results) * 100
        print(f"Evaluation accuracy: {accuracy:.2f}% ({correct}/{len(labeled_results)})")

    save_path = os.path.join(RESULT_DIR, "evaluation_visualization.png")
    print("Generating visualization...")
    visualize_results(results, images, save_path=save_path)


if __name__ == "__main__":
    main()
