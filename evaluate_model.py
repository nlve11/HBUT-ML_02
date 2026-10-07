from __future__ import annotations

import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch import nn

from config import CLASS_NAMES, OUTPUT_DIR, prepare_runtime
from data import build_loaders
from engine import evaluate
from model import FashionMLP


@torch.inference_mode()
def save_error_grid(model, loader, device) -> int:
    """保存前 12 个错误样本，标题中展示真实类与预测类。"""
    model.eval()
    errors = []
    for images, labels in loader:
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1).cpu()
        for image, truth, pred in zip(images, labels, predictions):
            if truth.item() != pred.item():
                errors.append((image, truth.item(), pred.item()))
            if len(errors) == 12:
                break
        if len(errors) == 12:
            break

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for axis, (image, truth, pred) in zip(axes.flat, errors):
        display_image = image.squeeze().mul(0.5).add(0.5).clamp(0, 1)
        axis.imshow(display_image, cmap="gray")
        axis.set_title(f"T:{CLASS_NAMES[truth]}\nP:{CLASS_NAMES[pred]}", fontsize=8)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_samples.png", dpi=160)
    plt.close(fig)
    return len(errors)


def main() -> None:
    device = prepare_runtime()
    _, _, test_loader = build_loaders()
    model = FashionMLP().to(device)
    weights_path = OUTPUT_DIR / "best_model.pth"
    # TODO(FILL-13): 以权重模式映射到 CPU，并装入相同结构的模型。
    state_dict = torch.load(
        weights_path, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)

    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device
    )
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=True
    )
    (OUTPUT_DIR / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES, cmap="Blues", xticks_rotation=45
    )
    display.figure_.set_size_inches(10, 8)
    display.figure_.tight_layout()
    display.figure_.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)

    error_count = save_error_grid(model, test_loader, device)
    print(f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    print(f"macro_f1={report['macro avg']['f1-score']:.4f}")
    print(f"saved_error_examples={error_count}")


if __name__ == "__main__":
    main()
