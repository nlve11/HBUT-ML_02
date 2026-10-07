from __future__ import annotations

import torch

from config import CLASS_NAMES, OUTPUT_DIR, prepare_runtime
from data import build_loaders
from model import FashionMLP


def main() -> None:
    device = prepare_runtime()
    _, _, test_loader = build_loaders()

    # 先重建相同结构，再把可信权重字典映射到 CPU。
    restored_model = FashionMLP().to(device)
    # TODO(FILL-14): 安全加载权重并把重建模型切换到评估模式。
    state_dict = torch.load(
        OUTPUT_DIR / "best_model.pth",
        map_location=device,
        weights_only=True,
    )
    restored_model.load_state_dict(state_dict)
    restored_model.eval()

    images, labels = next(iter(test_loader))
    with torch.inference_mode():
        probabilities = restored_model(images[:8].to(device)).softmax(dim=1)
        confidences, predictions = probabilities.max(dim=1)

    for index, (truth, pred, confidence) in enumerate(
        zip(labels[:8], predictions.cpu(), confidences.cpu()), start=1
    ):
        print(
            f"sample={index:02d} true={CLASS_NAMES[truth.item()]:<12} "
            f"pred={CLASS_NAMES[pred.item()]:<12} confidence={confidence.item():.4f}"
        )


if __name__ == "__main__":
    main()
