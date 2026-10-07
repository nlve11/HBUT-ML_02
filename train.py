from __future__ import annotations

import json
import platform
import time

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn

from config import EPOCHS, LEARNING_RATE, OUTPUT_DIR, prepare_runtime
from data import build_loaders
from engine import evaluate, train_one_epoch
from model import FashionMLP


def save_curves(history: dict[str, list[float]]) -> None:
    """把损失和准确率保存为一张双子图。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    device = prepare_runtime()
    train_loader, val_loader, test_loader = build_loaders()
    model = FashionMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    # TODO(FILL-11): 用全部可训练参数和配置学习率创建 Adam 优化器。
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}, torchvision={torchvision.__version__}")
    print(f"device={device}, cuda_available={torch.cuda.is_available()}")
    print(
        f"samples: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    history = {key: [] for key in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val_loss = float("inf")
    best_epoch = 0
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # TODO(FILL-12): 仅当验证损失下降时更新记录并保存 state_dict。
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
        print(
            f"epoch={epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    elapsed = time.perf_counter() - started
    (OUTPUT_DIR / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    save_curves(history)
    print(f"best_epoch={best_epoch}, elapsed_seconds={elapsed:.2f}")
    print(f"model_saved={OUTPUT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
