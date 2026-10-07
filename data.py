from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    DATA_DIR,
    SEED,
    TEST_LIMIT,
    TRAIN_LIMIT,
    VAL_LIMIT,
)


def build_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """下载 Fashion-MNIST，固定划分，并构造三个 DataLoader。"""
    # TODO(FILL-01): 组合张量转换与单通道归一化。
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    full_train = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    full_test = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(SEED)
    # TODO(FILL-02): 固定随机顺序，并建立互不重叠的训练、验证、测试子集。
    order = torch.randperm(len(full_train), generator=generator).tolist()
    train_set = Subset(full_train, order[:TRAIN_LIMIT])
    val_set = Subset(full_train, order[TRAIN_LIMIT:TRAIN_LIMIT + VAL_LIMIT])
    test_set = Subset(full_test, list(range(TEST_LIMIT)))

    common = {"batch_size": BATCH_SIZE, "num_workers": 0, "pin_memory": False}
    # TODO(FILL-03): 训练集打乱，验证集和测试集保持固定顺序。
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader
