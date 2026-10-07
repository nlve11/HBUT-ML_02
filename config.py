from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42
BATCH_SIZE = 128
TRAIN_LIMIT = 10_000
VAL_LIMIT = 2_000
TEST_LIMIT = 2_000
EPOCHS = 6
LEARNING_RATE = 1e-3

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def prepare_runtime() -> torch.device:
    """固定随机性、创建输出目录并强制使用 CPU。"""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return torch.device("cpu")
