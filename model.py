from torch import nn


class FashionMLP(nn.Module):
    """面向 28x28 灰度图的三层全连接分类器。"""

    def __init__(self) -> None:
        super().__init__()
        # TODO(FILL-04): 定义 784→128→64→10 的三层全连接网络。
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        # TODO(FILL-05): 把输入交给已注册的子模块并返回 logits。
        return self.network(images)
