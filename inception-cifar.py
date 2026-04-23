# cifar10_inception_simple.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ------- Inception v1 模块定义 -------
class InceptionV1Block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        """
        in_channels: 输入通道数
        out_1x1:     1x1 分支的输出通道数
        red_3x3:     3x3 分支前的降维通道数
        out_3x3:     3x3 分支的输出通道数
        red_5x5:     5x5 分支前的降维通道数
        out_5x5:     5x5 分支的输出通道数
        pool_proj:   池化分支后的 1x1 卷积通道数
        """
        super(InceptionV1Block, self).__init__()

        # 分支 1: 1x1 卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )

        # 分支 2: 1x1 降维 -> 3x3 卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.BatchNorm2d(red_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )

        # 分支 3: 1x1 降维 -> 5x5 卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.BatchNorm2d(red_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),  # padding=2 保持尺寸
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )

        # 分支 4: 3x3 最大池化 -> 1x1 卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


# ------- 修改后的 AlexNet 结构 (集成 Inception) -------
class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 初始卷积 (Stem)，将通道变为 64，尺寸变为 16x16
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )

        # Inception 主体部分
        self.features = nn.Sequential(
            # 输入 64 -> 输出 128 (32+32+32+32)
            InceptionV1Block(64, out_1x1=32, red_3x3=32, out_3x3=32, red_5x5=32, out_5x5=32, pool_proj=32),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            # 输入 128 -> 输出 256 (64+64+64+64)
            InceptionV1Block(128, out_1x1=64, red_3x3=64, out_3x3=64, red_5x5=64, out_5x5=64, pool_proj=64),
            nn.MaxPool2d(2, 2),  # 8 -> 4

            # 输入 256 -> 输出 512 (128+128+128+128)
            InceptionV1Block(256, out_1x1=128, red_3x3=128, out_3x3=128, red_5x5=128, out_5x5=128, pool_proj=128),
            nn.MaxPool2d(2, 2)  # 4 -> 2
        )

        # 分类器
        # 输入维度计算: 512通道 * 2(宽) * 2(高)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # 数据：只做 ToTensor + 标准化（最简）
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=tf_train)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = AlexNetCIFAR().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    def evaluate():
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss_sum += F.cross_entropy(logits, y).item() * x.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        return loss_sum / total, correct / total

    # 训练若干轮（默认 10 轮，想更好就多跑几轮）
    epochs = 30
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", ncols=90)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        te_loss, te_acc = evaluate()
        print(f"  >> test loss {te_loss:.4f} | test acc {te_acc:.4f}")


if __name__ == "__main__":
    main()