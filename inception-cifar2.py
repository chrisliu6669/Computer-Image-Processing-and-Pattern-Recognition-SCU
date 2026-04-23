# cifar10_inception_pro.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ===================== 标准 Inception 模块（带BN）=====================
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2_reduce, c2, c3_reduce, c3, c4):
        super().__init__()
        # 分支1：1x1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        # 分支2：1x1 → 3x3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, c2_reduce, 1, bias=False),
            nn.BatchNorm2d(c2_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2_reduce, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        # 分支3：1x1 → 5x5
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, c3_reduce, 1, bias=False),
            nn.BatchNorm2d(c3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3_reduce, c3, 5, padding=2, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        # 分支4：pool → 1x1
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

# ===================== 完整 Inception 网络（CIFAR10 专用）=====================
class InceptionCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32→16
        )

        self.inception1 = Inception(64, 32, 32, 48, 8, 12, 16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 16→8

        self.inception2 = Inception(32+48+12+16, 48, 48, 64, 12, 16, 16)
        self.pool2 = nn.MaxPool2d(2, 2)  # 8→4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(48+64+16+16, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.pool2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)

# ===================== 训练主函数 =====================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_set  = datasets.CIFAR10('./data', train=False, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = InceptionCIFAR().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    def test():
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total

    for epoch in range(1, 21):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/20")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.2f}")

        scheduler.step()
        acc = test()
        print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()