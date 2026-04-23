# cifar10_alexnet_simple.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ------- 最简 AlexNet 变体，适配 32x32 图像 -------
class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),             # 32 -> 16
            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),             # 16 -> 8
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)              # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*4*4, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # 数据：只做 ToTensor + 标准化（最简）
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=tf_train)
    test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

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
        return loss_sum/total, correct/total

    # 训练若干轮（默认 10 轮，想更好就多跑几轮）
    epochs = 10
    for ep in range(1, epochs+1):
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
