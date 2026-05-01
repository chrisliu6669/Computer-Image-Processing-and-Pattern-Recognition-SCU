import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ===================== CNN + GRU + 多层全连接 =====================
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUNet, self).__init__()

        # CNN 卷积层（
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 卷积
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 池化

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # GRU 序列学习
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接 + ReLU
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)

    def forward(self, x):
        # -------- 第一步：CNN --------
        x = self.cnn(x)  # [B, 32, 8, 8]

        # -------- reshape 成 GRU 可接收的格式 --------
        B, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3).reshape(B, H, -1)  # [B, 8, 256]

        # -------- 第二步：GRU --------
        out, _ = self.gru(x)
        x = out[:, -1, :]  # 取最后时刻

        # -------- 第三步：多层全连接 --------
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x


# ==========================================================================

def main():
    print("正在加载数据集...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR10 数据
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # ===================== 你的固定参数（完美适配） =====================
    input_size = 256  # CNN输出后自动匹配
    hidden_size = 8
    num_layers = 1
    num_classes = 10
    batch_size = 64
    num_epochs = 10
    lr = 0.0005
    # ======================================================================

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = GRUNet(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print('训练完成！')

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')


if __name__ == '__main__':
    main()