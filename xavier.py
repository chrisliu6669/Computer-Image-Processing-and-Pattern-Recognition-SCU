import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===================== 纯 CNN + LayerNorm + Xavier 初始化 =====================
class CNN_Net(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Net, self).__init__()

        # CNN 特征提取层 + LayerNorm
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.norm1 = nn.LayerNorm([32, 32, 32])
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = nn.LayerNorm([64, 16, 16])
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm3 = nn.LayerNorm([128, 8, 8])
        self.pool3 = nn.MaxPool2d(2, 2)

        # 全连接分类层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # ===================== Xavier 初始化 =====================
        self._initialize_weights()

    def _initialize_weights(self):
        # 对所有卷积层、全连接层做 Xavier 正态初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 卷积 1
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # 卷积 2
        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # 卷积 3
        x = self.conv3(x)
        x = self.norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        # 展平
        x = torch.flatten(x, 1)

        # 全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# ======================================================

def main():
    print("正在加载数据集...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 批次大小 = 64
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型
    model = CNN_Net(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

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
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')

    print('训练完成！')

    # 测试集准确率
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