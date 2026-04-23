import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    print("正在加载数据集...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 训练集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # 测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")


    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()

            # 第 1~3 层卷积
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)

            # 第 4~6 层卷积
            self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)

            # 第 7~10 层卷积
            self.conv7 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv9 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv10 = nn.Conv2d(128, 256, 3, padding=1)

            # 全连接层
            self.fc1 = nn.Linear(256 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            # 1-3层 + 池化
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool1(x)

            # 4-6层 + 池化
            x = torch.relu(self.conv4(x))
            x = torch.relu(self.conv5(x))
            x = torch.relu(self.conv6(x))
            x = self.pool2(x)

            # 7-10层
            x = torch.relu(self.conv7(x))
            x = torch.relu(self.conv8(x))
            x = torch.relu(self.conv9(x))
            x = torch.relu(self.conv10(x))

            # 展平 + 分类
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # ======================================================================

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # 训练
    num_epochs = 10
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