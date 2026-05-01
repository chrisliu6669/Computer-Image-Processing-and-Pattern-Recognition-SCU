import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===================== 精简 CNN + GRU + 多层全连接 =====================
class CNN_GRU_Net(nn.Module):
    def __init__(self, gru_input_size, hidden_size, num_layers, num_classes):
        super(CNN_GRU_Net, self).__init__()

        # ===================== 精简 CNN（层数大幅减少） =====================
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # ====================================================================

        # GRU
        self.gru = nn.GRU(gru_input_size, hidden_size, num_layers, batch_first=True)

        # 多层全连接
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)

    def forward(self, x):
        # 精简 CNN 前向
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = torch.relu(self.conv3(x))  # 输出 [B, 128, 8, 8]

        # 转成 GRU 格式 [B, 8, 128]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, H, -1)

        # GRU
        out, _ = self.gru(x)
        x = out[:, -1, :]

        # 多层全连接
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x

# ======================================================================

def main():
    print("正在加载数据集...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ===================== 超参数 =====================
    gru_input_size = 128 * 8  # 对应精简CNN
    hidden_size = 8
    num_layers = 1
    num_classes = 10
    num_epochs = 20
    lr = 0.0005

    model = CNN_GRU_Net(gru_input_size, hidden_size, num_layers, num_classes).to(device)
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print('训练完成！')

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