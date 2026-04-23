import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib

matplotlib.use('TkAgg')


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据预处理和加载
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 下载CIFAR10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 2. 显示图片的函数
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 3. 模型定义 - 图片1中的SimpleNN
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 1536)

        self.fc2 = nn.Linear(1536, 1024)

        self.fc3 = nn.Linear(1024, 768)

        self.fc4 = nn.Linear(768, 512)

        self.fc5 = nn.Linear(512, 384)

        self.fc6 = nn.Linear(384, 256)

        self.fc7 = nn.Linear(256, 192)

        self.fc8 = nn.Linear(192, 128)

        self.fc9 = nn.Linear(128, 64)

        self.fc10 = nn.Linear(64, 10)   ##输出层·

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))

        x = self.fc10(x)
        return x


# 4. 主程序
if __name__ == '__main__':
    # 展示CIFAR10数据集的部分图片
    dataiter = iter(trainloader)
    images, labels = dataiter.__next__()

    # 显示图片
    imshow(torchvision.utils.make_grid(images))
    # 打印类别名称
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型实例 - 使用SimpleNN而不是LinearClassifier
    model = SimpleNN().to(device)

    # 定义损失函数和优化器 - 图片2中的部分
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型 - 图片2中的部分
    num_epochs = 10
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

        # 打印本轮的平均损失
        avg_loss = running_loss / len(trainloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("训练完成!")

    # 测试模型 - 图片3中的部分
    model.eval()  # 设置为评估模式
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
    print(f'模型在测试集上的准确率: {accuracy:.2f}%')

    # 保存模型
    torch.save(model.state_dict(), 'simple_nn_cifar10.pth')
    print("模型已保存为 'simple_nn_cifar10.pth'")