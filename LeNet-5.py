import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义LeNet-5模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)   # 3个输入通道（RGB），6个输出通道
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 6个输入通道，16个输出通道
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)         # 展平后的输入大小：16个通道 * 5x5的特征图
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)                  # 10个输出类别（CIFAR-10）

    def forward(self, x):
        # 定义前向传播过程
        x = torch.relu(self.conv1(x))  # 使用ReLU激活函数
        x = torch.max_pool2d(x, 2)     # 2x2的最大池化
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)     # 2x2的最大池化
        x = x.view(-1, 16 * 5 * 5)     # 展平张量以便输入到全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载并预处理CIFAR-10数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]范围
])

# 加载CIFAR-10训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器

model = LeNet5()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 使用SGD优化器，学习率0.01，动量0.9

# 训练模型
num_epochs = 50  # 设置训练的轮次
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader)}")  # 输出每轮的平均损失

# 在测试集上评估模型
correct = 0
total = 0
with torch.no_grad():  # 在测试时不计算梯度
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # 获取预测结果
        total += labels.size(0)  # 总的样本数
        correct += (predicted == labels).sum().item()  # 统计预测正确的样本数

accuracy = 100 * correct / total  # 计算准确率
print(f'Accuracy on the 10000 test images: {accuracy}%')  # 输出最终测试准确率
