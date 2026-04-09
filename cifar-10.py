import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib

matplotlib.use('TkAgg')  # 如果安装了 tkinter
# matplotlib.use('Qt5Agg')  # 如果安装了 PyQt5

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 下载CIFAR10数据集，如果已下载则不会重复下载
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

# 展示部分训练集图片
def imshow(img):
    import numpy as np
    import matplotlib.pyplot as plt
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    # 展示CIFAR10数据集的部分图片
    dataiter = iter(trainloader)
    images, labels = dataiter.__next__()

    # 显示图片
    imshow(torchvision.utils.make_grid(images))
    # 打印类别名称
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Define a simple linear classifier
class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 32 * 3, 10) # CIFAR-10 images are 32x32 with 3 color channels

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.01  # 可根据训练效果调整（如损失震荡则减小，收敛慢则适当增大）
# Initialize the model, loss function, and optimizer
model = LinearClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)