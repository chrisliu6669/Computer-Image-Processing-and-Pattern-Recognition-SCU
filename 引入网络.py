import torch
import torch.nn as nn

# 1. 定义模型类
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)
        return x

# 2. 实例化与使用模型
model = SimpleNN()
print(model)

random_input = torch.randn(8, 784)
output = model(random_input)
print(output)