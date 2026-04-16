import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd


# ==============================================
# 加载波士顿房价数据集
# ==============================================
def load_data():
    """自定义实现 load_boston()，解决sklearn版本问题"""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return data, target


# ==============================================
# 定义全连接神经网络
# ==============================================
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 64)  # 隐藏层到隐藏层
        self.output = nn.Linear(64, 1)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)
        return x


# ==============================================
# 可视化函数
# ==============================================
def show_results(test_y, predict_y):
    plt.scatter(np.array(test_y), np.array(predict_y), marker='x', s=30, c='red')
    plt.plot(np.arange(0, 50), np.arange(0, 50), 'b--')
    plt.xlabel("original_label")
    plt.ylabel("predict_label")
    plt.title("Neural Network Regression")
    plt.show()


# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    # 1. 加载和预处理数据
    data, label = load_data()
    data = preprocessing.normalize(data.T).T

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, train_size=0.75, random_state=33
    )

    # 2. 将数据转换为PyTorch张量（来自图片2）
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print("数据加载和预处理完成")
    print(f"训练集大小: {X_train_tensor.shape}")
    print(f"测试集大小: {X_test_tensor.shape}")

    # 3. 创建模型实例
    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim)

    # 4. 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 训练模型
    num_epochs = 1000
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # 清零梯度
        outputs = model(X_train_tensor)#前向传播
        loss = criterion(outputs, y_train_tensor)# 计算损失
        loss.backward() # 反向传播
        optimizer.step()# 更新权重

        if (epoch + 1) % 100 == 0:
            train_losses.append(loss.item())
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, y_test_tensor)
        print(f'\n测试集损失: {test_loss.item():.4f}')

    # 7. 可视化结果
    #show_results(y_test, predictions.numpy().flatten())

