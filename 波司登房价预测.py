import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 切换兼容后端，彻底解决报错
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# ==============================================
# 自定义实现 load_boston()，彻底解决 sklearn 报错
# ==============================================
import pandas as pd


def show_results(predict_y, test_y):
    plt.scatter(np.array(test_y), np.array(predict_y), marker='x', s=30, c='red')
    plt.plot(np.arange(0, 50), np.arange(0, 50))
    plt.xlabel("original_label")
    plt.ylabel("predict_label")
    plt.title("LinerRegression")
    plt.show()

# def gradient_descent(train_x, train_y, maxCycle, alpha):
#     numSamples, numFeatures = np.shape(train_x)
#     weights = np.zeros((numFeatures, 1))
#
#     for i in range(maxCycle):
#         h = train_x * weights
#         err = h - train_y
#         weights = weights - (alpha * err.T * train_x).T
#
#     return weights

# def stochastic_gradient_descent(train_x, train_y, maxCycle, alpha):
#     numSamples, numFeatures = np.shape(train_x)
#     weights = np.zeros((numFeatures, 1))
#
#     for i in range(maxCycle):
#         for j in range(numSamples):
#             h = train_x[j, :] * weights
#             err = h - train_y[j, 0]
#             weights = weights - (alpha * err.T * train_x[j, :]).T
#
#     return weights

# 最小二乘法直接求解权重系数
# def least_square(train_x, train_y):
#     """
#     input: 训练数据（样本*属性）和标签
#     """
#     weights = (train_x.T * train_x).I * train_x.T * train_y
#     return weights

# 岭回归直接求解权重系数
def Ridge(train_x, train_y, lamda):
    """
    input: 训练数据（样本*属性）和标签
    """
    # 获取样本数和特征数
    numSamples, numFeatures = np.shape(train_x)

    # 岭回归闭式解：w = (X^T X + λI)^{-1} X^T y
    weights = (train_x.T @ train_x + lamda * np.eye(numFeatures)).I @ train_x.T @ train_y

    return weights
def load_data():
    # 直接从原始数据源下载，不需要旧版sklearn
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]


    return data, target



if __name__ == "__main__":
    data, label = load_data()
    data = preprocessing.normalize(data.T).T
##
    train_x, test_x, train_y, test_y = train_test_split(data, label, train_size=0.75, random_state=33)
    train_x = np.asmatrix(train_x)
    test_x = np.asmatrix(test_x)
    train_y = np.asmatrix(train_y).T
    test_y = np.asmatrix(test_y).T

    print(" 运行成功！数据已加载完成")

    weights = Ridge(train_x, train_y,lamda=0.1)
    predict_y = test_x * weights
    show_results(predict_y, test_y)