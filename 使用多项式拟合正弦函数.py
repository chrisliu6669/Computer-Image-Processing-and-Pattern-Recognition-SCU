import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 切换兼容后端，彻底解决报错
import matplotlib.pyplot as plt

# 生成样本数据
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.5, size=x.shape)  # 正弦函数加噪声

# 使用numpy的polyfit函数进行三次多项式拟合
coefficients = np.polyfit(x, y, 9)

# 生成拟合函数
polynomial = np.poly1d(coefficients)

# 计算拟合值
y_fit = polynomial(x)



# 输出拟合系数
print("Fitted coefficients (highest degree first):", coefficients)
# 绘制结果
plt.scatter(x, y, label='Data', color='blue', alpha=0.5)
plt.plot(x, y_fit, label='Cubic Fit', color='red')
plt.title('Cubic Polynomial Regression Fit to Sine Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()


