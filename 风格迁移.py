import os
import os
# 修复OMP错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 【新增】修复PyCharm matplotlib绘图报错
import matplotlib
matplotlib.use('TkAgg')  # 强制使用稳定绘图后端

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. 基础设置与图像处理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path, max_size=400, shape=None):
    """加载并处理图像"""
    image = Image.open(image_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape is not None:
        size = shape

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)


def im_convert(tensor):
    """将 Tensor 转回可显示的图片"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


# 2. 提取特征与格拉姆矩阵计算
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '10': 'conv3_1', '19': 'conv4_1',
                  '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


# 3. 加载 VGG19 模型 【修复】新版torchvision写法
from torchvision.models import VGG19_Weights
vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
for param in vgg.parameters():
    param.requires_grad_(False)
vgg.to(device)

# --- 图片路径 ---
content_image = load_image('mac.jpg')
style_image = load_image('水墨画.jpg', shape=content_image.shape[-2:])

target_image = content_image.clone().requires_grad_(True).to(device)

# 4. 优化器
optimizer = optim.Adam([target_image], lr=0.003)

style_weights = {'conv1_1': 1., 'conv2_1': 0.75, 'conv3_1': 0.2,
                 'conv4_1': 0.2, 'conv5_1': 0.2}

content_weight = 1
style_weight = 1e6

content_features = get_features(content_image, vgg)
style_features = get_features(style_image, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# 5. 训练
epochs = 1000
for epoch in range(1, epochs + 1):
    target_features = get_features(target_image, vgg)

    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]

        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch [{epoch}/{epochs}], Total Loss: {total_loss.item():.4f}')

# 6. 显示结果
plt.imshow(im_convert(target_image))
plt.axis('off')
plt.show()