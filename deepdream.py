import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ===================== VGG16 修复 =====================
try:
    from torchvision.models import vgg16, VGG16_Weights
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
except:
    from torchvision.models import vgg16
    model = vgg16(pretrained=True).features

for layer in model.modules():
    if isinstance(layer, nn.ReLU):
        layer.inplace = False

model.eval()
device = torch.device("cpu")
model = model.to(device)

# ===================== 图像工具 =====================
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert("RGB")
    size = max(image.size)
    if size > max_size:
        ratio = max_size / size
        image = image.resize((int(image.size[0]*ratio), int(image.size[1]*ratio)), Image.Resampling.LANCZOS)
    return image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def deprocess_image(tensor):
    image = tensor.squeeze().cpu().detach().numpy().transpose(1,2,0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

# ===================== DeepDream =====================
def dream(image, model, num_steps=50, step_size=0.02, layers=[10,19,28]):
    image.requires_grad = True

    for step in range(num_steps):
        x = image
        activations = []

        for i, layer in enumerate(model):
            x = layer(x)
            if i in layers:
                activations.append(x.norm())

        loss = -sum(activations)
        loss.backward()

        grad = image.grad.data
        grad /= grad.norm() + 1e-8
        image.data += step_size * grad
        image.grad.zero_()
        image.data = torch.clamp(image.data, -2.5, 2.5)

        if step % 10 == 0:
            print(f"Step {step}/{num_steps}")

    return image

# ===================== 主程序（保留弹窗 + 修复插件 bug） =====================
def main(image_path):
    image = load_image(image_path)
    image_tensor = preprocess_image(image)
    dreamed = dream(image_tensor, model)
    result = deprocess_image(dreamed)

    # ✅ 修复 PyCharm 画图插件 bug 的唯一方法
    import PIL.Image
    pil_img = PIL.Image.fromarray((result * 255).astype(np.uint8))
    pil_img.show()  # 用系统图片查看器弹窗，完美兼容 PyCharm

if __name__ == '__main__':
    main("川大.jpg")