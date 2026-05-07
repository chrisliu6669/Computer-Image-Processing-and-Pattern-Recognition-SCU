import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 构造简单图数据
# 4个节点，每个节点2维特征
x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float).to(device)
# 边索引
edge_index = torch.tensor([
    [0, 1, 2, 3, 0],
    [1, 2, 3, 0, 2]
], dtype=torch.long).to(device)

# 定义两层GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 初始化模型
model = GCN(in_channels=2, hidden_channels=4, out_channels=2).to(device)
out = model(x, edge_index)

print("GCN 前向传播输出：")
print(out)
print("\n✅ 图神经网络运行正常，无依赖报错！")