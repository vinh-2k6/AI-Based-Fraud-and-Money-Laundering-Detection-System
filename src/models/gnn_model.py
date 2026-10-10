import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class FraudDetectionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(FraudDetectionGNN, self).__init__()
        # 2 lớp SAGEConv để gom thông tin láng giềng trong bán kính 2 bước
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Linear layer cuối: Nhận đầu vào gấp đôi vì ghép vector (Người gửi + Người nhận)
        self.classifier = nn.Linear(hidden_channels * 2, 1)

    def forward(self, x, edge_index, edge_label_index):
        # 1. Cập nhật nhúng cho toàn bộ tài khoản (Nodes)
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()

        # 2. Rút trích vector của hai đầu mút giao dịch
        src_nodes = edge_label_index[0] # Chỉ số các tài khoản gửi
        dst_nodes = edge_label_index[1] # Chỉ số các tài khoản nhận
        
        # 3. Ghép nối để tạo đặc trưng cho giao dịch (Edge)
        edge_feat = torch.cat([h[src_nodes], h[dst_nodes]], dim=-1)

        # 4. Xuất logit dự đoán
        return self.classifier(edge_feat).squeeze()