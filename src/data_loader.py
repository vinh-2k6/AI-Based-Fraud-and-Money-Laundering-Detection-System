"""
src/data_loader.py
Module cấp phát mini-batch đồ thị chống tràn bộ nhớ (Anti-OOM)
"""
from pathlib import Path
import torch
from torch_geometric.loader import LinkNeighborLoader

def get_link_loaders(
    train_path: str = "data/processed/data_train.pt",
    test_path: str = "data/processed/data_test.pt",
    batch_size: int = 4096,
    num_neighbors: list[int] = [15, 10],
):
    print("=" * 60)
    print("Graph Engineer: Đang nạp đồ thị vào CPU RAM (Tránh tràn VRAM)...")
    print("=" * 60)

    # 1. NẠP DỮ LIỆU Ở CPU (Tuyệt đối KHÔNG gọi .to(device) ở đây)
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)

    print(f"Train edges: {train_data.edge_index.size(1):,} | Nodes: {train_data.num_nodes:,}")
    print(f"Test edges : {test_data.edge_index.size(1):,} | Nodes: {test_data.num_nodes:,}")

    # 2. KHỞI TẠO LOADER LẤY MẪU LÁNG GIỀNG CHO TẬP TRAIN
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neighbors,
        edge_label_index=train_data.edge_index,
        edge_label=train_data.y,
        batch_size=batch_size,
        shuffle=True,          # Xáo trộn cạnh trong lúc train
        num_workers=0,          # An toàn tuyệt đối trên Windows
        pin_memory=torch.cuda.is_available(), # Tăng tốc đẩy dữ liệu sang GPU
    )

    # 3. KHỞI TẠO LOADER CHO TẬP TEST
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=num_neighbors,
        edge_label_index=test_data.edge_index,
        edge_label=test_data.y,
        batch_size=batch_size,
        shuffle=False,         # Đánh giá tuần tự
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, train_data.x.shape[1]