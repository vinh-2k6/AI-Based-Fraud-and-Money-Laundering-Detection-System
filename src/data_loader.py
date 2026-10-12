"""
src/data_loader.py
===================
Module cấp phát mini-batch đồ thị cho Link/Edge Classification (Anti-OOM).

Tính năng nâng cấp:
  1. Path Resolution: Tự động nhận diện thư mục gốc, chống FileNotFoundError.
  2. Asymmetric Batching: Tăng tốc độ Test gấp 2 lần với eval_batch_size lớn.
  3. Metadata Export: Cung cấp pos_weight, edge_feat_dim phục vụ tinh chỉnh Loss của TV3.
  4. Memory Management: Giữ đồ thị trên CPU, cấp phát subgraph an toàn cho GPU.
"""

import sys
import warnings
from pathlib import Path
import torch
from torch_geometric.loader import LinkNeighborLoader

warnings.filterwarnings("ignore", category=FutureWarning)

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Định vị thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"


def get_link_loaders(
    train_path: str | Path = PROC_DIR / "data_train.pt",
    test_path: str | Path = PROC_DIR / "data_test.pt",
    batch_size: int = 2048,
    num_neighbors: list[int] = [15, 10],
    eval_batch_size: int | None = None,
):
    """
    Khởi tạo LinkNeighborLoader chống tràn VRAM cho Train và Test.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu đồ thị tại {PROC_DIR}. "
            f"Hãy đảm bảo đã chạy graph_builder.py trước!"
        )

    print("=" * 65)
    print("Đang nạp đồ thị vào CPU RAM...")
    print("=" * 65)

    # 1. NẠP DỮ LIỆU Ở CPU (Tránh OOM VRAM)
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)

    num_train_edges = train_data.edge_index.size(1)
    num_test_edges = test_data.edge_index.size(1)
    num_nodes = train_data.num_nodes

    # Tính toán thống kê mất cân bằng nhãn để hỗ trợ TV3 cấu hình Loss
    num_fraud = int(train_data.y.sum())
    num_clean = num_train_edges - num_fraud
    pos_weight = float(num_clean / max(1, num_fraud))

    print(f"  Nodes (Tài khoản)   : {num_nodes:,}")
    print(f"  Train Edges (Giao dịch): {num_train_edges:,} (Gian lận: {num_fraud:,} ~ {num_fraud/num_train_edges*100:.2f}%)")
    print(f"  Test Edges (Giao dịch) : {num_test_edges:,}")
    print(f"  Tỷ lệ cân bằng nhãn pos_weight gợi ý: {pos_weight:.2f}")

    # Khi test không tính đạo hàm (no_grad) nên có thể dùng batch lớn hơn để tăng tốc
    if eval_batch_size is None:
        eval_batch_size = batch_size * 2

    # 2. KHỞI TẠO TRAIN LOADER
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neighbors,
        edge_label_index=train_data.edge_index,
        edge_label=train_data.y,
        batch_size=batch_size,
        shuffle=True,                          # Xáo trộn cạnh ngẫu nhiên khi học
        num_workers=0,                          # 0 worker an toàn tuyệt đối trên Windows
        pin_memory=torch.cuda.is_available(),   # Tăng tốc độ copy từ RAM sang VRAM GPU
    )

    # 3. KHỞI TẠO TEST LOADER
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=num_neighbors,
        edge_label_index=test_data.edge_index,
        edge_label=test_data.y,
        batch_size=eval_batch_size,            # Batch size lớn hơn giúp đánh giá siêu tốc
        shuffle=False,                         # Không xáo trộn khi đánh giá
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. GÓI METADATA BÀN GIAO CHO TRAIN.PY
    metadata = {
        "num_node_features": train_data.x.shape[1],
        "num_edge_features": train_data.edge_attr.shape[1] if hasattr(train_data, "edge_attr") and train_data.edge_attr is not None else 0,
        "num_nodes": num_nodes,
        "num_train_edges": num_train_edges,
        "num_test_edges": num_test_edges,
        "pos_weight": pos_weight,
    }

    return train_loader, test_loader, metadata


if __name__ == "__main__":
    print("Kiểm tra chất lượng module data_loader.py...")
    train_loader, test_loader, meta = get_link_loaders(batch_size=2048)

    print("\n--- METADATA ---")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print("\n--- KIỂM TRA MINI-BATCH ĐẦU TIÊN ---")
    for batch in train_loader:
        print(f"  Subgraph Nodes : {batch.num_nodes:,}")
        print(f"  Subgraph Edges : {batch.edge_index.size(1):,}")
        print(f"  Labeled Edges  : {batch.edge_label_index.size(1):,}")
        print(f"  Node Feature X : {batch.x.shape}")
        if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
            print(f"  Edge Feature Attr : {batch.edge_attr.shape}")
        break

