# import torch
# from torch_geometric.loader import LinkNeighborLoader
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np


# from src.models.gnn_model import FraudDetectionGNN
# from src.losses.focal_loss import FocalLoss

# def train():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Đang sử dụng thiết bị: {device}")

#     # 1. Nạp dữ liệu đồ thị 
#     print("Đang nạp data_train.pt và data_test.pt...")
#     train_data = torch.load("data/processed/data_train.pt", weights_only=False).to(device)
#     test_data = torch.load("data/processed/data_test.pt", weights_only=False).to(device)


#     # 2. Khởi tạo DataLoader lấy mẫu láng giềng theo cạnh (Link Sampling)
#     # Lấy mẫu 15 láng giềng ở bậc 1 và 10 láng giềng ở bậc 2 cho mỗi node
#     train_loader = LinkNeighborLoader(
#         data=train_data,
#         num_neighbors=[15,10],
#         edge_label_index=train_data.edge_index,
#         edge_label=train_data.y,
#         batch_size=4096,
#         shuffle=True,
#     )

#     test_loader = LinkNeighborLoader(
#         data=test_data,
#         num_neighbors=[15, 10],
#         edge_label_index=test_data.edge_index,
#         edge_label=test_data.y,
#         batch_size=4096,
#         shuffle=False,
#     )

#     # 3. Khởi tạo Mô hình, Hàm Loss và Bộ tối ưu hóa
#     # in_channels  lấy từ số chiều đặc trưng của node 
#     model = FraudDetectionGNN(in_channels=train_data.x.shape[1], hidden_channels=64).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
#     # Sử dụng Focal Loss với trọng số alpha cao để tập trung vào lớp gian lận (1)
#     criterion = FocalLoss(alpha=0.75, gamma=2.0)

#     # 4. Vòng lặp huấn luyện 
#     epochs = 5
#     for epoch in range(1, epochs + 1):
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             optimizer.zero_grad()
            
#             # Mô hình dự đoán dựa trên thông tin láng giềng
#             out = model(batch.x, batch.edge_index, batch.edge_label_index)
            
#             # Tính toán Focal Loss
#             loss = criterion(out, batch.edge_label.float())
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item() * batch.edge_label_index.size(1)
            
#         print(f"Epoch {epoch}/{epochs} - Loss: {total_loss / train_data.edge_index.size(1):.4f}")

#     # 5. Đánh giá trên tập Test
#     print("\nĐang đánh giá mô hình trên tập Test...")
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in test_loader:
#             out = model(batch.x, batch.edge_index, batch.edge_label_index)
#             # Dùng hàm sigmoid để chuyển logit thành xác suất, ngưỡng 0.5
#             probs = torch.sigmoid(out)
#             preds = (probs > 0.5).long().cpu().numpy()
#             labels = batch.edge_label.cpu().numpy()
            
#             all_preds.extend(preds)
#             all_labels.extend(labels)

#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)

#     print("\n=== MA TRẬN NHẦM LẪN (GNN) ===")
#     print(confusion_matrix(all_labels, all_preds))

#     print("\n=== BÁO CÁO PHÂN LOẠI (GNN) ===")
#     print(classification_report(all_labels, all_preds))

# if __name__ == "__main__":
#     train()





import torch
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.models.gnn_model import FraudDetectionGNN
from src.losses.focal_loss import FocalLoss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang sử dụng thiết bị: {device}")

    # 1. Nạp dữ liệu đồ thị 
    print("Đang nạp data_train.pt và data_test.pt...")
    train_data = torch.load("data/processed/data_train.pt", weights_only=False).to(device)
    test_data = torch.load("data/processed/data_test.pt", weights_only=False).to(device)

    # 2. Khởi tạo DataLoader lấy mẫu láng giềng  cho CPU
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[10, 10],       #  số láng giềng lấy mẫu (tầng 1: 10, tầng 2: 5)
        edge_label_index=train_data.edge_index,
        edge_label=train_data.y,
        batch_size=4096,           
        shuffle=True,
        num_workers=0,             
        persistent_workers=False
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[10, 10],
        edge_label_index=test_data.edge_index,
        edge_label=test_data.y,
        batch_size=4096,
        shuffle=False,
        num_workers=0,
        persistent_workers=False
    )

    # 3. Khởi tạo Mô hình, Hàm Loss và Bộ tối ưu hóa
    model = FraudDetectionGNN(in_channels=train_data.x.shape[1], hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Sử dụng Focal Loss với trọng số alpha cao để tập trung vào lớp gian lận (1)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # 4. Vòng lặp huấn luyện 
    epochs = 12
    max_batches_per_epoch = 300  # giới hạn batch trong mỗi epoch 

    print("\nBắt đầu huấn luyện...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_samples = 0
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Mô hình dự đoán dựa trên thông tin láng giềng
            out = model(batch.x, batch.edge_index, batch.edge_label_index)
            
            # Tính toán Focal Loss
            loss = criterion(out, batch.edge_label.float())
            loss.backward()
            optimizer.step()
            
            batch_size_current = batch.edge_label_index.size(1)
            total_loss += loss.item() * batch_size_current
            total_samples += batch_size_current
            
            if i >= max_batches_per_epoch:
                break
                
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    # 5. Đánh giá trên tập Test
    print("\nĐang đánh giá mô hình trên tập Test...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.edge_label_index)
            # Dùng hàm sigmoid để chuyển logit thành xác suất, ngưỡng 0.5
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).long().cpu().numpy()
            labels = batch.edge_label.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n=== MA TRẬN NHẦM LẪN (GNN) ===")
    print(confusion_matrix(all_labels, all_preds))

    print("\n=== BÁO CÁO PHÂN LOẠI (GNN) ===")
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    train()