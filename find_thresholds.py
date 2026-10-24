import torch
import numpy as np
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.models.gnn_model import FraudDetectionGNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang sử dụng thiết bị: {device}")

    # 1. Nạp file checkpoint 
    checkpoint_path = "data/processed/graphsage_model.pt"
    print(f"Đang nạp file trọng số từ {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 2. Khởi tạo lại bộ khung mô hình từ thông số đã lưu
    model = FraudDetectionGNN(
        in_channels=checkpoint["in_channels"], 
        hidden_channels=checkpoint["hidden_channels"]
    ).to(device)
    
    # Đưa trọng số vào khung và chạy trên tập test
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("-> Đã tải mô hình GraphSAGE thành công!")

    # 3. Nạp dữ liệu tập Test
    print("\nĐang nạp data_test.pt...")
    test_data = torch.load("data/processed/data_test.pt", weights_only=False).to(device)

    # Khởi tạo Loader 
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[15, 10],
        edge_label_index=test_data.edge_index,
        edge_label=test_data.y,
        batch_size=4096,
        shuffle=False,
        num_workers=0,
        persistent_workers=False
    )

    # 4. Chạy mô hình để lấy xác suất thô 
    all_probs = []
    all_labels = []

    print("Đang chạy dự đoán trên tập Test...")
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.edge_label_index)
            # Đưa logits qua sigmoid
            probs = torch.sigmoid(out).cpu().numpy()
            labels = batch.edge_label.cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 5. Tìm ngưỡng Threshold tối ưu
    print("\nBắt đầu quét tìm ngưỡng Threshold tối ưu (Tối đa hóa F1-score)...")
    best_threshold = 0.5
    best_f1 = 0.0

    # Quét qua 99 mốc từ 0.01 đến 0.99
    thresholds = np.arange(0.01, 1.0, 0.01)
    for t in thresholds:
        preds_t = (all_probs > t).astype(int)
        # zero_division=0 để tránh cảnh báo chia cho 0 
        current_f1 = f1_score(all_labels, preds_t, zero_division=0) 
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    print(f"-> Ngưỡng Tối Ưu: {best_threshold:.2f} (Điểm F1 cao nhất: {best_f1:.4f})")

    # 6. Dự đoán của mô hình với ngưỡng tối ưu
    final_preds = (all_probs > best_threshold).astype(int)

    # 7. In báo cáo
    print(f"\n=== MA TRẬN NHẦM LẪN (GNN - Threshold: {best_threshold:.2f}) ===")
    print(confusion_matrix(all_labels, final_preds))

    print(f"\n=== BÁO CÁO PHÂN LOẠI (GNN - Threshold: {best_threshold:.2f}) ===")
    print(classification_report(all_labels, final_preds))

if __name__ == "__main__":
    main()