# AI-Based Fraud and Money Laundering Detection System
### Hệ thống Phát hiện Gian lận và Rửa tiền (AML) Dựa trên Học sâu Đồ thị (GraphSAGE & GNNExplainer)




---

## 📑 Mục lục (Table of Contents)
1. [Tổng quan Dự án (Project Overview)](#1-tổng-quan-dự-án-project-overview)
2. [Kiến trúc Hệ thống Toàn trình (System Architecture)](#2-kiến-trúc-hệ-thống-toàn-trình-system-architecture)
3. [Cấu trúc Thư mục Dự án (Project Structure)](#3-cấu-trúc-thư-mục-dự-án-project-structure)
4. [Yêu cầu Môi trường & Cài đặt (Prerequisites & Installation)](#4-yêu-cầu-môi-trường--cài-đặt-prerequisites--installation)
5. [Hướng dẫn Luồng Lệnh CLI Toàn trình (End-to-End CLI Pipeline)](#5-hướng-dẫn-luồng-lệnh-cli-toàn-trình-end-to-end-cli-pipeline)
   - [Bước 1: Tiền xử lý dữ liệu sạch & Causal Features](#bước-1-tiền-xử-lý-dữ-liệu-sạch--causal-features)
   - [Bước 2: Xây dựng đồ thị mở Inductive Graph](#bước-2-xây-dựng-đồ-thị-mở-inductive-graph)
   - [Bước 3: Huấn luyện Baseline XGBoost & GraphSAGE](#bước-3-huấn-luyện-baseline-xgboost--graphsage)
   - [Bước 4: Đánh giá & So sánh Đối chuẩn Mô hình](#bước-4-đánh-giá--so-sánh-đối-chuẩn-mô-hình)
   - [Bước 5: Trích xuất Subgraph & Giải thích Mô hình XAI](#bước-5-trích-xuất-subgraph--giải-thích-mô-hình-xai)
   - [Bước 6: Đóng băng Trọng số Mô hình (Model Freezing)](#bước-6-đóng-băng-trọng-số-mô-hình-model-freezing)
   - [Bước 7: Mô phỏng Suy luận Streaming Thời gian thực](#bước-7-mô-phỏng-suy-luận-streaming-thời-gian-thực)
   - [Bước 8: Chạy Toàn bộ Kiểm thử Tự động (Unit Tests)](#bước-8-chạy-toàn-bộ-kiểm-thử-tự-động-unit-tests)
6. [Kết quả Thực nghiệm & Đối chuẩn (Experimental Benchmark)](#6-kết-quả-thực-nghiệm--đối-chuẩn-experimental-benchmark)
7. [Phân tích Case Study Rửa tiền Thực tế (Real-world AML Case Studies)](#7-phân-tích-case-study-rửa-tiền-thực-tế-real-world-aml-case-studies)
8. [Phân công Nhiệm vụ & Vai trò Thành viên (Team Roles)](#8-phân-công-nhiệm-vụ--vai-trò-thành-viên-team-roles)

---

## 1. Tổng quan Dự án (Project Overview)

Rửa tiền (Money Laundering) và gian lận tài chính là các hành vi tội phạm có tổ chức ngày càng tinh vi, sử dụng mạng lưới tài khoản trung gian (*mule accounts*), kỹ thuật phân tán dòng tiền (*smurfing, layering*) và chuyển tiền xoay vòng (*cycle transfers*) để che giấu nguồn gốc phi pháp.

### Thách thức chính:
* **Mất cân bằng nhãn cực đoan (*Extreme Imbalance*):** Trên tập dữ liệu giao dịch tài chính chuẩn **SAML-D** (~9.5 triệu giao dịch), số ca gian lận/rửa tiền chỉ chiếm khoảng **~0.11%** (trong 1.9 triệu giao dịch kiểm tra chỉ có 2,143 ca vi phạm).
* **Bế tắc của Machine Learning truyền thống (Tabular ML):** Các thuật toán phân loại dạng bảng như XGBoost xét từng giao dịch độc lập, không nắm bắt được ngữ cảnh liên kết mạng lưới, dẫn đến tỷ lệ **báo động giả khổng lồ (*Alert Fatigue*)** — trong thử nghiệm thực tế, XGBoost tạo ra tới **26,324 cảnh báo sai** (Precision chỉ đạt 5.95%).

### Giải pháp Đề xuất của Nhóm:
1. **Mô hình hóa Đồ thị Động (*Inductive Graph Learning*):** Biến bài toán thành **Dự đoán nhãn cạnh nhị phân (*Binary Edge Classification*)** trên PyTorch Geometric: Node là tài khoản ngân hàng, Edge là giao dịch chuyển tiền.
2. **Kiến trúc GraphSAGE + Focal Loss:** 2 lớp `SAGEConv` gom tụ thông tin láng giềng trong bán kính 2 bước (*2-hop context*), kết hợp hàm mất mát **Focal Loss** ($\alpha = 0.75, \gamma = 2.0$) triệt tiêu ảnh hưởng của các mẫu âm tính dễ đoán.
3. **Giải thích Mô hình Minh bạch (*Explainable AI - XAI*):** Tích hợp **GNNExplainer** và **NetworkX Visualizer** để bóc tách đồ thị con cục bộ, làm nổi bật đường dây dòng tiền và mã lý do cảnh báo (*Reason Codes*) cho điều tra viên.
4. **Suy luận Streaming Thời gian thực (*Low-Latency Real-time Inference*):** Đóng băng trọng số (*Model Freezing*), cache embedding trên RAM và tối ưu hóa thời gian phản hồi ở mức **$< 1\text{ms}$** trên mỗi giao dịch.

---

## 2. Kiến trúc Hệ thống Toàn trình (System Architecture)

```
[Dữ liệu thô SAML-D (CSV)]
           │
           ▼ (1. Tiền xử lý & Causal Feature Engineering)
┌─────────────────────────────────────────────────────────────────────────┐
│ src/data_preprocessing.py                                              │
│  - Chronological Split (80% Train / 20% Test tuyệt đối không rò rỉ)    │
│  - Causal Sliding Window 24h (O(N) Deque): Tần suất, tổng tiền quá khứ │
│  - Cyclical Encoding (Sin/Cos cho hour, day_of_week, month)            │
│  - Zero-Leakage Scaling: MinMaxScaler fit CHỈ TRÊN TẬP TRAIN           │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼ (2. Xây dựng Đồ thị Mở Quy nạp)
┌─────────────────────────────────────────────────────────────────────────┐
│ src/graph_builder.py                                                   │
│  - Node: Tài khoản (N=855,460) | Edge: Giao dịch (E=9.5M)              │
│  - 8 Node Features (In/Out Degree, Avg Amount, Fraud History, New Node)│
│  - Chunked Edge Features (Continuous + Sin/Cos Time + OHE Categorical) │
│  - Lưu: data_train.pt (2.5GB), data_test.pt (650MB), account_mapping.pkl│
└─────────────────────────────────────────────────────────────────────────┘
           │
           ├───► (3a. Huấn luyện Baseline)
           │      src/models/baseline.py ──> XGBoost (scale_pos_weight=100)
           │
           ├───► (3b. Huấn luyện GraphSAGE + Focal Loss)
           │      src/data_loader.py ──> LinkNeighborLoader (Mini-batch Anti-OOM)
           │      train.py           ──> FraudDetectionGNN + FocalLoss (α=0.75, γ=2.0)
           │
           ├───► (4. Đánh giá Đối chuẩn)
           │      evaluate.py ──> So sánh GraphSAGE vs XGBoost (results/model_comparison.csv)
           │
           ├───► (5. Giải thích XAI với GNNExplainer & NetworkX)
           │      scripts/extract_fraud_subgraphs.py ──> Trích xuất 2-hop Subgraphs
           │      scripts/run_gnn_explainer.py       ──> Sinh edge_mask & node_mask
           │      scripts/visualize_subgraphs.py     ──> Xuất ảnh đồ thị PNG (figures_xai/)
           │
           └───► (6. Đóng băng & Suy luận Streaming Thời gian thực)
                  scripts/freeze_model.py        ──> Xuất graphsage_frozen.pt (requires_grad=False)
                  scripts/simulate_realtime.py   ──> Chấm điểm luồng JSON (Latency < 1ms/tx)
```

---

## 3. Cấu trúc Thư mục Dự án (Project Structure)

```text
AI-Based-Fraud-and-Money-Laundering-Detection-System/
├── config.yaml                    # File cấu hình tên cột, đường dẫn và tỷ lệ chia train/test
├── requirements.txt               # Danh sách thư viện phụ thuộc chuẩn hóa
├── README.md                      # Tài liệu hướng dẫn kỹ thuật toàn trình
├── train.py                       # Pipeline huấn luyện mô hình GraphSAGE với Focal Loss
├── evaluate.py                    # Script đánh giá và so sánh đối chuẩn GraphSAGE vs XGBoost
│
├── data/                          # Thư mục lưu trữ dữ liệu (được quản lý bởi .gitignore)
│   ├── raw/                       # File dữ liệu thô (SAML-D.csv)
│   └── processed/                 # Dữ liệu sạch, checkpoint và artifacts sinh ra
│       ├── train_cleaned.csv      # 80% dữ liệu Train dạng bảng sạch
│       ├── test_cleaned.csv       # 20% dữ liệu Test dạng bảng sạch
│       ├── data_train.pt          # Đồ thị PyTorch Geometric tập Train (2.5 GB)
│       ├── data_test.pt           # Đồ thị PyTorch Geometric tập Test (650 MB)
│       ├── account_mapping.pkl    # Từ điển ánh xạ ID tài khoản gốc hai chiều
│       ├── edge_encoders.pkl      # Bộ OneHotEncoder cho đặc trưng cạnh
│       ├── scalers.pkl            # Bộ MinMaxScaler cho đặc trưng liên tục
│       ├── graphsage_model.pt     # Checkpoint GraphSAGE sau huấn luyện
│       ├── graphsage_frozen.pt    # Checkpoint GraphSAGE đóng băng phục vụ inference
│       ├── xgboost_baseline.json  # Checkpoint mô hình XGBoost Baseline
│       ├── subgraphs/             # Thư mục lưu các đồ thị con (subgraphs .pt)
│       ├── xai_explanations/      # Thư mục lưu kết quả giải thích (explanation .pt & manifest.csv)
│       └── figures_xai/           # Ảnh trực quan hóa đồ thị con NetworkX (.png)
│
├── src/                           # Mã nguồn cốt lõi (Core Source Code)
│   ├── data_loader.py             # Bộ nạp mini-batch LinkNeighborLoader chống OOM
│   ├── data_preprocessing.py      # Tiền xử lý, lọc nhiễu, Causal 24h Window & Zero-Leakage
│   ├── graph_builder.py           # Chuyển đổi dữ liệu bảng sang đồ thị mở Inductive PyG
│   ├── graph_visualizer.py        # Bộ ánh xạ ngược ID và vẽ đồ thị mạng lưới NetworkX
│   ├── gnn_explainer.py           # Tích hợp thuật toán PyG GNNExplainer cho Link Classification
│   ├── subgraph_extractor.py      # Trích xuất đồ thị con 2-hop ego-subgraph xung quanh cạnh nghi vấn
│   ├── analysis/
│   │   └── eda.py                 # Khám phá và phân tích thống kê dữ liệu ban đầu (EDA)
│   ├── losses/
│   │   └── focal_loss.py          # Cài đặt hàm mất mát Focal Loss cho phân loại nhị phân lệch nhãn
│   ├── models/
│   │   ├── baseline.py            # Huấn luyện và đánh giá mô hình đối chuẩn XGBoost
│   │   └── gnn_model.py           # Kiến trúc FraudDetectionGNN (GraphSAGE 2-layer + Linear Classifier)
│   └── utils/
│       ├── __init__.py
│       └── graph_utils.py         # Tiện ích đồ thị hiệu năng cao: Caching ID, Cold-start, Edge Embeddings
│
├── scripts/                       # Các kịch bản thực thi CLI tự động hóa
│   ├── audit_data_leakage.py      # Kiểm toán chống rò rỉ dữ liệu (thời gian, scaler, target)
│   ├── extract_fraud_subgraphs.py # Script trích xuất hàng loạt đồ thị con từ tập Test
│   ├── freeze_model.py            # Script đóng băng trọng số mô hình và đo benchmark độ trễ
│   ├── run_gnn_explainer.py       # Script chạy GNNExplainer hàng loạt cho các đồ thị con
│   ├── simulate_realtime.py       # Mô phỏng luồng suy luận Streaming thời gian thực từ CSV
│   └── visualize_subgraphs.py     # Script xuất hàng loạt ảnh trực quan hóa NetworkX kèm chú giải XAI
│
├── results/                       # Báo cáo kết quả thực nghiệm
│   ├── model_comparison.csv       # Bảng số liệu so sánh GraphSAGE vs XGBoost
│   ├── model_comparison.json      # Báo cáo chi tiết metrics, confusion matrix, classification report
│   ├── graphsage_confusion_matrix.csv # Ma trận nhầm lẫn của GraphSAGE
│   └── xgboost_confusion_matrix.csv   # Ma trận nhầm lẫn của XGBoost
│
├── docs/                          # Tài liệu kỹ thuật chuyên sâu
│   └── xai_gnnexplainer.md        # Hướng dẫn kỹ thuật về GNNExplainer, edge_mask và node_mask
│
└── tests/                         # Bộ kiểm thử tự động (Unit Tests)
    ├── test_graph_utils.py        # Kiểm thử tiện ích đồ thị và xử lý tài khoản mới
    ├── test_model_freezing.py     # Kiểm thử đóng băng mô hình và tính toàn vẹn số học
    ├── test_subgraph_extractor.py # Kiểm thử lọc cạnh gian lận và trích xuất subgraph
    └── test_gnn_explainer.py      # Kiểm thử thuật toán tối ưu mask GNNExplainer
```

---

## 4. Yêu cầu Môi trường & Cài đặt (Prerequisites & Installation)

### Yêu cầu Hệ thống:
* **Hệ điều hành:** Windows 10/11, Ubuntu 20.04+, hoặc macOS.
* **Python:** Phiên bản khuyến nghị `Python 3.10` đến `Python 3.14`.
* **Phần cứng:** Tối thiểu 16GB RAM (khuyến nghị 32GB RAM khi build đồ thị 9.5M cạnh). GPU NVIDIA CUDA (tùy chọn, hệ thống hỗ trợ CPU mượt mà nhờ mini-batch sampling).

### Các bước cài đặt:

1. **Clone repository:**
   ```bash
   git clone https://github.com/vinh-2k6/AI-Based-Fraud-and-Money-Laundering-Detection-System.git
   cd AI-Based-Fraud-and-Money-Laundering-Detection-System
   ```

2. **Khởi tạo môi trường ảo (Virtual Environment):**
   * *Trên Windows (PowerShell):*
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * *Trên Linux / macOS:*
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Cài đặt các gói phụ thuộc từ `requirements.txt`:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > **Lưu ý về PyTorch Geometric:** File `requirements.txt` đã cấu hình cờ `--find-links` tải sẵn wheel CPU tương thích với `torch==2.14.0`, đảm bảo cài đặt thành công 100% trên Windows mà không yêu cầu trình biên dịch C++ Visual Studio.

---

## 5. Hướng dẫn Luồng Lệnh CLI Toàn trình (End-to-End CLI Pipeline)

Dưới đây là chuỗi lệnh CLI hoàn chỉnh để tái hiện toàn bộ dự án từ dữ liệu thô ban đầu:

### Bước 1: Tiền xử lý dữ liệu sạch & Causal Features
Làm sạch dữ liệu thô `data/raw/SAML-D.csv`, sắp xếp trật tự thời gian tuyệt đối, tính toán 4 đặc trưng thời gian cửa sổ trượt 24h (`sender_tx_count_24h`, `receiver_tx_count_24h`, `sender_amount_sum_24h`, `time_since_last_sender_tx`), chia tập 80/20 theo thời gian và chuẩn hóa Zero-Leakage:
```powershell
python src/data_preprocessing.py
```
*Kiểm toán an toàn chống rò rỉ dữ liệu:*
```powershell
python scripts/audit_data_leakage.py
```
*(Kết quả mong đợi: `[PASS]` 100% các tiêu chí về ranh giới thời gian, tính toàn vẹn dữ liệu và scaler leakage).*

---

### Bước 2: Xây dựng đồ thị mở Inductive Graph
Ánh xạ các tài khoản ngân hàng thành các đỉnh (Nodes), chuyển đổi giao dịch thành các cạnh (Edges), mã hóa chu kỳ thời gian $\sin/\cos$, xử lý One-Hot Categorical theo chunk chống tràn RAM và xuất ra file đồ thị PyG:
```powershell
python src/graph_builder.py
```
*Đầu ra sinh ra tại `data/processed/`: `data_train.pt`, `data_test.pt`, `account_mapping.pkl`, `scalers.pkl`, `edge_encoders.pkl`.*

---

### Bước 3: Huấn luyện Baseline XGBoost & GraphSAGE
1. **Huấn luyện mô hình đối chuẩn XGBoost dạng bảng:**
   ```powershell
   python src/models/baseline.py
   ```
   *Lưu mô hình tại: `data/processed/xgboost_baseline.json`.*

2. **Huấn luyện mô hình Học sâu Đồ thị GraphSAGE với Focal Loss:**
   ```powershell
   python train.py
   ```
   *Sử dụng `LinkNeighborLoader` lấy mẫu láng giềng `[10, 10]`, hàm mất mát `FocalLoss(alpha=0.75, gamma=2.0)`. Lưu checkpoint tại: `data/processed/graphsage_model.pt`.*

---

### Bước 4: Đánh giá & So sánh Đối chuẩn Mô hình
Chạy đánh giá đồng thời cả hai mô hình trên cùng tập kiểm tra (1,900,971 giao dịch), tính toán Precision, Recall, F1, PR-AUC, Confusion Matrix và xuất báo cáo so sánh:
```powershell
python evaluate.py --threshold 0.5
```
*Kết quả xuất ra tại: `results/model_comparison.csv` và `results/model_comparison.json`.*

---

### Bước 5: Trích xuất Subgraph & Giải thích Mô hình XAI
1. **Trích xuất các đồ thị con 2-hop ego-subgraph xung quanh các giao dịch gian lận:**
   ```powershell
   python scripts/extract_fraud_subgraphs.py --limit 100 --mode ground_truth
   ```
   *Lưu các đồ thị con tại: `data/processed/subgraphs/subgraph_edge_*.pt`.*

2. **Chạy thuật toán GNNExplainer để tối ưu hóa `edge_mask` và `node_mask`:**
   ```powershell
   python scripts/run_gnn_explainer.py --checkpoint data/processed/graphsage_model.pt --epochs 200 --device auto
   ```
   *Lưu các artifact giải thích tại: `data/processed/xai_explanations/explanation_*.pt` và `manifest.csv`.*

3. **Trực quan hóa mạng lưới đồ thị giải thích với NetworkX:**
   ```powershell
   python scripts/visualize_subgraphs.py --threshold 0.5
   ```
   *Xuất toàn bộ ảnh PNG độ nét cao (DPI 300) làm nổi bật các cạnh và nút trọng số cao tại: `data/processed/figures_xai/`.*

---

### Bước 6: Đóng băng Trọng số Mô hình (Model Freezing)
Khóa vĩnh viễn gradient của 9,473 tham số (`requires_grad = False`), kiểm tra xác thực sai lệch số học bằng 0 và đo kiểm tra Benchmark độ trễ forward-pass:
```powershell
python scripts/freeze_model.py --checkpoint data/processed/graphsage_model.pt --output data/processed/graphsage_frozen.pt --benchmark-runs 1000
```
*Lưu artifact siêu nhẹ phục vụ production tại: `data/processed/graphsage_frozen.pt`.*

---

### Bước 7: Mô phỏng Suy luận Streaming Thời gian thực
Mô phỏng máy chủ ngân hàng tiếp nhận liên tục các gói tin JSON giao dịch mới, tra cứu embedding đồ thị nền và chấm điểm gian lận tức thì (kèm kích hoạt XAI on-the-fly nếu phát hiện gian lận):
```powershell
python scripts/simulate_realtime.py --limit 100 --verbose
```
*(Kết quả mong đợi: Thời gian phản hồi ổn định từ giao dịch thứ 2 trở đi chỉ dao động **0.37ms - 2.12ms**, vượt xa yêu cầu < 100ms).*

---

### Bước 8: Chạy Toàn bộ Kiểm thử Tự động (Unit Tests)
Kiểm thử toàn diện 4 module trọng yếu (tiện ích đồ thị, đóng băng mô hình, trích xuất subgraph, và GNNExplainer):
```powershell
python -m unittest discover tests
```
*(Kết quả mong đợi: `Ran 14 tests in ~2.1s - OK 100% Passed`).*

---

## 6. Kết quả Thực nghiệm & Đối chuẩn (Experimental Benchmark)

Đánh giá thực nghiệm trên cùng tập kiểm tra **1,900,971 giao dịch** thuộc tập dữ liệu **SAML-D** (chứa 2,143 ca rửa tiền thực tế):

### 1. Bảng So sánh Hiệu năng Phát hiện Gian lận (Class 1 = Fraud/AML):

| Chỉ số Đánh giá (Metrics) | XGBoost (Baseline) | GraphSAGE + Focal Loss (Đề xuất) | Chênh lệch / Ý nghĩa Thực tiễn |
| :--- | :---: | :---: | :--- |
| **Recall (Tỷ lệ bắt gian lận)** | **77.69%** (1,665 / 2,143 ca) | **76.34%** (1,636 / 2,143 ca) | Duy trì khả năng phát hiện tương đương (~3/4 số vụ) |
| **Precision (Độ chính xác cảnh báo)** | 5.95% | **31.27%** | **Tăng gấp 5.25 lần** |
| **Số ca báo động giả (False Positives)**| 26,324 ca | **3,596 ca** | **Giảm hơn 7.3 lần khối lượng kiểm tra sai** |
| **F1-Score (Cân bằng P-R)** | 0.1105 | **0.4437** | **Tăng gấp 4 lần (+0.3332)** |
| **PR-AUC (Area Under PR Curve)** | 0.5402 | **0.7545** | **Tăng +0.2143** (vượt trội trên dữ liệu lệch nhãn) |
| **Accuracy (Độ chính xác toàn thể)** | 98.59% | **99.78%** | Tăng +1.19% |

> **Nhận định then chốt:** Nhờ nắm bắt cấu trúc mạng lưới lan truyền và luân chuyển qua 2 bước láng giềng, **GraphSAGE đã triệt tiêu được 22,728 ca báo động sai** so với XGBoost, giải quyết triệt để vấn đề "ngập lụt cảnh báo" (*Alert Fatigue*) gây quá tải cho các chuyên viên tuân thủ phòng chống rửa tiền trong ngân hàng.

### 2. Đo kiểm Độ trễ Suy luận (Latency Benchmark):
* **Thời gian Forward-pass mô hình đóng băng:**
  * Trung bình (*Mean Latency*): **0.4314 ms**
  * Phân vị 95 (*P95 Latency*): **0.5185 ms**
  * Phân vị 99 (*P99 Latency*): **0.6224 ms**
* **Độ trễ toàn trình trong luồng Streaming (*End-to-End Streaming Latency*):**
  * Giao dịch khởi tạo đầu tiên (Nạp Base Graph vào bộ nhớ): ~7.8s
  * Các giao dịch tiếp theo (*Steady-State Inference*): **0.37 ms – 2.12 ms / giao dịch**
  * **Đạt chuẩn:** Vượt xa mục tiêu đề ra cho hệ thống thời gian thực ($< 100\text{ms}$).

---

## 6.1. Reason Code và báo cáo Gemini tùy chọn

Sau khi chạy GNNExplainer, hệ thống xuất reason code JSON cục bộ:

```powershell
python scripts\run_gnn_explainer.py `
  --checkpoint data\processed\graphsage_model.pt `
  --subgraphs-dir data\processed\subgraphs `
  --output-dir data\processed\xai_explanations
```

Reason code được lưu tại `data\processed\reason_codes\transaction_<id>.json`.
Các mã này do `ReasonCodeGenerator` sinh từ xác suất model, metadata graph và
mask của GNNExplainer; chúng không phải kết luận pháp lý.

Gemini chỉ được gọi khi chạy script riêng:

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:GEMINI_MODEL = "gemini-3.6-flash"
python scripts\generate_transaction_reports.py `
  --input-dir data\processed\reason_codes `
  --output-dir data\processed\transaction_reports `
  --top-k 5 `
  --max-prompt-chars 12000
```

Có thể dùng `--force` để bỏ qua cache. Mặc định pipeline GraphSAGE/XAI không
gọi Gemini. Payload gửi ra ngoài chỉ chứa dữ liệu đã lọc, giới hạn kích thước
và ẩn danh account/transaction ID; không gửi tensor hoặc toàn bộ graph. Không
đặt API key trong source code, commit hoặc log. Cần đánh giá quyền riêng tư,
chi phí API và phê duyệt việc gửi dữ liệu tới dịch vụ bên ngoài trước khi dùng
production.

Nếu Gemini lỗi hoặc thiếu `GEMINI_API_KEY`, reason code gốc vẫn được giữ lại và
batch tiếp tục xử lý file khác.

## 7. Phân tích Case Study Rửa tiền Thực tế (Real-world AML Case Studies)

Dưới đây là 3 mẫu đồ thị con điển hình được trích xuất và giải thích trực quan bởi hệ thống:

### Case Study 1: Mạng lưới rửa tiền vòng lặp khép kín (*Collusion / Dense Ring*)
* **File minh họa:** [`data/processed/figures_xai/subgraph_edge_1736_xai.png`](data/processed/figures_xai/subgraph_edge_1736_xai.png)
* **Giao dịch mục tiêu:** Tài khoản `8856680971` $\to$ `7248487713`.
* **Quy mô đồ thị con:** 5 tài khoản, 20 giao dịch liên kết. Đáng chú ý: **5/20 giao dịch (25%) bị gắn cờ rửa tiền**.
* **Phân tích XAI:** Dự đoán xác suất $P = 64.16\%$. `GNNExplainer` gán trọng số cạnh tối đa **$0.8060$** và độ quan trọng nút **$0.3649$**, làm nổi bật các giao dịch xoay vòng qua lại giữa nhóm 5 tài khoản cấu kết nhằm làm mờ dấu vết dòng tiền gốc.

### Case Study 2: Chuỗi chuyển tiền phân tầng qua tài khoản đệm (*Layering Transfer*)
* **File minh họa:** [`data/processed/figures_xai/subgraph_edge_2132_xai.png`](data/processed/figures_xai/subgraph_edge_2132_xai.png)
* **Giao dịch mục tiêu:** Tài khoản `3130936563` $\to$ `9140659953`.
* **Quy mô đồ thị con:** 25 tài khoản, 60 giao dịch. Dòng tiền được xé nhỏ chuyển qua các tài khoản trung chuyển (*mule accounts* có bậc luân chuyển cao `degree > 4`).
* **Phân tích XAI:** Dự đoán xác suất gian lận $P \approx 100.00\%$, Max edge weight = **$0.7177$**. Các cạnh dẫn tiền từ tài khoản đệm được tô màu đậm nét, chứng minh dòng tiền không đi thẳng mà được phân tầng (*layering*) có chủ đích.

### Case Study 3: Rửa tiền quy mô lớn lồng ghép giao dịch sạch (*Large-Scale Smurfing*)
* **File minh họa:** [`data/processed/figures_xai/subgraph_edge_1317_xai.png`](data/processed/figures_xai/subgraph_edge_1317_xai.png)
* **Giao dịch mục tiêu:** Tài khoản `5565879161` $\to$ `4497771501`.
* **Quy mô đồ thị con:** 140 tài khoản, 337 giao dịch, chứa 12 giao dịch gian lận ẩn giấu trong mạng lưới giao dịch thông thường khổng lồ.
* **Phân tích XAI:** Xác suất gian lận $P = 99.99\%$. Biểu đồ bóc tách sắc bén: các giao dịch thông thường được làm mờ (xám nhạt), trong khi các cạnh thuộc đường dây chuyển tiền bẩn được `GNNExplainer` gán trọng số $>0.72$ và làm nổi bật màu đỏ rực, giúp điều tra viên định vị chính xác đường dây ngầm.

---

## 8. Phân công Nhiệm vụ & Vai trò Thành viên (Team Roles)

Dự án được xây dựng và phối hợp hoàn thiện qua 6 tuần bởi 4 thành viên:

| Thành viên | Vai trò Kỹ thuật | Trách nhiệm & Đóng góp Cốt lõi |
| :--- | :--- | :--- |
| **Đào Trọng VInh** (`24021677`) | **Data Engineer** | • Tiền xử lý dữ liệu SAML-D, giải thuật cửa sổ trượt 24h Causal Sliding Window $O(N)$ chống rò rỉ.<br>• Script trích xuất đồ thị con 2-hop ego-subgraph (`subgraph_extractor.py`).<br>• Đóng gói quy trình tạo tensor và đặc trưng phục vụ luồng streaming. |
| **Lê Tùng Dương** (`24021438`) | **Graph Engineer** | • Thiết kế đồ thị quy nạp Inductive Graph (`graph_builder.py`) và bộ nạp mini-batch `data_loader.py`.<br>• Module ánh xạ ngược ID tài khoản và trực quan hóa NetworkX (`graph_visualizer.py`, `visualize_subgraphs.py`).<br>• Đóng băng mô hình (`freeze_model.py`), tối ưu hóa tiện ích đồ thị (`graph_utils.py`), chuẩn hóa `requirements.txt` và `README.md`. |
| **Nguyễn Mạnh Cường** (`24021398`) | **AI Architect** | • Thiết kế kiến trúc `FraudDetectionGNN` (GraphSAGE 2 lớp) và cài đặt hàm mất mát `FocalLoss`.<br>• Tích hợp thuật toán `GNNExplainer` cho bài toán Link Classification, tối ưu hóa `edge_mask` và `node_mask`.<br>• Tinh chỉnh siêu tham số và phân tích đối chiếu với các mô hình/bằng sáng chế thực tế. |
| **Lưu Việt Dũng** (`24021430`) | **MLOps & Reporter** | • Xây dựng hệ thống suy luận thời gian thực (`simulate_realtime.py`) tích hợp XAI on-the-fly.<br>• Xây dựng kịch bản kiểm thử đánh giá đối chuẩn (`evaluate.py`), đo kiểm độ trễ toàn trình End-to-End.<br>• Tổng hợp số liệu, xuất báo cáo kỹ thuật và slide bảo vệ dự án. |

---

