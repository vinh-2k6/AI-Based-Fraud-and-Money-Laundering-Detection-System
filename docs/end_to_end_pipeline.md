# Hướng dẫn chạy pipeline từ dữ liệu đến báo cáo

Tóm tắt luồng chạy:
data/raw/SAML-D.csv
    ↓
src/data_preprocessing.py
    ↓
src/graph_builder.py
    ↓
train.py
    ↓
scripts/extract_fraud_subgraphs.py
    ↓
scripts/run_gnn_explainer.py
    ↓
src/reason_codes.py
    ↓
scripts/generate_transaction_reports.py
    ↓
src/transaction_reporter.py
    ↓
src/gemini_client.py
    ↓
transaction_reports/*.md và *.json

Tài liệu này hướng dẫn chạy toàn bộ pipeline của project trên Windows:

```text
Dữ liệu thô
  -> tiền xử lý và tạo feature
  -> xây dựng graph giao dịch
  -> huấn luyện GraphSAGE
  -> đánh giá model
  -> trích xuất subgraph
  -> GNNExplainer
  -> Reason Code JSON
  -> báo cáo Gemini Markdown/JSON (tùy chọn)
```

Pipeline GraphSAGE/XAI không tự động gọi dịch vụ AI bên ngoài. Gemini chỉ được
gọi ở bước cuối khi chạy riêng `generate_transaction_reports.py`.

## 1. Chuẩn bị môi trường

Mở PowerShell tại thư mục gốc project:

```powershell
cd C:\Users\Hi\Downloads\AI-Based-Fraud-and-Money-Laundering-Detection-System
```

Tạo và kích hoạt virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn việc activate:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Cài dependency:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra Python và package:

```powershell
python --version
python -c "import torch, torch_geometric; print(torch.__version__)"
```

## 2. Chuẩn bị dữ liệu đầu vào

Đặt dữ liệu SAML-D tại:

```text
data/raw/SAML-D.csv
```

Các cột đầu vào chính được cấu hình trong [config.yaml](../config.yaml):

- `step`;
- `type`;
- `amount`;
- `nameOrig`;
- `oldbalanceOrg`;
- `newbalanceOrig`;
- `nameDest`;
- `oldbalanceDest`;
- `newbalanceDest`;
- `isFraud`;
- `isFlaggedFraud`.

Không commit dữ liệu thô, checkpoint hoặc dữ liệu giao dịch vào Git. Các thư
mục dữ liệu lớn đã được loại khỏi Git trong [.gitignore](../.gitignore).

## 3. Tiền xử lý dữ liệu

Chạy:

```powershell
python src\data_preprocessing.py
```

Module [data_preprocessing.py](../src/data_preprocessing.py) thực hiện:

1. Đọc dữ liệu thô.
2. Chuẩn hóa tên và kiểu dữ liệu.
3. Sắp xếp theo thời gian.
4. Tạo các feature cửa sổ trượt 24 giờ:
   - số giao dịch của sender;
   - số giao dịch của receiver;
   - tổng tiền sender đã chuyển;
   - thời gian từ giao dịch trước của sender.
5. Mã hóa thời gian theo chu kỳ.
6. Chia train/test theo thời gian.
7. Fit scaler trên train để tránh data leakage.
8. Ghi dữ liệu đã xử lý.

Các output thường dùng:

```text
data/processed/train.csv
data/processed/test.csv
data/processed/scalers.pkl
data/processed/edge_encoders.pkl
```

Kiểm tra rò rỉ dữ liệu:

```powershell
python scripts\audit_data_leakage.py
```

Chỉ chuyển sang bước tiếp theo khi kiểm toán không phát hiện lỗi về ranh giới
thời gian, target leakage hoặc scaler leakage.

## 4. Xây dựng graph giao dịch

Chạy:

```powershell
python src\graph_builder.py
```

Module [graph_builder.py](../src/graph_builder.py) chuyển dữ liệu bảng thành
PyTorch Geometric graph:

- node là tài khoản;
- edge là giao dịch;
- `x` là node feature;
- `edge_attr` là transaction feature;
- `y` là fraud label.

Các output cần có:

```text
data/processed/data_train.pt
data/processed/data_test.pt
data/processed/account_mapping.pkl
data/processed/scalers.pkl
data/processed/edge_encoders.pkl
```

`account_mapping.pkl` dùng để ánh xạ account ID sang node index và ngược lại.
Không gửi mapping này cho Gemini.

## 5. Huấn luyện model

### 5.1. Model baseline tùy chọn

Huấn luyện XGBoost để so sánh:

```powershell
python src\models\baseline.py
```

Output:

```text
data/processed/xgboost_baseline.json
```

### 5.2. Huấn luyện GraphSAGE

Chạy:

```powershell
python train.py
```

Model [FraudDetectionGNN](../src/models/gnn_model.py) gồm hai lớp GraphSAGE
và classifier cho bài toán edge classification. Checkpoint được lưu tại:

```text
data/processed/graphsage_model.pt
```

Không thay đổi checkpoint sau khi bắt đầu sinh explanation nếu cần tái lập
kết quả. Explanation phải dùng đúng checkpoint đã tạo ra prediction.

## 6. Đánh giá model

Đánh giá trên tập test:

```powershell
python evaluate.py --threshold 0.5
```

Output:

```text
results/model_comparison.csv
results/model_comparison.json
results/graphsage_confusion_matrix.csv
results/xgboost_confusion_matrix.csv
```

`threshold` là ngưỡng phân loại fraud. Cần dùng cùng threshold khi lập Reason
Code để `prediction.label`, `fraud_probability` và kết quả đánh giá nhất quán.

## 7. Đóng băng model cho realtime (tùy chọn)

Nếu cần chạy realtime:

```powershell
python scripts\freeze_model.py `
  --checkpoint data\processed\graphsage_model.pt `
  --output data\processed\graphsage_frozen.pt `
  --benchmark-runs 1000
```

Output:

```text
data/processed/graphsage_frozen.pt
```

Bước này không bắt buộc cho batch XAI. Nó tạo checkpoint không còn gradient để
phục vụ suy luận nhẹ hơn.

## 8. Trích xuất subgraph cho XAI

Trích xuất tối đa 100 giao dịch fraud theo ground truth:

```powershell
python scripts\extract_fraud_subgraphs.py `
  --mode ground_truth `
  --limit 100 `
  --num-hops 2 `
  --device auto
```

Nếu muốn chọn các giao dịch model dự đoán fraud:

```powershell
python scripts\extract_fraud_subgraphs.py `
  --mode model_pred `
  --threshold 0.5 `
  --limit 100 `
  --num-hops 2 `
  --checkpoint data\processed\graphsage_model.pt
```

Module [subgraph_extractor.py](../src/subgraph_extractor.py) sẽ:

1. Chọn edge mục tiêu.
2. Lấy các node trong bán kính `num_hops`.
3. Giữ lại edge gốc giữa các node đó.
4. Relabel node cục bộ.
5. Lưu target edge.
6. Lưu metadata graph và account mapping cục bộ.

Output:

```text
data/processed/subgraphs/subgraph_edge_<id>.pt
data/processed/subgraphs_manifest.csv
data/processed/subgraphs_manifest.json
```

## 9. Chạy GNNExplainer và tạo Reason Code JSON

Chạy:

```powershell
python scripts\run_gnn_explainer.py `
  --checkpoint data\processed\graphsage_model.pt `
  --subgraphs-dir data\processed\subgraphs `
  --output-dir data\processed\xai_explanations `
  --reason-codes-dir data\processed\reason_codes `
  --epochs 200 `
  --lr 0.01 `
  --threshold 0.5 `
  --device auto
```

Script [run_gnn_explainer.py](../scripts/run_gnn_explainer.py) thực hiện lần
lượt cho từng subgraph:

1. Nạp checkpoint GraphSAGE.
2. Chạy [FraudGNNExplainer](../src/gnn_explainer.py).
3. Sinh `edge_mask` và `node_mask`.
4. Tính `node_importance`.
5. Tính lại `prediction_probability`.
6. Gọi [ReasonCodeGenerator](../src/reason_codes.py).
7. Lưu explanation artifact.
8. Lưu Reason Code JSON.
9. Ghi manifest tổng hợp.

Output XAI:

```text
data/processed/xai_explanations/explanation_subgraph_edge_<id>.pt
data/processed/xai_explanations/manifest.csv
```

Output Reason Code:

```text
data/processed/reason_codes/transaction_edge_<id>.json
```

Reason Code được tạo từ dữ liệu model, mask và metadata graph. Các mã hiện có
thể gồm:

```text
RC-STRUCT-CHAIN
RC-STRUCT-HUB
RC-STRUCT-CYCLE
RC-TEMPORAL-BURST
RC-ACCOUNT-NEW
RC-XAI-LOW-SIGNAL
```

Gemini chưa được gọi ở bước này.

## 10. Kiểm tra và trực quan hóa XAI

Xuất ảnh subgraph:

```powershell
python scripts\visualize_subgraphs.py `
  --subgraphs-dir data\processed\subgraphs `
  --xai-dir data\processed\xai_explanations `
  --output-dir data\processed\figures_xai `
  --threshold 0.5
```

Output:

```text
data/processed/figures_xai/*.png
```

Ảnh chỉ phục vụ kiểm tra/trình bày. Nguồn dữ liệu truy vết chính vẫn là
Reason Code JSON và explanation `.pt`.

## 11. Sinh báo cáo Gemini tùy chọn

### 11.1. Cấu hình biến môi trường

Không ghi API key vào source code hoặc file được commit:

```powershell
$env:GEMINI_API_KEY = "your-api-key" #(vào web google AI Studio lấy api key cá nhân thay vào)
$env:GEMINI_MODEL = "gemini-3.6-flash"
```

Kiểm tra biến môi trường mà không in API key:

```powershell
if ($env:GEMINI_API_KEY) { "GEMINI_API_KEY is configured" } else { "GEMINI_API_KEY is missing" }
```

### 11.2. Chạy reporter

Nếu muốn tự chỉnh yêu cầu của báo cáo:
```powershell
python scripts\generate_transaction_reports.py `
  --input-dir data\processed\reason_codes `
  --output-dir data\processed\transaction_reports `
  --top-k 5 `
  --max-prompt-chars 12000 `
  --max-string-length 500 `
  --timeout 30 `
  --retry-count 2
```
Tạo báo cáo mặc định:
```powershell
python scripts\generate_transaction_reports.py `
  --input-dir data\processed\reason_codes `
  --output-dir data\processed\transaction_reports `
  --limit 1 `
  --force
```


Các tham số:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--input-dir` | `data/processed/reason_codes` | Thư mục Reason Code JSON |
| `--output-dir` | `data/processed/transaction_reports` | Thư mục report |
| `--limit` | không giới hạn | Số file tối đa |
| `--top-k` | `5` | Số reason code/edge/node tối đa |
| `--max-prompt-chars` | `12000` | Kích thước payload tối đa |
| `--max-string-length` | `500` | Kích thước tối đa mỗi chuỗi |
| `--gemini-model` | từ `GEMINI_MODEL` | Model Gemini |
| `--timeout` | `30` giây | Timeout mỗi request |
| `--retry-count` | `2` | Số lần retry giới hạn |
| `--prompt-file` | prompt mặc định | File system instruction |
| `--force` | tắt | Bỏ qua cache và tạo lại |

Output:

```text
data/processed/transaction_reports/transaction_<id>.md
data/processed/transaction_reports/transaction_<id>.json
```

Reporter [transaction_reporter.py](../src/transaction_reporter.py) không gửi:

- toàn bộ tensor;
- toàn bộ graph;
- `edge_index` đầy đủ;
- account ID nguyên bản;
- API key.

Payload chỉ chứa các trường cần thiết, được cắt theo giới hạn và đặt trong
vùng `<transaction_data>`. Dữ liệu bên trong vùng này được coi là dữ liệu
không đáng tin cậy, không phải instruction.

Nếu thiếu API key, timeout hoặc Gemini trả về lỗi, reporter ghi trạng thái
`failed` cho file đó và tiếp tục file tiếp theo. Reason Code JSON gốc không bị
thay đổi.

## 12. Cache và chạy lại report

Reporter tạo cache key từ:

```text
hash(filtered_payload + prompt_version + model)
```

Nếu report JSON và Markdown hợp lệ đã tồn tại với cùng cache key, file sẽ được
bỏ qua:

```text
skipped: transaction_edge_123.json
```

Tạo lại bắt buộc:

```powershell
python scripts\generate_transaction_reports.py `
  --input-dir data\processed\reason_codes `
  --output-dir data\processed\transaction_reports `
  --force
```

## 13. Realtime inference (nhánh riêng)

Realtime không thay thế batch pipeline. Sau khi đã có graph train, mapping và
checkpoint, có thể chạy:

```powershell
python scripts\simulate_realtime.py `
  --limit 100 `
  --verbose
```

Bật XAI realtime:

```powershell
python scripts\simulate_realtime.py `
  --limit 10 `
  --verbose `
  --enable-xai `
  --xai-output-dir data\processed\xai_realtime
```

Realtime pipeline dùng:

1. [DataProcessor](../src/data_processor.py) để tạo feature giao dịch.
2. GraphSAGE checkpoint để tính fraud probability.
3. Subgraph cục bộ nếu giao dịch bị đánh dấu fraud.
4. GNNExplainer và ReasonCodeGenerator nếu XAI được bật.

Giao dịch cold-start có thể được dự đoán nhưng có thể không tạo được subgraph
XAI nếu account chưa tồn tại trong graph nền.

## 14. Kiểm thử

Chạy test toàn bộ:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Chạy nhóm test liên quan XAI/report:

```powershell
python -m unittest `
  tests.test_reason_codes `
  tests.test_gnn_explainer `
  tests.test_transaction_reporting
```

Kiểm tra CLI mà không gọi Gemini:

```powershell
python scripts\generate_transaction_reports.py --help
```

Import module mới không tạo request mạng. Unit test dùng fake opener, không gọi
Gemini thật.

## 15. Sơ đồ output cuối cùng

Sau khi chạy batch đầy đủ, cấu trúc chính có dạng:

```text
data/
├── raw/
│   └── SAML-D.csv
└── processed/
    ├── train.csv
    ├── test.csv
    ├── data_train.pt
    ├── data_test.pt
    ├── account_mapping.pkl
    ├── graphsage_model.pt
    ├── subgraphs/
    │   └── subgraph_edge_<id>.pt
    ├── xai_explanations/
    │   ├── explanation_subgraph_edge_<id>.pt
    │   └── manifest.csv
    ├── reason_codes/
    │   └── transaction_edge_<id>.json
    └── transaction_reports/
        ├── transaction_edge_<id>.md
        └── transaction_edge_<id>.json
```

## 16. Lưu ý vận hành và pháp lý

- Reason Code là bằng chứng giải thích tương đối của model, không phải quan
  hệ nhân quả.
- Báo cáo Gemini chỉ diễn giải dữ liệu đã có; không dùng để quyết định fraud.
- Không dùng report làm kết luận pháp lý hoặc bằng chứng độc lập.
- Kiểm tra quyền riêng tư và phê duyệt việc gửi dữ liệu tới Gemini trước khi
  chạy trên dữ liệu production.
- Theo dõi chi phí bằng `--limit`, `--top-k`, `--max-prompt-chars` và cache.
- Giữ lại checkpoint, manifest, prompt version và cấu hình threshold để tái
  lập kết quả.



