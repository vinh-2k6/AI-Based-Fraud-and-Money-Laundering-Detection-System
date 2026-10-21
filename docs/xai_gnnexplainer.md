# Cơ chế sinh Reason Codes từ XAI

## 1. Mục đích và phạm vi

Tài liệu này mô tả cách hệ thống tạo bằng chứng giải thích cho một giao dịch
bị GraphSAGE đánh dấu là gian lận, và cách chuyển bằng chứng đó thành
**Reason Codes** để hiển thị trong báo cáo điều tra.

Reason Code không phải là một nhãn được dự đoán trực tiếp bởi mô hình. Quy
trình gồm hai lớp:

1. **XAI/model layer**: `GNNExplainer` tìm các cạnh và node có ảnh hưởng lớn
   đến dự đoán của GraphSAGE.
2. **Reporting layer**: một bộ quy tắc hậu xử lý xếp hạng các cạnh/node đó,
   đối chiếu với dữ liệu giao dịch và gán mã lý do chuẩn hóa.

Trong phiên bản hiện tại, mã nguồn triển khai lớp XAI và lớp hậu xử lý
`ReasonCodeGenerator` tại `src/reason_codes.py`. XAI lưu `edge_mask`,
`node_mask`, `node_importance` cùng metadata; `ReasonCodeGenerator` chuẩn hóa
mask, đối chiếu cấu trúc subgraph/feature giao dịch và trả về các mã nghiệp vụ.

## 2. Luồng xử lý tổng thể

```text
CSV/API transaction
        |
        v
DataProcessor -> feature tensor và account index
        |
        v
GraphSAGE -> logit -> sigmoid -> fraud_probability
        |
        | probability >= threshold và enable_xai=True
        v
Trích xuất subgraph k-hop quanh sender/receiver
        |
        v
GNNExplainer -> edge_mask, node_mask
        |
        v
Chuẩn hóa + xếp hạng bằng chứng
        |
        v
Reason Code(s) + evidence + confidence
```

XAI chỉ được gọi khi giao dịch bị đánh dấu:

```text
is_fraud = fraud_probability >= threshold
```

Trong realtime pipeline, XAI còn yêu cầu sender và receiver tồn tại trong
đồ thị train. Giao dịch cold-start vẫn có thể được dự đoán, nhưng không thể
giải thích bằng subgraph của đồ thị train nếu node chưa tồn tại.

## 3. Dữ liệu đầu vào của GNNExplainer

### 3.1. Subgraph

`_build_transaction_subgraph()` tạo subgraph quanh sender và receiver với
bán kính mặc định `num_hops=2`. Việc mở rộng dùng cạnh vô hướng để không bỏ
qua các mẫu vòng tiền/chuỗi trung gian thường gặp trong AML, sau đó giữ lại
`edge_index` gốc để đưa vào model.

Subgraph cần có:

| Trường | Ý nghĩa |
|---|---|
| `x` | Ma trận đặc trưng node, shape `[num_nodes, num_features]` |
| `edge_index` | Các giao dịch trong subgraph |
| `target_edge_label_index` | Cặp node của giao dịch cần giải thích |
| `target_edge_idx_in_sub` | Vị trí cạnh mục tiêu nếu có |
| `metadata` | Nhãn thật, account ID và thông tin truy vết nếu extractor đã lưu |

### 3.2. Dự đoán mục tiêu

Model được cấu hình cho:

```python
model_config = {
    "mode": "binary_classification",
    "task_level": "edge",
    "return_type": "raw",
}
```

Model trả về `logit` cho cạnh mục tiêu. Xác suất được tính bằng:

```text
fraud_probability = sigmoid(logit)
```

Ngưỡng mặc định là `0.5`, nhưng Reason Code phải luôn lưu cả xác suất và
ngưỡng tại thời điểm chạy để kết quả có thể tái lập.

## 4. GNNExplainer sinh bằng chứng như thế nào

Với graph `G=(V,E)` và đặc trưng node `X`, GNNExplainer tối ưu hai mask:

- `M_E`: mask cho từng cạnh trong subgraph.
- `M_X`: mask cho từng đặc trưng của từng node.

Mục tiêu là giữ lại phần nhỏ của graph/features nhưng vẫn bảo toàn dự đoán
của model cho cạnh mục tiêu. Có thể mô tả khái quát:

```text
minimize  prediction_loss(model(X ⊙ M_X, E ⊙ M_E), target)
          + sparsity_regularization
          + entropy_regularization
```

Trong đó `⊙` là phép nhân theo phần tử. Đây là lời giải thích cục bộ cho
subgraph và lần chạy cụ thể, không phải quan hệ nhân quả.

### 4.1. `edge_mask`

`edge_mask[i]` là mức đóng góp tương đối của cạnh thứ `i` trong
`edge_index`. Cạnh có trị tuyệt đối lớn hơn được xem là bằng chứng cấu trúc
mạnh hơn cho dự đoán hiện tại. Vì mask có thể mang dấu, bộ hậu xử lý nên:

1. dùng `abs(edge_mask)` để xếp hạng độ mạnh;
2. giữ dấu trong evidence để biết hướng tác động nếu cần phân tích chuyên sâu;
3. không so sánh trực tiếp mask giữa các giao dịch khác nhau nếu chưa chuẩn
   hóa trong từng subgraph.

### 4.2. `node_mask` và `node_importance`

Với `node_mask` dạng `[num_nodes, num_features]`, hệ thống hiện tại tính:

```python
node_importance = node_mask.abs().mean(dim=-1)
```

Do đó `node_importance[j]` là độ quan trọng tổng hợp của node `j`, không phải
độ quan trọng của một feature cụ thể. Nếu cần mã lý do theo feature (ví dụ
`RC-AMOUNT-EXTREME`), reporter phải giữ lại `node_mask` gốc và biết mapping
`feature_index -> feature_name` được dùng khi tạo `x`.

## 5. Chuẩn hóa và xếp hạng bằng chứng

Do scale của mask phụ thuộc subgraph và lần tối ưu, reporter nên chuẩn hóa
trong phạm vi từng explanation:

```text
edge_score_i = abs(edge_mask_i) / max(abs(edge_mask))
node_score_j = node_importance_j / max(node_importance)
```

Nếu mẫu số bằng `0`, explanation không có bằng chứng phân biệt; không được
gán Reason Code mạnh, mà phải trả về trạng thái `XAI_LOW_SIGNAL`.

Quy tắc đề xuất:

1. Lấy tối đa 3 cạnh có `edge_score >= 0.5`, theo thứ tự giảm dần.
2. Lấy tối đa 3 node có `node_score >= 0.5`, không trùng các node đã chọn nếu
   báo cáo cần tránh lặp.
3. Đối chiếu các cạnh/node được chọn với dữ liệu gốc: amount, loại giao dịch,
   thời gian, account ID, số giao dịch lân cận và các feature đã chuẩn hóa.
4. Chỉ phát hành Reason Code khi vừa có tín hiệu XAI vừa có điều kiện dữ liệu
   tương ứng. Mask cao một mình không đủ để kết luận nghiệp vụ.

Ví dụ cấu trúc output nên có:

```json
{
  "reason_codes": [
    {
      "code": "RC-STRUCT-CHAIN",
      "label": "Chuỗi giao dịch liên kết bất thường",
      "score": 0.91,
      "evidence_edges": [3, 7],
      "evidence_nodes": [0, 4],
      "basis": "edge_mask và node_importance sau chuẩn hóa"
    }
  ],
  "fraud_probability": 0.87,
  "threshold": 0.5,
  "explanation_status": "OK"
}
```

`score` trong Reason Code là điểm xếp hạng bằng chứng, không phải xác suất
gian lận và không được trình bày như xác suất.

## 5.1. Lớp triển khai và tích hợp

API chính:

```python
from src.reason_codes import ReasonCodeGenerator

generator = ReasonCodeGenerator(min_score=0.5, max_codes=3)
reason_result = generator.generate(
    explanation,
    subgraph=sub_data,
    transaction_features=feature_dict,
)
```

Kết quả được gắn vào:

- artifact `explanation_*.pt` và `manifest.csv` khi chạy
  `scripts\run_gnn_explainer.py`;
- kết quả `infer_processed_transaction()` trong realtime khi bật XAI.

Các mã đang được sinh tự động là `RC-STRUCT-CHAIN`,
`RC-STRUCT-HUB`, `RC-STRUCT-CYCLE`, `RC-TEMPORAL-BURST`,
`RC-ACCOUNT-NEW` và fallback `RC-XAI-LOW-SIGNAL`. Các mã dựa trên amount,
loại giao dịch hoặc feature chi tiết chỉ nên bật sau khi feature-to-name
mapping và ngưỡng nghiệp vụ đã được xác nhận.

## 6. Danh mục Reason Code đề xuất

Các mã dưới đây là lớp nghiệp vụ tham chiếu. Điều kiện dữ liệu cụ thể phải
được hiệu chỉnh theo phân phối train và được kiểm thử trước khi dùng trong
quy trình tuân thủ.

| Mã | Ý nghĩa | Bằng chứng XAI tối thiểu | Đối chiếu dữ liệu |
|---|---|---|---|
| `RC-STRUCT-CHAIN` | Chuỗi/chuyển tiếp qua nhiều tài khoản | Nhiều cạnh lân cận có edge score cao | Có đường sender -> trung gian -> receiver trong k-hop |
| `RC-STRUCT-CYCLE` | Vòng giao dịch khép kín | Các cạnh tạo chu kỳ có score cao | Phát hiện cycle hoặc dòng tiền quay lại account |
| `RC-STRUCT-HUB` | Tài khoản trung tâm bất thường | Một node trung gian có node score cao, nhiều cạnh liên quan | Bậc vào/ra hoặc số counterparties cao |
| `RC-AMOUNT-ANOMALY` | Giá trị giao dịch bất thường | Node/feature amount nằm trong nhóm importance cao | Amount lệch đáng kể so với baseline phù hợp |
| `RC-TEMPORAL-BURST` | Cụm giao dịch dồn dập | Các cạnh trong cửa sổ thời gian có edge score cao | Tần suất/total amount trong cửa sổ 24h tăng bất thường |
| `RC-ACCOUNT-NEW` | Account mới hoặc cold-start | Node liên quan có importance cao | Account chưa có lịch sử hoặc mới xuất hiện |
| `RC-TYPE-RISK` | Loại giao dịch có rủi ro cao | Feature loại giao dịch có node feature mask cao | Loại giao dịch khớp chính sách rủi ro |
| `RC-XAI-LOW-SIGNAL` | Không đủ bằng chứng giải thích | Mask gần như đồng đều hoặc bằng 0 | Không phát hành kết luận chi tiết |

Không được suy ra các mã như `RC-AMOUNT-ANOMALY` chỉ từ `node_importance` tổng
hợp, vì tổng hợp này không cho biết feature amount có thực sự đứng đầu hay
không. Cần dùng `node_mask` và feature schema.

## 7. Truy vết artifact và tái lập

Lệnh batch hiện tại:

```powershell
python scripts\run_gnn_explainer.py `
  --checkpoint data\processed\graphsage_model.pt `
  --subgraphs-dir data\processed\subgraphs `
  --output-dir data\processed\xai_explanations `
  --epochs 200 `
  --device auto
```

Mỗi `explanation_*.pt` chứa:

- `edge_mask`
- `node_mask`
- `node_importance`
- `prediction_logit`
- `prediction_probability`
- `target_edge_label_index`
- `target_edge_idx_in_sub`
- `metadata`
- `source_subgraph` khi được lưu qua `save_explanation()`

`manifest.csv` bổ sung số node/cạnh, xác suất dự đoán, nhãn thật, max edge
importance, max node importance và trạng thái xử lý. Reporter phải lưu liên
kết giữa Reason Code và các artifact này để điều tra viên có thể dựng lại
bằng chứng gốc.

Để tái lập, cần cố định tối thiểu:

- checkpoint và phiên bản model;
- file subgraph và `num_hops`;
- `epochs`, `lr`, device;
- threshold;
- phiên bản feature schema và bộ quy tắc Reason Code.

## 8. Realtime và xử lý lỗi

Realtime pipeline gọi XAI sau inference khi `is_fraud=True` và
`enable_xai=True`. Nếu XAI thất bại, kết quả inference vẫn giữ lại nhưng phải
ghi rõ `xai_error`; không được tạo Reason Code dạng thành công từ dữ liệu
thiếu. Các trạng thái nên phân biệt:

- `OK`: có mask hợp lệ và đã tạo được bằng chứng;
- `XAI_LOW_SIGNAL`: mask hợp lệ nhưng không đủ nổi bật để gán mã;
- `XAI_UNAVAILABLE`: cold-start, thiếu subgraph hoặc lỗi runtime;
- `XAI_ERROR`: lỗi cần điều tra trong quá trình sinh explanation.

## 9. Giới hạn và kiểm soát

- Mask là giải thích tương đối của model, không phải quan hệ nhân quả.
- GNNExplainer có tính cục bộ và có thể thay đổi theo subgraph, epoch và
  learning rate.
- `fraud_probability` dùng để ra quyết định ngưỡng; Reason Code chỉ giải
  thích quyết định đó, không thay thế threshold.
- Không dùng Reason Code như bằng chứng pháp lý duy nhất; cần đối chiếu log
  giao dịch và quy tắc AML độc lập.
- Cần đánh giá độ ổn định bằng cách chạy lại explanation với seed/cấu hình
  phù hợp và đo tỷ lệ cạnh/node được chọn lặp lại.
- Mọi ngưỡng Reason Code phải được version hóa, kiểm thử trên tập validation
  và được phê duyệt trước khi đưa vào production.
