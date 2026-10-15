# GNNExplainer cho FraudDetectionGNN

Sinh explanation cho các subgraph đã trích xuất:

```powershell
python scripts\run_gnn_explainer.py `
  --checkpoint data\processed\graphsage_model.pt `
  --subgraphs-dir data\processed\subgraphs `
  --output-dir data\processed\xai_explanations `
  --epochs 200 `
  --device auto
```

Mỗi file `explanation_*.pt` chứa `edge_mask`, `node_mask`,
`node_importance`, logit/xác suất dự đoán, target edge và metadata gốc.
`edge_mask` thể hiện mức đóng góp tương đối của từng giao dịch vào dự đoán.
`node_mask` thể hiện đóng góp theo từng feature của node; `node_importance`
là trị tuyệt đối trung bình theo feature.

Visualizer có thể dùng mask thật:

```python
result = torch.load("data/processed/xai_explanations/explanation_subgraph_edge_1035.pt")
visualizer.plot_subgraph_explanation(
    sub_data,
    edge_mask=result["edge_mask"],
    node_importance=result["node_importance"],
    output_path="data/processed/figures_xai/edge_1035.png",
)
```

GNNExplainer tối ưu mask cho từng graph và có thể tốn nhiều thời gian. Mask
giải thích là bằng chứng tương đối của model, không phải quan hệ nhân quả hay
quyết định nghiệp vụ độc lập. Kết quả cũng phụ thuộc vào số epoch, learning
rate, subgraph và target edge.
