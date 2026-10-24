"""Đánh giá các bộ phân loại gian lận GraphSAGE và XGBoost đã được huấn luyện.

Dự án thực hiện phân loại cạnh nhị phân: mỗi cạnh trong đồ thị là một
giao dịch và ``data.y`` là nhãn rửa tiền của giao dịch đó. Module này không
huấn luyện mô hình; chỉ nạp các artifact đã lưu và đánh giá trên cùng một
tập kiểm tra.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.loader import LinkNeighborLoader

from src.models.gnn_model import FraudDetectionGNN


BASE_DIR = Path(__file__).resolve().parent
PROC_DIR = BASE_DIR / "data" / "processed"
DEFAULT_GRAPH_DATA = PROC_DIR / "data_test.pt"
DEFAULT_TEST_CSV = PROC_DIR / "test_cleaned.csv"
DEFAULT_GRAPH_CHECKPOINT = PROC_DIR / "graphsage_model.pt"
DEFAULT_XGBOOST_MODEL = PROC_DIR / "xgboost_baseline.json"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Trả về metric nhị phân, metric của lớp gian lận và ma trận nhầm lẫn."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud/AML"],
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fraud_precision": float(report["Fraud/AML"]["precision"]),
        "fraud_recall": float(report["Fraud/AML"]["recall"]),
        "fraud_f1": float(report["Fraud/AML"]["f1-score"]),
        "pr_auc": (
            float(average_precision_score(y_true, y_prob))
            if y_prob is not None
            else None
        ),
        "test_samples": int(len(y_true)),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    return result


def _write_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    pd.DataFrame(
        matrix,
        index=["actual_0_legitimate", "actual_1_fraud_aml"],
        columns=["predicted_0_legitimate", "predicted_1_fraud_aml"],
    ).to_csv(path, index=True)


def _load_graph_checkpoint(model: FraudDetectionGNN, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported GraphSAGE checkpoint format: {path}")
    model.load_state_dict(state_dict)


def evaluate_graphsage(
    checkpoint_path: Path,
    data_path: Path,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đánh giá logit cạnh nhị phân ở chế độ eval và không theo dõi gradient."""
    data = torch.load(data_path, map_location=device, weights_only=False)
    model = FraudDetectionGNN(in_channels=data.x.shape[1], hidden_channels=64).to(device)
    _load_graph_checkpoint(model, checkpoint_path)
    model.eval()

    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[10, 10],
        edge_label_index=data.edge_index,
        edge_label=data.y,
        batch_size=4096,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_label_index)
            probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            labels.append(batch.edge_label.cpu().numpy().reshape(-1))

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(labels).astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    return y_true, y_pred, y_prob


def _prepare_xgboost_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    categorical_cols = [
        "Payment_currency",
        "Received_currency",
        "Sender_bank_location",
        "Receiver_bank_location",
        "Payment_type",
    ]
    df = df.copy()
    for column in categorical_cols:
        df[column] = df[column].astype("category")
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    return df.drop(columns=["Is_laundering"]), df["Is_laundering"]


def evaluate_xgboost(model_path: Path, test_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    test_df = pd.read_csv(test_csv)
    features, labels = _prepare_xgboost_data(test_df)
    y_prob = model.predict_proba(features)[:, 1]
    return labels.to_numpy(dtype=int), (y_prob >= 0.5).astype(int), y_prob


def _ensure_same_test_size(graph_labels: np.ndarray, xgb_labels: np.ndarray) -> None:
    if len(graph_labels) != len(xgb_labels):
        raise ValueError(
            "GraphSAGE and XGBoost are not using the same test split: "
            f"{len(graph_labels)} vs {len(xgb_labels)} samples."
        )
    if not np.array_equal(graph_labels, xgb_labels):
        raise ValueError(
            "GraphSAGE and XGBoost test labels differ. Refusing to compare "
            "metrics without an explicit transaction/edge ID mapping."
        )


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_checkpoint = Path(args.graphsage_checkpoint)
    xgb_model = Path(args.xgboost_model)
    for path, description in (
        (graph_checkpoint, "GraphSAGE checkpoint"),
        (xgb_model, "XGBoost model"),
        (Path(args.graph_data), "Graph test data"),
        (Path(args.test_csv), "CSV test data"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{description} not found: {path}. "
                "Train the corresponding model first or pass its path explicitly."
            )

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    graph_true, graph_pred, graph_prob = evaluate_graphsage(
        graph_checkpoint, Path(args.graph_data), device, args.threshold
    )
    xgb_true, xgb_pred, xgb_prob = evaluate_xgboost(xgb_model, Path(args.test_csv))
    _ensure_same_test_size(graph_true, xgb_true)

    results = {
        "GraphSAGE": compute_metrics(graph_true, graph_pred, graph_prob),
        "XGBoost": compute_metrics(xgb_true, xgb_pred, xgb_prob),
    }
    for name, metrics in results.items():
        _write_confusion_matrix(
            metrics["confusion_matrix"],
            output_dir / f"{name.lower()}_confusion_matrix.csv",
        )

    comparison = pd.DataFrame(
        [
            {
                "Model": name,
                **{key: value for key, value in metrics.items()
                   if key in {"fraud_precision", "fraud_recall", "fraud_f1",
                              "pr_auc", "accuracy", "test_samples", "tn", "fp", "fn", "tp"}},
            }
            for name, metrics in results.items()
        ]
    )
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "model_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\n=== MODEL COMPARISON (class 1 = Fraud/AML) ===")
    print(comparison.to_string(index=False))
    for metric, label in (("fraud_recall", "Fraud Recall"), ("fraud_f1", "Fraud F1"), ("pr_auc", "PR-AUC")):
        winner = comparison.loc[comparison[metric].fillna(-1).idxmax(), "Model"]
        difference = results["GraphSAGE"][metric] - results["XGBoost"][metric]
        print(f"{label}: {winner} cao hơn; GraphSAGE - XGBoost = {difference:.6f}")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphsage-checkpoint", type=Path, default=DEFAULT_GRAPH_CHECKPOINT)
    parser.add_argument("--xgboost-model", type=Path, default=DEFAULT_XGBOOST_MODEL)
    parser.add_argument("--graph-data", type=Path, default=DEFAULT_GRAPH_DATA)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.66)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
