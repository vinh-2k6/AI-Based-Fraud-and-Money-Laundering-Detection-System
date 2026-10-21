"""
scripts/simulate_realtime.py
============================
Mô phỏng luồng giao dịch thời gian thực (Streaming Pipeline Simulation) phục vụ
kiểm thử End-to-End, đo độ trễ toàn trình và tối ưu hóa hệ thống (Tuần 5).

Kiến trúc tích hợp:
  1. DataProcessor (Thành viên 1 - Data Engineer):
     - Tiếp nhận bản ghi giao dịch thô hoặc pre-cleaned từ luồng streaming.
     - Trích xuất đặc trưng Causal Sliding Window 24h, mã hóa chu kỳ (cyclical) và One-Hot.
     - Chuẩn hóa MinMaxScaler và chuyển đổi trực tiếp sang tensor PyTorch Geometric.
  2. GraphSAGE Frozen Inference (Thành viên 2 - Graph Engineer & Thành viên 3 - AI Architect):
     - Nạp mô hình đóng băng (graphsage_frozen.pt) với requires_grad=False nhằm tối ưu bộ nhớ.
     - Truy xuất embedding của nút gửi (sender) và nút nhận (receiver).
     - Đưa qua lớp Classifier để tính xác suất gian lận / rửa tiền (Fraud Probability).
  3. Explainable AI - GNNExplainer (Thành viên 3 - AI Architect & Thành viên 2):
     - Khi phát hiện giao dịch gian lận (is_fraud=True) và có yêu cầu giải thích,
       trích xuất đồ thị con cục bộ k-hop và sinh lời giải thích (node & edge importance).
  4. Đo kiểm và Phân rã độ trễ (Thành viên 4 - MLOps & Reporter):
     - Data Processing Latency (Thành viên 1 - DataProcessor)
     - GraphSAGE Inference Latency (Thành viên 2/3 - GraphSAGE Frozen)
     - Total End-to-End Latency
     - Xác nhận tiêu chuẩn kỹ thuật SLA: Tổng độ trễ toàn trình P99 < 100ms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import pickle
import statistics
import sys
import time
from typing import Any, cast

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph, to_undirected

from src.data_processor import DataProcessor, ProcessedTransaction
from src.gnn_explainer import FraudGNNExplainer
from src.reason_codes import ReasonCodeGenerator
from src.models.gnn_model import FraudDetectionGNN

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_FILE_PATH = Path("data") / "processed" / "test_cleaned.csv"
DEFAULT_GRAPH_PATH = BASE_DIR / "data" / "processed" / "data_train.pt"
DEFAULT_MAPPING_PATH = BASE_DIR / "data" / "processed" / "account_mapping.pkl"
DEFAULT_FROZEN_CHECKPOINT = BASE_DIR / "data" / "processed" / "graphsage_frozen.pt"
DEFAULT_CHECKPOINT_PATH = (
    DEFAULT_FROZEN_CHECKPOINT
    if DEFAULT_FROZEN_CHECKPOINT.exists()
    else BASE_DIR / "data" / "processed" / "graphsage_model.pt"
)
DEFAULT_XAI_OUTPUT_DIR = BASE_DIR / "data" / "processed" / "xai_explanations"
DEFAULT_THRESHOLD = 0.5
DEFAULT_XAI_EPOCHS = 200
MAX_VERBOSE_ROWS = 10

_INFERENCE_RUNTIME: dict[str, Any] | None = None
_DATA_PROCESSOR: DataProcessor | None = None
_XAI_EXPLAINER: FraudGNNExplainer | None = None
_REASON_CODE_GENERATOR = ReasonCodeGenerator()


def get_data_processor() -> DataProcessor:
    """Khởi tạo hoặc lấy singleton DataProcessor."""
    global _DATA_PROCESSOR
    if _DATA_PROCESSOR is None:
        _DATA_PROCESSOR = DataProcessor(base_dir=BASE_DIR, device="cpu")
    return _DATA_PROCESSOR


def parse_row(row: dict[str, str | None]) -> dict[str, object]:
    """Phân tích và ép kiểu các giá trị từ một dòng CSV."""
    parsed_row: dict[str, object] = {}

    for column, value in row.items():
        if value is None or value == "":
            parsed_row[column] = None
            continue

        try:
            parsed_row[column] = int(value)
            continue
        except ValueError:
            pass

        try:
            parsed_row[column] = float(value)
            continue
        except ValueError:
            parsed_row[column] = value

    return parsed_row


def _load_inference_runtime() -> dict[str, Any]:
    """Nạp các artifact suy luận một lần và giữ chúng trong bộ nhớ."""
    global _INFERENCE_RUNTIME
    if _INFERENCE_RUNTIME is not None:
        return _INFERENCE_RUNTIME

    active_checkpoint_path = (
        DEFAULT_FROZEN_CHECKPOINT
        if DEFAULT_FROZEN_CHECKPOINT.exists()
        else DEFAULT_CHECKPOINT_PATH
    )

    for path, description in (
        (DEFAULT_GRAPH_PATH, "đồ thị train"),
        (DEFAULT_MAPPING_PATH, "account mapping"),
        (active_checkpoint_path, "GraphSAGE checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {description}: {path}. "
                "Hãy chạy pipeline xây dựng graph và huấn luyện model trước."
            )

    graph = torch.load(DEFAULT_GRAPH_PATH, map_location="cpu", weights_only=False)
    with DEFAULT_MAPPING_PATH.open("rb") as mapping_file:
        mapping = pickle.load(mapping_file)

    if not isinstance(mapping, dict):
        raise ValueError(
            f"Định dạng account mapping không hợp lệ: {DEFAULT_MAPPING_PATH}"
        )
    account_to_idx = mapping.get("account_to_idx", mapping)

    checkpoint = torch.load(
        active_checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    in_channels = (
        int(checkpoint.get("in_channels", graph.x.shape[1]))
        if isinstance(checkpoint, dict)
        else graph.x.shape[1]
    )
    hidden_channels = (
        int(checkpoint.get("hidden_channels", 64))
        if isinstance(checkpoint, dict)
        else 64
    )

    model = FraudDetectionGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Tắt gradient toàn bộ tham số nếu nạp mô hình đóng băng
    for param in model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        node_embeddings = model.conv1(graph.x, graph.edge_index).relu()
        node_embeddings = model.conv2(node_embeddings, graph.edge_index).relu()

    _INFERENCE_RUNTIME = {
        "graph": graph,
        "account_to_idx": account_to_idx,
        "node_embeddings": node_embeddings,
        "model": model,
        "classifier": model.classifier,
        "is_frozen": active_checkpoint_path == DEFAULT_FROZEN_CHECKPOINT,
    }
    return _INFERENCE_RUNTIME


def _build_transaction_subgraph(
    graph: Data,
    sender_idx: int,
    receiver_idx: int,
    account_to_idx: dict[int, int],
    num_hops: int = 2,
) -> Data:
    """Xây dựng đồ thị con cục bộ xung quanh giao dịch để phục vụ giải thích XAI."""
    if num_hops <= 0:
        raise ValueError("num_hops phải lớn hơn 0")
    graph_node_count = int(graph.x.size(0))
    for node_idx in (sender_idx, receiver_idx):
        if not 0 <= node_idx < graph_node_count:
            raise ValueError(
                f"Node index {node_idx} nằm ngoài phạm vi graph "
                f"[0, {graph_node_count - 1}]"
            )

    undirected_edges = to_undirected(graph.edge_index)
    subset_nodes, _, _, _ = k_hop_subgraph(
        node_idx=torch.tensor([sender_idx, receiver_idx], dtype=torch.long),
        num_hops=num_hops,
        edge_index=undirected_edges,
        relabel_nodes=False,
        num_nodes=graph_node_count,
    )
    local_edge_index, local_edge_attr = subgraph(
        subset=subset_nodes,
        edge_index=graph.edge_index,
        edge_attr=getattr(graph, "edge_attr", None),
        relabel_nodes=True,
        num_nodes=graph_node_count,
    )
    global_to_local = {
        int(global_idx): local_idx
        for local_idx, global_idx in enumerate(subset_nodes.tolist())
    }
    target_edge_label_index = torch.tensor(
        [
            [global_to_local[sender_idx]],
            [global_to_local[receiver_idx]],
        ],
        dtype=torch.long,
    )
    idx_to_account = {idx: account for account, idx in account_to_idx.items()}
    sub_data = Data(
        x=graph.x[subset_nodes],
        edge_index=local_edge_index,
        edge_attr=local_edge_attr,
        num_nodes=subset_nodes.numel(),
    )
    sub_data.target_edge_label_index = target_edge_label_index
    sub_data.orig_node_indices = subset_nodes
    sub_data.orig_account_ids = [
        idx_to_account.get(int(node_idx), int(node_idx))
        for node_idx in subset_nodes
    ]
    sub_data.metadata = {
        "src_account": idx_to_account.get(sender_idx, sender_idx),
        "dst_account": idx_to_account.get(receiver_idx, receiver_idx),
        "num_hops": num_hops,
        "source": "realtime_inference",
    }
    return sub_data


def _explain_transaction(
    sub_data: Data,
    epochs: int = DEFAULT_XAI_EPOCHS,
) -> dict[str, object]:
    """Sinh lời giải thích GNNExplainer cho giao dịch bị đánh dấu gian lận."""
    global _XAI_EXPLAINER
    runtime = _load_inference_runtime()
    if _XAI_EXPLAINER is None:
        _XAI_EXPLAINER = FraudGNNExplainer(
            model=runtime["model"],
            device="cpu",
            epochs=epochs,
        )
    explanation = _XAI_EXPLAINER.explain_subgraph(sub_data)
    return explanation


def infer_processed_transaction(
    processed_tx: ProcessedTransaction,
    threshold: float = DEFAULT_THRESHOLD,
    enable_xai: bool = False,
    xai_epochs: int = DEFAULT_XAI_EPOCHS,
    xai_num_hops: int = 2,
    xai_output_path: str | Path | None = None,
) -> dict[str, object]:
    """Dự đoán một giao dịch đã được DataProcessor xử lý thành tensor."""
    runtime = _load_inference_runtime()
    node_embeddings = runtime["node_embeddings"]
    node_count = int(node_embeddings.size(0))

    sender_idx = processed_tx.sender_idx
    receiver_idx = processed_tx.receiver_idx

    # Xử lý Cold-Start an toàn nếu node_idx ngoài phạm vi node_embeddings của base train graph
    if sender_idx >= node_count:
        sender_emb = torch.zeros(node_embeddings.shape[1], dtype=torch.float32)
    else:
        sender_emb = node_embeddings[sender_idx]

    if receiver_idx >= node_count:
        receiver_emb = torch.zeros(node_embeddings.shape[1], dtype=torch.float32)
    else:
        receiver_emb = node_embeddings[receiver_idx]

    edge_embedding = torch.cat([sender_emb, receiver_emb], dim=-1)

    with torch.no_grad():
        logit = runtime["classifier"](edge_embedding)
        probability = float(torch.sigmoid(logit).reshape(-1)[0].item())

    is_fraud = probability >= threshold
    result: dict[str, object] = {
        "sender_account": processed_tx.sender_account,
        "receiver_account": processed_tx.receiver_account,
        "fraud_probability": probability,
        "is_fraud": is_fraud,
        "threshold": threshold,
        "is_cold_start": processed_tx.is_new_sender or processed_tx.is_new_receiver,
    }

    # Kích hoạt GNNExplainer khi phát hiện gian lận và được yêu cầu
    if is_fraud and enable_xai:
        try:
            if sender_idx < node_count and receiver_idx < node_count:
                sub_data = _build_transaction_subgraph(
                    graph=runtime["graph"],
                    sender_idx=sender_idx,
                    receiver_idx=receiver_idx,
                    account_to_idx=runtime["account_to_idx"],
                    num_hops=xai_num_hops,
                )
                explanation = _explain_transaction(sub_data, epochs=xai_epochs)
                reason_codes = _REASON_CODE_GENERATOR.generate(
                    explanation,
                    subgraph=sub_data,
                    transaction_features={
                        **processed_tx.feature_dict,
                        "is_new_sender": processed_tx.is_new_sender,
                        "is_new_receiver": processed_tx.is_new_receiver,
                    },
                )
                explanation["reason_codes"] = reason_codes
                if xai_output_path is not None:
                    FraudGNNExplainer.save_explanation(
                        explanation,
                        xai_output_path,
                        source_subgraph="realtime_transaction",
                    )
                    result["xai_output_path"] = str(xai_output_path)
                result["explanation"] = explanation
                result["reason_codes"] = reason_codes
        except Exception as exc:
            result["xai_error"] = str(exc)

    return result


def infer_transaction(
    tx_json: str,
    threshold: float = DEFAULT_THRESHOLD,
    enable_xai: bool = False,
    xai_epochs: int = DEFAULT_XAI_EPOCHS,
    xai_num_hops: int = 2,
    xai_output_path: str | Path | None = None,
) -> dict[str, object]:
    """Hàm tương thích ngược với API cũ nhận vào chuỗi JSON."""
    processor = get_data_processor()
    processed_tx = processor.process_json(tx_json)
    return infer_processed_transaction(
        processed_tx,
        threshold=threshold,
        enable_xai=enable_xai,
        xai_epochs=xai_epochs,
        xai_num_hops=xai_num_hops,
        xai_output_path=xai_output_path,
    )


def process_transaction(
    row: dict[str, str | None],
    verbose: bool = False,
    enable_xai: bool = False,
    xai_epochs: int = DEFAULT_XAI_EPOCHS,
    xai_num_hops: int = 2,
    xai_output_dir: Path | None = None,
) -> tuple[str, float, float, float, dict[str, Any]]:
    """
    Xử lý giao dịch qua pipeline 2 bước:
      Bước 1: Data Processing (DataProcessor - TV1)
      Bước 2: GraphSAGE Inference (GraphSAGE Frozen - TV2)

    Returns:
      (tx_json, data_proc_latency, inference_latency, total_latency, inference_result)
    """
    processor = get_data_processor()

    # Bước 1: Data Processing & Feature Extraction (TV1)
    t0 = time.perf_counter()
    parsed_row = parse_row(row)
    processed_tx = processor.process_transaction(parsed_row)
    t1 = time.perf_counter()
    data_proc_latency = t1 - t0

    # Bước 2: Model Inference & Optional XAI (TV2 / TV3)
    xai_output_path = None
    if enable_xai and xai_output_dir is not None:
        xai_output_dir.mkdir(parents=True, exist_ok=True)
        xai_output_path = xai_output_dir / (
            f"realtime_{parsed_row.get('Sender_account')}_"
            f"{parsed_row.get('Receiver_account')}.pt"
        )

    inference = infer_processed_transaction(
        processed_tx,
        threshold=DEFAULT_THRESHOLD,
        enable_xai=enable_xai,
        xai_epochs=xai_epochs,
        xai_num_hops=xai_num_hops,
        xai_output_path=xai_output_path,
    )
    t2 = time.perf_counter()
    inference_latency = t2 - t1

    total_latency = data_proc_latency + inference_latency
    tx_json = json.dumps(parsed_row, ensure_ascii=False)

    if verbose:
        cold_tag = " [COLD-START]" if inference.get("is_cold_start") else ""
        print(
            f"Kết quả GraphSAGE: fraud_prob={inference['fraud_probability']:.4f}, "
            f"is_fraud={inference['is_fraud']}{cold_tag} | "
            f"DataProc={data_proc_latency*1000:.3f}ms, Infer={inference_latency*1000:.3f}ms, "
            f"Total={total_latency*1000:.3f}ms"
        )
        if inference.get("explanation") is not None:
            print("   -> Đã trích xuất đồ thị con và sinh giải thích GNNExplainer.")

    return tx_json, data_proc_latency, inference_latency, total_latency, inference


def _repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_file_path(file_path: str | Path) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else _repository_root() / path


def _validate_row(
    row: dict[str | None, str | None],
    fieldnames: list[str],
) -> dict[str, str | None]:
    if None in row:
        raise ValueError("Dòng chứa nhiều trường hơn số cột trong header")

    missing_columns = [
        column for column in fieldnames if row.get(column) is None
    ]
    if missing_columns:
        raise ValueError(
            "Dòng bị thiếu các trường: " + ", ".join(missing_columns)
        )

    return cast(dict[str, str | None], row)


def load_and_simulate_transactions(
    file_path: str | Path,
    limit: int | None = None,
    verbose: bool = False,
    enable_xai: bool = False,
    xai_epochs: int = DEFAULT_XAI_EPOCHS,
    xai_num_hops: int = 2,
    xai_output_dir: str | Path | None = None,
) -> tuple[list[float], list[float], list[float], int, int, float]:
    """
    Đọc tuần tự giao dịch từ CSV và thu thập số liệu chi tiết về độ trễ.

    Returns:
        (total_latencies, data_proc_latencies, inference_latencies, total_rows, error_rows, total_time)
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit phải lớn hơn 0")

    print("Khởi động môi trường suy luận (Warming up inference runtime & DataProcessor)...")
    _load_inference_runtime()
    get_data_processor()

    total_latencies: list[float] = []
    data_proc_latencies: list[float] = []
    inference_latencies: list[float] = []

    resolved_xai_dir: Path | None = None
    if enable_xai and xai_output_dir is not None:
        resolved_xai_dir = (
            Path(xai_output_dir)
            if Path(xai_output_dir).is_absolute()
            else _repository_root() / xai_output_dir
        )

    total_rows = 0
    error_rows = 0
    start_time = time.perf_counter()
    resolved_path = _resolve_file_path(file_path)

    try:
        with resolved_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames

            if not fieldnames:
                raise ValueError("tệp CSV rỗng hoặc không có header")
            if any(not field.strip() for field in fieldnames):
                raise ValueError("header CSV chứa tên cột rỗng")
            if len(set(fieldnames)) != len(fieldnames):
                raise ValueError("header CSV chứa tên cột trùng lặp")

            print(f"Bắt đầu luồng kiểm thử Streaming từ {resolved_path.name}...")
            for raw_row in reader:
                total_rows += 1
                try:
                    row = _validate_row(raw_row, fieldnames)
                    tx_json, d_lat, i_lat, t_lat, _ = process_transaction(
                        row,
                        verbose=verbose,
                        enable_xai=enable_xai,
                        xai_epochs=xai_epochs,
                        xai_num_hops=xai_num_hops,
                        xai_output_dir=resolved_xai_dir,
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    error_rows += 1
                    print(
                        f"Bỏ qua dòng {total_rows}: {exc}",
                        file=sys.stderr,
                    )
                    if limit is not None and total_rows >= limit:
                        break
                    continue

                data_proc_latencies.append(d_lat)
                inference_latencies.append(i_lat)
                total_latencies.append(t_lat)

                if verbose and len(total_latencies) <= MAX_VERBOSE_ROWS:
                    print(
                        f"Giao dịch {len(total_latencies)}: "
                        f"(DataProc={d_lat*1000:.3f}ms, Infer={i_lat*1000:.3f}ms -> Total={t_lat*1000:.3f}ms)"
                    )
                if limit is not None and total_rows >= limit:
                    break

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Tệp CSV không tồn tại: {resolved_path}"
        ) from None
    except PermissionError:
        raise PermissionError(
            f"Không có quyền đọc tệp CSV: {resolved_path}"
        ) from None
    except csv.Error as exc:
        error_rows += 1
        print(f"Lỗi phân tích CSV: {exc}", file=sys.stderr)
    except (UnicodeDecodeError, OSError, ValueError):
        raise
    except KeyboardInterrupt:
        print(
            "\nĐã dừng mô phỏng; đang báo cáo các giao dịch đã xử lý.",
            file=sys.stderr,
        )

    total_time = time.perf_counter() - start_time
    return (
        total_latencies,
        data_proc_latencies,
        inference_latencies,
        total_rows,
        error_rows,
        total_time,
    )


def print_summary(
    total_latencies: list[float],
    data_proc_latencies: list[float],
    inference_latencies: list[float],
    total_rows: int,
    error_rows: int,
    total_time: float,
) -> None:
    """In thống kê phân rã độ trễ và đánh giá tiêu chuẩn SLA (< 100ms)."""
    successful_rows = len(total_latencies)
    print("\n" + "=" * 65)
    print("BÁO CÁO KẾT QUẢ KIỂM THỬ ĐỘ TRỄ TOÀN TRÌNH (STREAMING E2E)")
    print("=" * 65)
    print(f"Tổng số dòng đã nạp     : {total_rows:,}")
    print(f"Số giao dịch thành công : {successful_rows:,}")
    print(f"Số dòng lỗi/bỏ qua      : {error_rows:,}")
    print(f"Tổng thời gian mô phỏng : {total_time * 1000:.2f} ms")

    if not total_latencies:
        print("Không có dữ liệu đo đạc hợp lệ.")
        return

    def _stats(lats: list[float]) -> dict[str, float]:
        ms = np.array(lats) * 1000.0
        return {
            "mean": float(np.mean(ms)),
            "std": float(np.std(ms)) if len(ms) > 1 else 0.0,
            "min": float(np.min(ms)),
            "p50": float(np.percentile(ms, 50)),
            "p95": float(np.percentile(ms, 95)),
            "p99": float(np.percentile(ms, 99)),
            "max": float(np.max(ms)),
        }

    d_stats = _stats(data_proc_latencies)
    i_stats = _stats(inference_latencies)
    t_stats = _stats(total_latencies)

    print("\n1. ĐỘ TRỄ DATA PROCESSING (Thành viên 1 - DataProcessor):")
    print(f"   - Trung bình (Mean) : {d_stats['mean']:.4f} ms")
    print(f"   - Trung vị (P50)    : {d_stats['p50']:.4f} ms")
    print(f"   - Phân vị P95       : {d_stats['p95']:.4f} ms")
    print(f"   - Phân vị P99       : {d_stats['p99']:.4f} ms")
    print(f"   - Nhỏ nhất / Lớn nhất: {d_stats['min']:.4f} ms / {d_stats['max']:.4f} ms")

    print("\n2. ĐỘ TRỄ GRAPHSAGE INFERENCE (Thành viên 2/3):")
    print(f"   - Trung bình (Mean) : {i_stats['mean']:.4f} ms")
    print(f"   - Trung vị (P50)    : {i_stats['p50']:.4f} ms")
    print(f"   - Phân vị P95       : {i_stats['p95']:.4f} ms")
    print(f"   - Phân vị P99       : {i_stats['p99']:.4f} ms")
    print(f"   - Nhỏ nhất / Lớn nhất: {i_stats['min']:.4f} ms / {i_stats['max']:.4f} ms")

    print("\n3. TỔNG ĐỘ TRỄ TOÀN TRÌNH (END-TO-END LATENCY):")
    print(f"   - Trung bình (Mean) : {t_stats['mean']:.4f} ms")
    print(f"   - Trung vị (P50)    : {t_stats['p50']:.4f} ms")
    print(f"   - Phân vị P95       : {t_stats['p95']:.4f} ms")
    print(f"   - Phân vị P99       : {t_stats['p99']:.4f} ms")
    print(f"   - Nhỏ nhất / Lớn nhất: {t_stats['min']:.4f} ms / {t_stats['max']:.4f} ms")

    sla_target = 100.0  # ms
    is_sla_met = t_stats["p99"] < sla_target
    status_str = (
        "ĐẠT CHUẨN KỸ THUẬT (< 100ms)"
        if is_sla_met
        else "CẢNH BÁO: VƯỢT QUÁ SLA"
    )
    print("\n" + "-" * 65)
    print(
        f"KẾT LUẬN TIÊU CHUẨN SLA (P99 < {sla_target}ms): {status_str} "
        f"[P99 = {t_stats['p99']:.2f}ms]"
    )
    print("-" * 65)


def _positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("giá trị phải là một số nguyên") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("giá trị phải lớn hơn 0")
    return parsed_value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Đọc tuần tự giao dịch CSV qua pipeline kiểm thử thời gian thực."
    )
    parser.add_argument(
        "--file_path",
        default=str(DEFAULT_FILE_PATH),
        help="đường dẫn tệp CSV đầu vào (mặc định: data/processed/test_cleaned.csv)",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="số giao dịch tối đa cần xử lý",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="in chi tiết tối đa 10 giao dịch đầu tiên",
    )
    parser.add_argument(
        "--enable-xai",
        action="store_true",
        help="bật GNNExplainer khi phát hiện giao dịch gian lận",
    )
    parser.add_argument(
        "--xai-epochs",
        type=_positive_int,
        default=DEFAULT_XAI_EPOCHS,
        help="số epoch GNNExplainer khi transaction bị đánh dấu fraud",
    )
    parser.add_argument(
        "--xai-num-hops",
        type=_positive_int,
        default=2,
        help="bán kính subgraph realtime đưa vào GNNExplainer",
    )
    parser.add_argument(
        "--xai-output-dir",
        type=Path,
        default=None,
        help="thư mục lưu explanation realtime; bỏ qua nếu không truyền",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    try:
        metrics = load_and_simulate_transactions(
            file_path=args.file_path,
            limit=args.limit,
            verbose=args.verbose,
            enable_xai=args.enable_xai,
            xai_epochs=args.xai_epochs,
            xai_num_hops=args.xai_num_hops,
            xai_output_dir=args.xai_output_dir,
        )
    except (
        FileNotFoundError,
        PermissionError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return

    print_summary(*metrics)


if __name__ == "__main__":
    main()
