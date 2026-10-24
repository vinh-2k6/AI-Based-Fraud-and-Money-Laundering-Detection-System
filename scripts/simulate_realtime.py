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
import os
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
from src.gemini_client import GeminiClient
from src.transaction_reporter import DEFAULT_PROMPT, TransactionReporter
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
DEFAULT_THRESHOLD = 0.66
DEFAULT_XAI_EPOCHS = 200
MAX_VERBOSE_ROWS = 10

_INFERENCE_RUNTIME: dict[str, Any] | None = None
_DATA_PROCESSOR: DataProcessor | None = None
_XAI_EXPLAINER: FraudGNNExplainer | None = None
_REASON_CODE_GENERATOR = ReasonCodeGenerator()
_TRANSACTION_REPORTER: TransactionReporter | None = None
_LAST_LATENCY_RECORDS: list[dict[str, object]] = []
_LAST_BENCHMARK_SUMMARY: dict[str, object] = {}


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

    graphsage_start = time.perf_counter()
    with torch.no_grad():
        logit = runtime["classifier"](edge_embedding)
        probability = float(torch.sigmoid(logit).reshape(-1)[0].item())
    graphsage_ms = (time.perf_counter() - graphsage_start) * 1000.0

    is_fraud = probability >= threshold
    result: dict[str, object] = {
        "sender_account": processed_tx.sender_account,
        "receiver_account": processed_tx.receiver_account,
        "fraud_probability": probability,
        "is_fraud": is_fraud,
        "threshold": threshold,
        "is_cold_start": processed_tx.is_new_sender or processed_tx.is_new_receiver,
        "latency_breakdown": {
            "graphsage_inference_ms": graphsage_ms,
            "xai_subgraph_ms": 0.0,
            "gnnexplainer_ms": 0.0,
            "reason_code_ms": 0.0,
            "reason_code_output_ms": 0.0,
            "explanation_generated": False,
        },
    }

    # Kích hoạt GNNExplainer khi phát hiện gian lận và được yêu cầu
    if is_fraud and enable_xai:
        try:
            if sender_idx < node_count and receiver_idx < node_count:
                subgraph_start = time.perf_counter()
                sub_data = _build_transaction_subgraph(
                    graph=runtime["graph"],
                    sender_idx=sender_idx,
                    receiver_idx=receiver_idx,
                    account_to_idx=runtime["account_to_idx"],
                    num_hops=xai_num_hops,
                )
                breakdown = result["latency_breakdown"]
                breakdown["xai_subgraph_ms"] = (
                    time.perf_counter() - subgraph_start
                ) * 1000.0
                explainer_start = time.perf_counter()
                explanation = _explain_transaction(sub_data, epochs=xai_epochs)
                breakdown["gnnexplainer_ms"] = (
                    time.perf_counter() - explainer_start
                ) * 1000.0
                reason_start = time.perf_counter()
                reason_codes = _REASON_CODE_GENERATOR.generate(
                    explanation,
                    subgraph=sub_data,
                    transaction_features={
                        **processed_tx.feature_dict,
                        "is_new_sender": processed_tx.is_new_sender,
                        "is_new_receiver": processed_tx.is_new_receiver,
                    },
                )
                breakdown["reason_code_ms"] = (
                    time.perf_counter() - reason_start
                ) * 1000.0
                breakdown["explanation_generated"] = True
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
    mode: str = "latency",
    reporter: TransactionReporter | None = None,
    report_dir: Path | None = None,
    reason_codes_output_dir: Path | None = None,
    force_report: bool = False,
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
    breakdown = inference["latency_breakdown"]
    breakdown["data_processing_ms"] = data_proc_latency * 1000.0
    breakdown["total_model_xai_ms"] = (data_proc_latency + inference_latency) * 1000.0
    breakdown["gemini_called"] = False
    breakdown["gemini_report_ms"] = 0.0
    breakdown["total_with_gemini_ms"] = breakdown["total_model_xai_ms"]
    breakdown["report_generated"] = False

    if mode not in {"latency", "local-reason-codes", "gemini-report"}:
        raise ValueError(
            "mode phải là 'latency', 'local-reason-codes' hoặc 'gemini-report'"
        )

    if mode in {"local-reason-codes", "gemini-report"} and inference.get("is_fraud") is True:
        explanation = inference.get("explanation")
        reason_codes = inference.get("reason_codes")
        if (
            mode == "local-reason-codes"
            and explanation is not None
            and reason_codes is not None
            and reason_codes_output_dir is not None
        ):
            reason_output_start = time.perf_counter()
            transaction_id = (
                f"{parsed_row.get('Sender_account', 'unknown')}_"
                f"{parsed_row.get('Receiver_account', 'unknown')}"
            )
            document = _REASON_CODE_GENERATOR.build_document(
                explanation,
                transaction_features={
                    **processed_tx.feature_dict,
                    "is_new_sender": processed_tx.is_new_sender,
                    "is_new_receiver": processed_tx.is_new_receiver,
                },
                transaction_id=transaction_id,
                top_k=5,
            )
            reason_codes_output_dir.mkdir(parents=True, exist_ok=True)
            safe_id = "".join(
                character if character.isalnum() or character in "._-" else "_"
                for character in transaction_id
            )
            reason_code_path = reason_codes_output_dir / f"transaction_{safe_id}.json"
            reason_code_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            inference["reason_code_output_path"] = str(reason_code_path)
            inference["reason_code_status"] = "success"
            reason_output_ms = (
                time.perf_counter() - reason_output_start
            ) * 1000.0
            breakdown["reason_code_output_ms"] = reason_output_ms
            breakdown["reason_code_ms"] += reason_output_ms
            inference_latency += reason_output_ms / 1000.0
            breakdown["total_model_xai_ms"] += reason_output_ms
            breakdown["total_with_gemini_ms"] = breakdown["total_model_xai_ms"]
        elif mode == "local-reason-codes" and inference.get("xai_error"):
            inference["reason_code_status"] = "failed"
        elif mode == "local-reason-codes":
            inference["reason_code_status"] = "success"

    if mode == "gemini-report" and inference.get("is_fraud") is True:
        explanation = inference.get("explanation")
        reason_codes = inference.get("reason_codes")
        if explanation is not None and reason_codes is not None and reporter is not None:
            report_start = time.perf_counter()
            transaction_id = (
                f"{parsed_row.get('Sender_account', 'unknown')}_"
                f"{parsed_row.get('Receiver_account', 'unknown')}"
            )
            try:
                document = _REASON_CODE_GENERATOR.build_document(
                    explanation,
                    transaction_features={
                        **processed_tx.feature_dict,
                        "is_new_sender": processed_tx.is_new_sender,
                        "is_new_receiver": processed_tx.is_new_receiver,
                    },
                    transaction_id=transaction_id,
                    top_k=reporter.top_k,
                )
                report_result = reporter.report_document(
                    document,
                    report_dir or DEFAULT_XAI_OUTPUT_DIR.parent / "transaction_reports",
                    force=force_report,
                )
                inference["report_status"] = report_result.get("status", "unknown")
                inference["report_generated"] = report_result.get("status") == "success"
                if report_result.get("status") == "failed":
                    inference["gemini_error"] = report_result.get("error_type", "report_failed")
                breakdown["gemini_called"] = report_result.get("status") != "skipped"
                breakdown["report_generated"] = inference["report_generated"]
            except Exception as exc:
                inference["gemini_error"] = str(exc)
                inference["report_status"] = "failed"
            finally:
                breakdown["gemini_report_ms"] = (
                    time.perf_counter() - report_start
                ) * 1000.0
                breakdown["total_with_gemini_ms"] = (
                    breakdown["total_model_xai_ms"] + breakdown["gemini_report_ms"]
                )
        elif inference.get("xai_error"):
            inference["report_status"] = "skipped_xai_error"

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
    metrics_output: str | Path | None = None,
    summary_output: str | Path | None = None,
    warmup_runs: int = 0,
    include_cold_start: bool = False,
    latency_plot_output: str | Path | None = None,
    mode: str = "latency",
    gemini_model: str | None = None,
    gemini_timeout: float = 30.0,
    gemini_retry_count: int = 2,
    report_dir: str | Path | None = None,
    reason_codes_output_dir: str | Path | None = None,
    prompt_file: str | Path | None = None,
    top_k: int = 5,
    max_prompt_chars: int = 12000,
    force_report: bool = False,
) -> tuple[list[float], list[float], list[float], int, int, float]:
    """
    Đọc tuần tự giao dịch từ CSV và thu thập số liệu chi tiết về độ trễ.

    Returns:
        (total_latencies, data_proc_latencies, inference_latencies, total_rows, error_rows, total_time)
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit phải lớn hơn 0")
    if warmup_runs < 0:
        raise ValueError("warmup_runs không được âm")
    if mode not in {"latency", "local-reason-codes", "gemini-report"}:
        raise ValueError(
            "mode phải là 'latency', 'local-reason-codes' hoặc 'gemini-report'"
        )

    reporter = None
    if mode == "gemini-report":
        reporter = TransactionReporter(
            GeminiClient(
                model=gemini_model,
                timeout=gemini_timeout,
                retry_count=gemini_retry_count,
            ),
            prompt_file=prompt_file or DEFAULT_PROMPT,
            top_k=top_k,
            max_prompt_chars=max_prompt_chars,
        )

    print("Khởi động môi trường suy luận (Warming up inference runtime & DataProcessor)...")
    runtime = _load_inference_runtime()
    get_data_processor()
    for _ in range(warmup_runs):
        embeddings = runtime["node_embeddings"]
        if embeddings.size(0) >= 2:
            warmup_embedding = torch.cat((embeddings[0], embeddings[1]), dim=-1)
            with torch.no_grad():
                runtime["classifier"](warmup_embedding)

    total_latencies: list[float] = []
    data_proc_latencies: list[float] = []
    inference_latencies: list[float] = []
    records: list[dict[str, object]] = []

    resolved_xai_dir: Path | None = None
    if enable_xai and xai_output_dir is not None:
        resolved_xai_dir = (
            Path(xai_output_dir)
            if Path(xai_output_dir).is_absolute()
            else _repository_root() / xai_output_dir
        )
    resolved_reason_codes_dir: Path | None = None
    if reason_codes_output_dir is not None:
        resolved_reason_codes_dir = (
            Path(reason_codes_output_dir)
            if Path(reason_codes_output_dir).is_absolute()
            else _repository_root() / reason_codes_output_dir
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
                    tx_json, d_lat, i_lat, t_lat, inference = process_transaction(
                        row,
                        verbose=verbose,
                        enable_xai=enable_xai or mode in {
                            "local-reason-codes",
                            "gemini-report",
                        },
                        xai_epochs=xai_epochs,
                        xai_num_hops=xai_num_hops,
                        xai_output_dir=resolved_xai_dir,
                        mode=mode,
                        reporter=reporter,
                        report_dir=Path(report_dir) if report_dir else None,
                        reason_codes_output_dir=resolved_reason_codes_dir,
                        force_report=force_report,
                    )
                except Exception as exc:
                    error_rows += 1
                    records.append({
                        "transaction_index": total_rows,
                        "sender_account": raw_row.get("Sender_account") or "",
                        "receiver_account": raw_row.get("Receiver_account") or "",
                        "fraud_probability": "",
                        "is_fraud": "",
                        "data_processing_ms": "",
                        "graphsage_inference_ms": "",
                        "xai_subgraph_ms": "",
                        "gnnexplainer_ms": "",
                        "reason_code_ms": "",
                        "total_ms": "",
                        "explanation_generated": False,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "is_cold_start": False,
                    })
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
                breakdown = inference.get("latency_breakdown", {})
                records.append({
                    "transaction_index": total_rows,
                    "sender_account": inference.get("sender_account", ""),
                    "receiver_account": inference.get("receiver_account", ""),
                    "fraud_probability": inference.get("fraud_probability", ""),
                    "is_fraud": inference.get("is_fraud", ""),
                    **breakdown,
                    "status": "success",
                    "error_type": "",
                    "error_message": inference.get("xai_error", ""),
                    "report_status": inference.get("report_status", ""),
                    "gemini_error": inference.get("gemini_error", ""),
                    "reason_code_status": inference.get("reason_code_status", ""),
                    "reason_code_output_path": inference.get(
                        "reason_code_output_path", ""
                    ),
                    "reason_codes": "|".join(
                        item.get("code", "")
                        for item in inference.get("reason_codes", {}).get(
                            "reason_codes", []
                        )
                        if isinstance(item, dict)
                    ),
                    "is_cold_start": inference.get("is_cold_start", False),
                })

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
    _write_benchmark_outputs(
        records,
        total_rows=total_rows,
        error_rows=error_rows,
        total_time_ms=total_time * 1000.0,
        metrics_output=metrics_output,
        summary_output=summary_output,
        enable_xai=enable_xai,
        xai_epochs=xai_epochs,
        xai_num_hops=xai_num_hops,
        warmup_runs=warmup_runs,
        include_cold_start=include_cold_start,
        latency_plot_output=latency_plot_output,
        mode=mode,
    )
    global _LAST_LATENCY_RECORDS
    _LAST_LATENCY_RECORDS = records
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
    fraud_transactions: int | None = None,
) -> None:
    """In thống kê phân rã độ trễ và đánh giá tiêu chuẩn SLA (< 100ms)."""
    successful_rows = len(total_latencies)
    print("\n" + "=" * 65)
    print("BÁO CÁO KẾT QUẢ KIỂM THỬ ĐỘ TRỄ TOÀN TRÌNH (STREAMING E2E)")
    print("=" * 65)
    print(f"Tổng số dòng đã nạp     : {total_rows:,}")
    print(f"Số giao dịch thành công : {successful_rows:,}")
    print(f"Số dòng lỗi/bỏ qua      : {error_rows:,}")
    if fraud_transactions is None:
        fraud_transactions = sum(
            record.get("is_fraud") is True
            for record in _LAST_LATENCY_RECORDS
            if record.get("status") == "success"
        )
    fraud_rate = (
        fraud_transactions / successful_rows * 100.0
        if successful_rows
        else 0.0
    )
    print(
        f"Số giao dịch nghi ngờ gian lận: {fraud_transactions:,} "
        f"({fraud_rate:.2f}%)"
    )
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


def _latency_stats(
    records: list[dict[str, object]],
    field: str,
    include_cold_start: bool = False,
) -> dict[str, float]:
    values = [
        float(record[field])
        for record in records
        if record.get("status") == "success"
        and record.get(field) not in ("", None)
        and (include_cold_start or not record.get("is_cold_start", False))
    ]
    if not values:
        return {key: 0.0 for key in ("mean", "p50", "p95", "p99", "min", "max")}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _write_benchmark_outputs(
    records: list[dict[str, object]],
    total_rows: int,
    error_rows: int,
    total_time_ms: float,
    metrics_output: str | Path | None,
    summary_output: str | Path | None,
    enable_xai: bool,
    xai_epochs: int,
    xai_num_hops: int,
    warmup_runs: int,
    include_cold_start: bool,
    latency_plot_output: str | Path | None = None,
    mode: str = "latency",
) -> None:
    global _LAST_BENCHMARK_SUMMARY
    successful = [r for r in records if r.get("status") == "success"]
    fraud = [r for r in successful if r.get("is_fraud") is True]
    explained = [r for r in successful if r.get("explanation_generated") is True]
    report_success = [
        r for r in fraud if r.get("report_status") == "success"
    ]
    report_failed = [
        r for r in fraud if r.get("report_status") == "failed"
    ]
    fields = [
        "transaction_index", "sender_account", "receiver_account",
        "fraud_probability", "is_fraud", "data_processing_ms",
        "graphsage_inference_ms", "xai_subgraph_ms", "gnnexplainer_ms",
        "reason_code_ms", "reason_code_output_ms", "total_ms",
        "total_model_xai_ms",
        "gemini_called", "gemini_report_ms", "total_with_gemini_ms",
        "explanation_generated", "report_generated", "report_status",
        "status", "error_type", "error_message",
    ]
    if metrics_output is not None:
        path = _resolve_file_path(metrics_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in fields} for row in records)

    summary = {
        "total_rows_read": total_rows,
        "successful_transactions": len(successful),
        "error_transactions": error_rows,
        "fraud_transactions": len(fraud),
        "xai_transactions": len(explained),
        "gemini_success_transactions": len(report_success),
        "gemini_failed_transactions": len(report_failed),
        "total_time_ms": total_time_ms,
        "device": str(_INFERENCE_RUNTIME["node_embeddings"].device) if _INFERENCE_RUNTIME else "unknown",
        "xai_enabled": enable_xai,
        "mode": mode,
        "xai_epochs": xai_epochs,
        "xai_num_hops": xai_num_hops,
        "warmup_runs": warmup_runs,
        "warmup_excluded_from_statistics": True,
        "cold_start_included": include_cold_start,
        "statistics": {
            "all": {field: _latency_stats(successful, field, include_cold_start) for field in (
                "data_processing_ms", "graphsage_inference_ms", "xai_subgraph_ms",
                "gnnexplainer_ms", "reason_code_ms", "reason_code_output_ms",
                "total_ms",
                "total_model_xai_ms", "gemini_report_ms", "total_with_gemini_ms")},
            "fraud": {field: _latency_stats(fraud, field, include_cold_start) for field in (
                "data_processing_ms", "graphsage_inference_ms", "xai_subgraph_ms",
                "gnnexplainer_ms", "reason_code_ms", "reason_code_output_ms",
                "total_ms",
                "total_model_xai_ms", "gemini_report_ms", "total_with_gemini_ms")},
            "xai": {field: _latency_stats(explained, field, include_cold_start) for field in (
                "data_processing_ms", "graphsage_inference_ms", "xai_subgraph_ms",
                "gnnexplainer_ms", "reason_code_ms", "reason_code_output_ms",
                "total_ms",
                "total_model_xai_ms", "gemini_report_ms", "total_with_gemini_ms")},
            "gemini_success": {
                field: _latency_stats(report_success, field, include_cold_start)
                for field in ("total_ms", "total_model_xai_ms", "gemini_report_ms", "total_with_gemini_ms")
            },
            "gemini_failed": {
                field: _latency_stats(report_failed, field, include_cold_start)
                for field in ("total_ms", "total_model_xai_ms", "gemini_report_ms", "total_with_gemini_ms")
            },
        },
    }
    total_values = [
        float(r["total_ms"])
        for r in successful
        if r.get("total_ms") not in ("", None)
        and (include_cold_start or not r.get("is_cold_start", False))
    ]
    summary["total_under_100ms_ratio"] = (
        sum(value < 100.0 for value in total_values) / len(total_values)
        if total_values else 0.0
    )
    if latency_plot_output is not None:
        plot_path = _write_latency_distribution_plot(
            successful,
            latency_plot_output,
            include_cold_start=include_cold_start,
        )
        summary["latency_distribution_plot"] = str(plot_path)
    for group in summary["statistics"].values():
        if "total_ms" in group:
            group["sla_p95_under_100ms"] = group["total_ms"]["p95"] < 100.0
            group["sla_p99_under_100ms"] = group["total_ms"]["p99"] < 100.0
    _LAST_BENCHMARK_SUMMARY = summary
    if summary_output is not None:
        path = _resolve_file_path(summary_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_latency_distribution_plot(
    records: list[dict[str, object]],
    output_path: str | Path,
    include_cold_start: bool = False,
) -> Path:
    """Vẽ phân phối total latency và các mốc percentile/SLA."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Không thể tạo biểu đồ latency vì thiếu matplotlib. "
            "Cài bằng: python -m pip install matplotlib"
        ) from exc

    values = [
        float(record["total_ms"])
        for record in records
        if record.get("total_ms") not in ("", None)
        and (include_cold_start or not record.get("is_cold_start", False))
    ]
    if not values:
        raise ValueError("Không có latency thành công để tạo biểu đồ")

    array = np.asarray(values, dtype=float)
    percentiles = {
        "P50": float(np.percentile(array, 50)),
        "P95": float(np.percentile(array, 95)),
        "P99": float(np.percentile(array, 99)),
    }
    path = _resolve_file_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 6))
    bins = min(30, max(5, int(np.sqrt(len(array)))))
    axis.hist(array, bins=bins, color="#457b9d", alpha=0.82, edgecolor="white")
    for label, value, color in (
        ("P50", percentiles["P50"], "#2a9d8f"),
        ("P95", percentiles["P95"], "#f4a261"),
        ("P99", percentiles["P99"], "#e63946"),
    ):
        axis.axvline(value, color=color, linestyle="--", linewidth=1.8,
                     label=f"{label}: {value:.2f} ms")
    axis.axvline(
        100.0, color="#6a4c93", linestyle="-.", linewidth=1.8,
        label="SLA: 100 ms",
    )
    axis.set_title("Latency Distribution")
    axis.set_xlabel("End-to-end latency (ms)")
    axis.set_ylabel("Number of transactions")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


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
        "--mode",
        choices=("latency", "local-reason-codes", "gemini-report"),
        default="latency",
        help=(
            "latency, reason code cục bộ không gọi API, "
            "hoặc tạo report Gemini cho fraud"
        ),
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
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="đường dẫn CSV latency theo từng giao dịch",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="đường dẫn JSON summary latency",
    )
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--gemini-timeout", type=float, default=30.0)
    parser.add_argument("--gemini-retry-count", type=int, default=2)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--reason-codes-output-dir",
        type=Path,
        default=None,
        help="thư mục lưu reason-code JSON khi dùng local-reason-codes",
    )
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--max-prompt-chars", type=_positive_int, default=12000)
    parser.add_argument("--force", action="store_true", help="tạo lại report đã cache")
    parser.add_argument(
        "--latency-plot-output",
        type=Path,
        default=None,
        help="đường dẫn PNG biểu đồ phân phối total latency",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="số lần warm-up được loại khỏi thống kê chính",
    )
    parser.add_argument(
        "--include-cold-start",
        action="store_true",
        help="đưa giao dịch cold-start vào thống kê latency",
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
            metrics_output=args.metrics_output,
            summary_output=args.summary_output,
            warmup_runs=args.warmup_runs,
            include_cold_start=args.include_cold_start,
            latency_plot_output=args.latency_plot_output,
            mode=args.mode,
            gemini_model=args.gemini_model,
            gemini_timeout=args.gemini_timeout,
            gemini_retry_count=args.gemini_retry_count,
            report_dir=args.report_dir,
            reason_codes_output_dir=args.reason_codes_output_dir,
            prompt_file=args.prompt_file,
            top_k=args.top_k,
            max_prompt_chars=args.max_prompt_chars,
            force_report=args.force,
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

    print_summary(*metrics, fraud_transactions=len([
        record for record in _LAST_LATENCY_RECORDS
        if record.get("status") == "success" and record.get("is_fraud") is True
    ]))


if __name__ == "__main__":
    main()
