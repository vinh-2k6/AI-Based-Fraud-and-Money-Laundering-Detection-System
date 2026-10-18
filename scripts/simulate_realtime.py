"""Mô phỏng luồng giao dịch thời gian thực từ tệp CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph, to_undirected

from src.gnn_explainer import FraudGNNExplainer
from src.models.gnn_model import FraudDetectionGNN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
_XAI_EXPLAINER: FraudGNNExplainer | None = None


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

    for path, description in (
        (DEFAULT_GRAPH_PATH, "đồ thị train"),
        (DEFAULT_MAPPING_PATH, "account mapping"),
        (DEFAULT_CHECKPOINT_PATH, "GraphSAGE checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {description}: {path}. "
                "Hãy chạy pipeline xây dựng graph và huấn luyện model trước."
            )

    graph = torch.load(DEFAULT_GRAPH_PATH, map_location="cpu", weights_only=False)
    with DEFAULT_MAPPING_PATH.open("rb") as mapping_file:
        mapping = pickle.load(mapping_file)

    if not isinstance(mapping, dict) or "account_to_idx" not in mapping:
        raise ValueError(
            f"Định dạng account mapping không hợp lệ: {DEFAULT_MAPPING_PATH}"
        )
    account_to_idx = mapping["account_to_idx"]
    if not isinstance(account_to_idx, dict):
        raise ValueError(
            f"account_to_idx trong mapping không phải dictionary: {DEFAULT_MAPPING_PATH}"
        )

    checkpoint = torch.load(
        DEFAULT_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Định dạng checkpoint GraphSAGE không hợp lệ: {DEFAULT_CHECKPOINT_PATH}"
        )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint không chứa model_state_dict: {DEFAULT_CHECKPOINT_PATH}"
        )

    in_channels = int(checkpoint.get("in_channels", graph.x.shape[1]))
    hidden_channels = int(checkpoint.get("hidden_channels", 64))
    if graph.x.shape[1] != in_channels:
        raise ValueError(
            "Số chiều node feature không khớp checkpoint: "
            f"graph={graph.x.shape[1]}, checkpoint={in_channels}"
        )

    model = FraudDetectionGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
    )
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        node_embeddings = model.conv1(graph.x, graph.edge_index).relu()
        node_embeddings = model.conv2(
            node_embeddings, graph.edge_index
        ).relu()

    _INFERENCE_RUNTIME = {
        "graph": graph,
        "account_to_idx": account_to_idx,
        "node_embeddings": node_embeddings,
        "classifier": model.classifier,
        "model": model,
    }
    return _INFERENCE_RUNTIME


def _build_transaction_subgraph(
    graph: Data,
    sender_idx: int,
    receiver_idx: int,
    account_to_idx: dict[int, int],
    num_hops: int = 2,
) -> Data:
    """Build a local graph around a realtime transaction."""
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
    """Run XAI lazily; this function is called only for fraud predictions."""
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


def infer_transaction(
    tx_json: str,
    threshold: float = DEFAULT_THRESHOLD,
    xai_epochs: int = DEFAULT_XAI_EPOCHS,
    xai_num_hops: int = 2,
    xai_output_path: str | Path | None = None,
) -> dict[str, object]:
    """Dự đoán một giao dịch bằng GraphSAGE trên graph train.

    ``tx_json`` phải chứa ``Sender_account`` và ``Receiver_account``. Graph
    train được dùng làm ngữ cảnh để không dùng giao dịch tương lai khi mô
    phỏng streaming.
    """
    if not 0.0 <= threshold <= 1.0 or not math.isfinite(threshold):
        raise ValueError("threshold phải là số hữu hạn trong khoảng [0, 1]")

    try:
        transaction = json.loads(tx_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tx_json không phải JSON hợp lệ: {exc.msg}") from exc
    if not isinstance(transaction, dict):
        raise ValueError("tx_json phải mã hóa một JSON object")

    required_fields = ("Sender_account", "Receiver_account")
    missing_fields = [field for field in required_fields if field not in transaction]
    if missing_fields:
        raise ValueError(
            "Giao dịch thiếu trường bắt buộc: " + ", ".join(missing_fields)
        )

    try:
        sender_account = int(transaction["Sender_account"])
        receiver_account = int(transaction["Receiver_account"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Sender_account và Receiver_account phải là số nguyên"
        ) from exc

    runtime = _load_inference_runtime()
    account_to_idx = runtime["account_to_idx"]
    missing_accounts = [
        account
        for account in (sender_account, receiver_account)
        if account not in account_to_idx
    ]
    if missing_accounts:
        unique_missing = dict.fromkeys(missing_accounts)
        raise ValueError(
            "Không tìm thấy Account ID trong mapping: "
            + ", ".join(str(account) for account in unique_missing)
        )

    sender_idx = int(account_to_idx[sender_account])
    receiver_idx = int(account_to_idx[receiver_account])
    node_embeddings = runtime["node_embeddings"]
    node_count = int(node_embeddings.size(0))
    if not 0 <= sender_idx < node_count or not 0 <= receiver_idx < node_count:
        raise ValueError(
            "Account mapping trỏ tới node index ngoài phạm vi graph train"
        )

    edge_embedding = torch.cat(
        [
            node_embeddings[sender_idx],
            node_embeddings[receiver_idx],
        ],
        dim=-1,
    )
    with torch.no_grad():
        logit = runtime["classifier"](edge_embedding)
        probability = float(torch.sigmoid(logit).reshape(-1)[0].item())

    result: dict[str, object] = {
        "sender_account": sender_account,
        "receiver_account": receiver_account,
        "fraud_probability": probability,
        "is_fraud": probability >= threshold,
        "threshold": threshold,
    }
    if not result["is_fraud"]:
        return result

    sub_data = _build_transaction_subgraph(
        graph=runtime["graph"],
        sender_idx=sender_idx,
        receiver_idx=receiver_idx,
        account_to_idx=account_to_idx,
        num_hops=xai_num_hops,
    )
    explanation = _explain_transaction(sub_data, epochs=xai_epochs)
    if xai_output_path is not None:
        FraudGNNExplainer.save_explanation(
            explanation,
            xai_output_path,
            source_subgraph="realtime_transaction",
        )
        result["xai_output_path"] = str(xai_output_path)
    result["explanation"] = explanation
    return result


def process_transaction(
    row: dict[str, str | None],
    verbose: bool = False,
    xai_epochs: int = DEFAULT_XAI_EPOCHS,
    xai_num_hops: int = 2,
    xai_output_dir: Path | None = None,
) -> tuple[str, float]:
    """Phân tích, chuyển một dòng CSV thành JSON và trả về JSON cùng độ trễ."""
    start_time = time.perf_counter()
    parsed_row = parse_row(row)
    tx_json = json.dumps(parsed_row, ensure_ascii=False)
    xai_output_path = None
    if xai_output_dir is not None:
        xai_output_path = xai_output_dir / (
            f"realtime_{parsed_row['Sender_account']}_"
            f"{parsed_row['Receiver_account']}.pt"
        )
    inference = infer_transaction(
        tx_json,
        xai_epochs=xai_epochs,
        xai_num_hops=xai_num_hops,
        xai_output_path=xai_output_path,
    )
    end_time = time.perf_counter()
    latency = end_time - start_time
    if verbose:
        print(
            "Kết quả GraphSAGE: "
            f"fraud_probability={inference['fraud_probability']:.6f}, "
            f"is_fraud={inference['is_fraud']}"
        )
        if inference["is_fraud"]:
            print("Đã chạy GNNExplainer cho giao dịch bị đánh dấu fraud")
    return tx_json, latency


def _repository_root() -> Path:
    """Trả về thư mục gốc repository dựa trên vị trí của script."""
    return Path(__file__).resolve().parent.parent


def _resolve_file_path(file_path: str | Path) -> Path:
    """Phân giải đường dẫn đầu vào tương đối từ thư mục gốc repository."""
    path = Path(file_path)
    return path if path.is_absolute() else _repository_root() / path


def _validate_row(
    row: dict[str | None, str | None],
    fieldnames: list[str],
) -> dict[str, str | None]:
    """Từ chối các dòng có cấu trúc CSV không khớp với header."""
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
    xai_epochs: int = DEFAULT_XAI_EPOCHS,
    xai_num_hops: int = 2,
    xai_output_dir: str | Path | None = None,
) -> tuple[list[float], int, int, float]:
    """Đọc tuần tự giao dịch từ CSV và thu thập số liệu xử lý."""
    if limit is not None and limit <= 0:
        raise ValueError("limit phải lớn hơn 0")

    latencies: list[float] = []
    total_rows = 0
    error_rows = 0
    start_time = time.perf_counter()
    resolved_path = _resolve_file_path(file_path)
    resolved_xai_dir = (
        _resolve_file_path(xai_output_dir)
        if xai_output_dir is not None
        else None
    )

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

            for raw_row in reader:
                total_rows += 1
                try:
                    row = _validate_row(raw_row, fieldnames)
                    tx_json, latency = process_transaction(
                        row,
                        verbose=verbose,
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

                latencies.append(latency)
                if verbose and len(latencies) <= MAX_VERBOSE_ROWS:
                    print(
                        f"Giao dịch {len(latencies)}: "
                        f"{tx_json} (độ trễ={latency * 1000:.3f} ms)"
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
    return latencies, total_rows, error_rows, total_time


def print_summary(
    latencies: list[float],
    total_rows: int,
    error_rows: int,
    total_time: float,
) -> None:
    """In thống kê xử lý giao dịch và hiệu năng."""
    successful_rows = len(latencies)
    print("\nTóm tắt mô phỏng")
    print(f"Tổng số dòng đã đọc: {total_rows}")
    print(f"Số giao dịch thành công: {successful_rows}")
    print(f"Số dòng lỗi/bỏ qua: {error_rows}")
    print(f"Tổng thời gian chạy: {total_time * 1000:.3f} ms")

    if not latencies:
        print("Độ trễ trung bình: N/A")
        print("Độ trễ nhỏ nhất: N/A")
        print("Độ trễ lớn nhất: N/A")
        print("Độ lệch chuẩn: N/A")
        return

    latency_ms = [latency * 1000 for latency in latencies]
    standard_deviation = (
        statistics.stdev(latency_ms) if len(latency_ms) > 1 else 0.0
    )
    print(f"Độ trễ trung bình: {statistics.mean(latency_ms):.3f} ms")
    print(f"Độ trễ nhỏ nhất: {min(latency_ms):.3f} ms")
    print(f"Độ trễ lớn nhất: {max(latency_ms):.3f} ms")
    print(f"Độ lệch chuẩn: {standard_deviation:.3f} ms")


def _positive_int(value: str) -> int:
    """Phân tích giá trị số nguyên dương cho argparse."""
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit phải là một số nguyên") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("limit phải lớn hơn 0")
    return parsed_value


def build_argument_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Đọc tuần tự giao dịch CSV qua pipeline kiểm thử thời gian thực."
    )
    parser.add_argument(
        "--file_path",
        default=str(DEFAULT_FILE_PATH),
        help="đường dẫn tệp CSV đầu vào (mặc định: data/test_cleaned.csv)",
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
    """Phân tích tham số, chạy mô phỏng và in phần tóm tắt."""
    args = build_argument_parser().parse_args()
    try:
        metrics = load_and_simulate_transactions(
            file_path=args.file_path,
            limit=args.limit,
            verbose=args.verbose,
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
