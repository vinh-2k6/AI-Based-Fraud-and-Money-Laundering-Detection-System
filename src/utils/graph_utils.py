"""
src/utils/graph_utils.py
========================
Module tiện ích đồ thị hiệu năng cao phục vụ Streaming & Production Inference

Tính năng cốt lõi:
  1. Caching & Fast Lookup: Nạp mapping và ánh xạ ngược ID tài khoản với độ phức tạp O(1)
  2. Cold-start Handling: Quản lý và gán đặc trưng an toàn cho các tài khoản mới xuất hiện trong luồng streaming
  3. Fast Edge Embedding: Ghép nối vector nhúng hai đầu mút giao dịch với kiểm tra biên an toàn
  4. Memory-safe Subgraph Utilities: Hỗ trợ trích xuất nhanh láng giềng k-hop phục vụ XAI theo thời gian thực
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_undirected

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_PATH = BASE_DIR / "data" / "processed" / "account_mapping.pkl"

# Global in-memory cache to prevent repeated disk I/O in streaming
_MAPPING_CACHE: dict[str, Any] | None = None


def load_account_mapping(
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    use_cache: bool = True,
) -> tuple[dict[int, int], set[int], dict[int, int]]:
    """
    Nạp và cache từ điển ánh xạ tài khoản hai chiều:
      - account_to_idx: map Account ID gốc -> Node index (0..N-1)
      - known_train_accounts: tập hợp các Account ID có trong Base Graph (Train)
      - idx_to_account: map Node index -> Account ID gốc (Reverse Mapping)

    Returns:
        tuple[account_to_idx, known_train_accounts, idx_to_account]
    """
    global _MAPPING_CACHE
    path = Path(mapping_path)

    if use_cache and _MAPPING_CACHE is not None and _MAPPING_CACHE.get("path") == str(path):
        return (
            _MAPPING_CACHE["account_to_idx"],
            _MAPPING_CACHE["known_train_accounts"],
            _MAPPING_CACHE["idx_to_account"],
        )

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file account_mapping.pkl tại: {path}")

    with open(path, "rb") as f:
        raw_data = pickle.load(f)

    if isinstance(raw_data, dict) and "account_to_idx" in raw_data:
        account_to_idx = raw_data["account_to_idx"]
        known_train_accounts = raw_data.get("known_train_accounts", set(account_to_idx.keys()))
    elif isinstance(raw_data, dict):
        account_to_idx = raw_data
        known_train_accounts = set(account_to_idx.keys())
    else:
        raise ValueError(f"Định dạng mapping không hợp lệ trong file: {path}")

    idx_to_account = {idx: acc for acc, idx in account_to_idx.items()}

    _MAPPING_CACHE = {
        "path": str(path),
        "account_to_idx": account_to_idx,
        "known_train_accounts": known_train_accounts,
        "idx_to_account": idx_to_account,
    }

    return account_to_idx, known_train_accounts, idx_to_account


def resolve_node_index(
    account_id: int | str,
    account_to_idx: dict[int, int],
    known_train_accounts: set[int] | None = None,
) -> tuple[int, bool]:
    """
    Tra cứu node index từ Account ID với độ phức tạp O(1).

    Returns:
        tuple[node_idx, is_cold_start]:
          - node_idx: chỉ số node tương ứng nếu tìm thấy, ngược lại -1
          - is_cold_start: True nếu tài khoản là mới (chưa có trong known_train_accounts)
    """
    try:
        acc_int = int(account_id)
    except (ValueError, TypeError):
        return -1, True

    if acc_int not in account_to_idx:
        return -1, True

    node_idx = account_to_idx[acc_int]
    is_cold_start = False
    if known_train_accounts is not None:
        is_cold_start = acc_int not in known_train_accounts

    return node_idx, is_cold_start


def resolve_account_id(
    node_idx: int,
    idx_to_account: dict[int, int],
) -> int | str:
    """Ánh xạ ngược node index về Account ID gốc."""
    return idx_to_account.get(node_idx, f"Node_{node_idx}")


def create_cold_start_node_features(
    num_nodes: int = 1,
    feature_dim: int = 8,
) -> torch.Tensor:
    """
    Tạo vector đặc trưng chuẩn cho tài khoản cold-start (mới xuất hiện):
      - 7 chiều đầu (degree, amount, fraud ratio) = 0.0
      - Chiều cuối (is_new_node) = 1.0
    """
    features = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
    features[:, -1] = 1.0
    return features


def build_edge_embedding(
    node_embeddings: torch.Tensor,
    sender_idx: int,
    receiver_idx: int,
) -> torch.Tensor:
    """
    Ghép nối vector nhúng của node gửi và node nhận để tạo đặc trưng cạnh giao dịch.
    Hỗ trợ kiểm tra biên an toàn chống IndexError.
    """
    n_nodes = node_embeddings.size(0)
    if not (0 <= sender_idx < n_nodes and 0 <= receiver_idx < n_nodes):
        raise IndexError(
            f"Chỉ số node ({sender_idx}, {receiver_idx}) vượt quá số lượng node trong embeddings ({n_nodes})"
        )

    return torch.cat([node_embeddings[sender_idx], node_embeddings[receiver_idx]], dim=-1)


def fast_extract_ego_subgraph(
    sender_idx: int,
    receiver_idx: int,
    edge_index: torch.Tensor,
    num_hops: int = 2,
    undirected_edge_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """
    Trích xuất nhanh tập node và cạnh của ego-subgraph xung quanh giao dịch mục tiêu.

    Returns:
        tuple[subset_nodes, sub_edge_index, relabel_map]:
          - subset_nodes: các node gốc trong subgraph
          - sub_edge_index: cạnh sau khi relabel (0..K-1)
          - relabel_map: mapping từ node gốc -> node index trong subgraph
    """
    search_index = undirected_edge_index if undirected_edge_index is not None else edge_index
    sub_nodes, _, _, _ = k_hop_subgraph(
        node_idx=[sender_idx, receiver_idx],
        num_hops=num_hops,
        edge_index=search_index,
        relabel_nodes=False,
    )

    relabel_map = {int(nid): i for i, nid in enumerate(sub_nodes.tolist())}
    mask = torch.isin(edge_index[0], sub_nodes) & torch.isin(edge_index[1], sub_nodes)
    filtered_edges = edge_index[:, mask]

    src_relabelled = torch.tensor([relabel_map[int(u)] for u in filtered_edges[0]], dtype=torch.long)
    dst_relabelled = torch.tensor([relabel_map[int(v)] for v in filtered_edges[1]], dtype=torch.long)
    sub_edge_index = torch.stack([src_relabelled, dst_relabelled], dim=0)

    return sub_nodes, sub_edge_index, relabel_map
