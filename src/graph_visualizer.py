"""
src/graph_visualizer.py
========================
Module trực quan hóa đồ thị con giải thích XAI

Nhiệm vụ:
  1. Ánh xạ ngược chỉ số tensor nội bộ về ID tài khoản gốc
  2. Chuyển đổi đồ thị con PyG sang NetworkX DiGraph
  3. Trực quan hóa và làm nổi bật các cạnh có trọng số edge_mask cao
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Chế độ headless, lưu ảnh trực tiếp không cần mở cửa sổ GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"
DEFAULT_MAPPING_PATH = PROC_DIR / "account_mapping.pkl"

from src.utils.graph_utils import load_account_mapping


class SubgraphXaiVisualizer:
    """
    Bộ công cụ ánh xạ ngược và trực quan hóa đồ thị con giải thích GNN.
    """

    def __init__(self, mapping_path: Path | str | None = DEFAULT_MAPPING_PATH) -> None:
        self.idx_to_account: dict[int, Any] = {}
        if mapping_path is not None:
            p = Path(mapping_path)
            if p.exists():
                try:
                    _, _, self.idx_to_account = load_account_mapping(p, use_cache=True)
                except Exception as e:
                    print(f"[Visualizer] Cảnh báo không đọc được account_mapping.pkl: {e}")

    # 1. HÀM ÁNH XẠ NGƯỢC (INVERSE MAPPING)
    def map_tensor_indices_to_accounts(self, sub_data: Data) -> dict[int, str]:
        """
        Ánh xạ ngược chỉ số node trong subgraph (0, 1, 2...) về ID tài khoản gốc.
        """
        node_to_account: dict[int, str] = {}
        num_nodes = sub_data.num_nodes

        # 1. Ưu tiên đọc từ orig_account_ids có sẵn trong subgraph của TV1
        if hasattr(sub_data, "orig_account_ids") and sub_data.orig_account_ids is not None:
            for local_idx, acc_id in enumerate(sub_data.orig_account_ids):
                node_to_account[local_idx] = str(acc_id)
            return node_to_account

        # 2. Tra cứu qua chỉ số gốc (orig_node_indices)
        if hasattr(sub_data, "orig_node_indices") and sub_data.orig_node_indices is not None:
            for local_idx, global_nid in enumerate(sub_data.orig_node_indices.tolist()):
                acc_id = self.idx_to_account.get(global_nid, f"Node_{global_nid}")
                node_to_account[local_idx] = str(acc_id)
            return node_to_account

        # 3. Fallback mặc định
        for i in range(num_nodes):
            node_to_account[i] = f"Node_{i}"
        return node_to_account

    @staticmethod
    def format_account_label(account_id: str, max_len: int = 10) -> str:
        """Rút gọn nhãn tài khoản để tránh đè chữ trên hình vẽ."""
        s = str(account_id)
        return s if len(s) <= max_len else f"{s[:4]}..{s[-3:]}"

    @staticmethod
    def _normalize_edge_importance(
        edge_mask: torch.Tensor | np.ndarray | None,
        num_edges: int,
    ) -> np.ndarray:
        if edge_mask is None:
            return np.zeros(num_edges, dtype=np.float32)
        values = (
            edge_mask.detach().cpu().numpy()
            if isinstance(edge_mask, torch.Tensor)
            else np.asarray(edge_mask)
        )
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != num_edges:
            print(
                f"[Visualizer] Cảnh báo edge_mask có {values.size} phần tử, "
                f"cần {num_edges}; dùng importance mặc định."
            )
            return np.zeros(num_edges, dtype=np.float32)
        values = np.nan_to_num(np.abs(values), nan=0.0, posinf=0.0, neginf=0.0)
        maximum = float(values.max()) if values.size else 0.0
        if maximum <= 0.0:
            return np.zeros(num_edges, dtype=np.float32)
        return np.clip(values / maximum, 0.0, 1.0)

    @staticmethod
    def _get_edge_labels(sub_data: Data, num_edges: int) -> np.ndarray | None:
        labels = getattr(sub_data, "y", None)
        if labels is None:
            print("[Visualizer] Cảnh báo subgraph không có y; dùng màu fallback cho cạnh.")
            return None
        if isinstance(labels, torch.Tensor):
            values = labels.detach().cpu().numpy()
        else:
            values = np.asarray(labels)
        values = np.asarray(values).reshape(-1)
        if values.size != num_edges:
            print(
                f"[Visualizer] Cảnh báo y có {values.size} phần tử, "
                f"cần {num_edges}; dùng màu fallback cho cạnh."
            )
            return None
        return values

    # 2. CHUYỂN ĐỔI PYG SANG NETWORKX VÀ VẼ ĐỒ THỊ
    def plot_subgraph_explanation(
        self,
        sub_data: Data,
        edge_mask: torch.Tensor | np.ndarray | None = None,
        output_path: Path | str | None = None,
        mask_threshold: float = 0.5,
        dpi: int = 300,
        figsize: tuple[int, int] = (12, 9),
        node_importance: torch.Tensor | np.ndarray | None = None,
    ) -> Path:
        """
        Vẽ đồ thị con giải thích bằng NetworkX và Matplotlib.
        Màu cạnh lấy từ sub_data.y; edge_mask chỉ điều chỉnh importance và độ dày.
        """
        edge_index = sub_data.edge_index.cpu().numpy()
        num_edges = edge_index.shape[1]
        num_nodes = sub_data.num_nodes

        edge_importance = self._normalize_edge_importance(edge_mask, num_edges)
        edge_labels = self._get_edge_labels(sub_data, num_edges)

        if node_importance is None and hasattr(sub_data, "node_importance"):
            node_importance = sub_data.node_importance
        if node_importance is not None:
            if isinstance(node_importance, torch.Tensor):
                node_weights = node_importance.detach().cpu().numpy().astype(np.float32)
            else:
                node_weights = np.asarray(node_importance, dtype=np.float32)
            node_weights = node_weights.reshape(-1)
            if node_weights.size != num_nodes:
                raise ValueError(
                    f"node_importance có {node_weights.size} phần tử, cần {num_nodes}"
                )
            node_max = float(node_weights.max()) if node_weights.size else 0.0
            node_cutoff = node_max * mask_threshold
        else:
            node_weights = None
            node_cutoff = float("inf")

        node_accounts = self.map_tensor_indices_to_accounts(sub_data)
        node_labels = {
            nid: self.format_account_label(acc)
            for nid, acc in node_accounts.items()
        }

        # MultiDiGraph giữ lại cả các giao dịch song song giữa cùng một cặp node.
        G = nx.MultiDiGraph()
        for i in range(num_nodes):
            G.add_node(i, label=node_labels[i])

        target_sub_idx = getattr(sub_data, "target_edge_idx_in_sub", -1)
        if not isinstance(target_sub_idx, (int, np.integer)) or not 0 <= target_sub_idx < num_edges:
            target_sub_idx = -1
        for i in range(num_edges):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            G.add_edge(
                u,
                v,
                key=i,
                edge_index=i,
                weight=float(edge_importance[i]),
                label=None if edge_labels is None else int(edge_labels[i] == 1),
                is_target=(i == target_sub_idx),
            )

        # Phân loại màu cho nút trung tâm (Sender/Receiver mục tiêu)
        target_u, target_v = 0, 1
        target_index = getattr(sub_data, "target_edge_label_index", None)
        if isinstance(target_index, torch.Tensor) and target_index.ndim == 2 and target_index.shape[1] > 0:
            target_u = int(target_index[0, 0].item())
            target_v = int(target_index[1, 0].item())

        node_colors = []
        node_sizes = []
        for n in G.nodes():
            if n == target_u:
                node_colors.append("#d90429")  # Đỏ - Sender mục tiêu
                node_sizes.append(1000)
            elif n == target_v:
                node_colors.append("#7209b7")  # Tím - Receiver mục tiêu
                node_sizes.append(1000)
            elif node_weights is not None and node_weights[n] >= node_cutoff:
                node_colors.append("#90be6d")  # Xanh - node quan trọng theo XAI
                node_sizes.append(850)
            elif G.degree(n) > 4:
                node_colors.append("#f77f00")  # Cam - Nút luân chuyển
                node_sizes.append(700)
            else:
                node_colors.append("#8ecae6")  # Xanh nhạt - Nút thông thường
                node_sizes.append(450)

        fraud_edges, normal_edges, fallback_edges = [], [], []
        fraud_widths, normal_widths, fallback_widths = [], [], []
        fraud_colors, normal_colors, fallback_colors = [], [], []
        target_edges = []
        target_colors = []
        for u, v, key, d in G.edges(keys=True, data=True):
            importance = float(d["weight"])
            width = 1.2 + importance * 3.8
            if d["is_target"]:
                width += 1.0
                target_edges.append((u, v, key))
                if edge_labels is None:
                    target_colors.append("#e85d04")
                elif d["label"] == 1:
                    target_colors.append("#d90429")
                else:
                    target_colors.append("#2a9d8f")
            if edge_labels is None:
                fallback_edges.append((u, v, key))
                fallback_widths.append(width)
                fallback_colors.append("#e85d04" if importance >= mask_threshold else "#adb5bd")
            elif d["label"] == 1:
                fraud_edges.append((u, v, key))
                fraud_widths.append(width)
                fraud_colors.append("#d90429" if importance >= mask_threshold else "#ef476f")
            else:
                normal_edges.append((u, v, key))
                normal_widths.append(width)
                normal_colors.append("#2a9d8f" if importance >= mask_threshold else "#8ecae6")

        # Tính bố cục hiển thị (Spring Layout)
        pos = nx.spring_layout(G, k=1.8 / np.sqrt(max(1, num_nodes)), iterations=70, seed=42)

        fig, ax = plt.subplots(figsize=figsize, facecolor="#fafafa")
        ax.set_facecolor("#fafafa")

        # Vẽ cạnh nền mờ
        if fallback_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=fallback_edges, edge_color=fallback_colors,
                width=fallback_widths, alpha=0.75, arrows=True, arrowsize=10, ax=ax,
            )
        if fraud_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=fraud_edges, edge_color=fraud_colors,
                width=fraud_widths, alpha=0.95, arrows=True, arrowsize=16, ax=ax,
            )
        if normal_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=normal_edges, edge_color=normal_colors,
                width=normal_widths, alpha=0.95, arrows=True, arrowsize=16, ax=ax,
            )
        if target_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=target_edges, edge_color=target_colors,
                width=[1.0 + float(G.edges[u, v, key]["weight"]) * 3.8 + 2.0
                       for u, v, key in target_edges],
                style="dashed", arrows=True, arrowsize=18, ax=ax,
            )

        # Vẽ nút và nhãn
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="#2b2d42", linewidths=1.2, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold", font_color="#1d3557", ax=ax)

        # Chú giải (Legend)
        legends = [
            mpatches.Patch(facecolor="#d90429", label="Sender mục tiêu"),
            mpatches.Patch(facecolor="#7209b7", label="Receiver mục tiêu"),
            mpatches.Patch(facecolor="#f77f00", label="Tài khoản luân chuyển (Degree > 4)"),
            mpatches.Patch(facecolor="#90be6d", label="Node quan trọng (GNNExplainer)"),
            mpatches.Patch(facecolor="#8ecae6", label="Tài khoản lân cận"),
            mpatches.Patch(facecolor="#ef476f", label="Cạnh fraud (y=1)"),
            mpatches.Patch(facecolor="#8ecae6", label="Cạnh hợp lệ (y=0)"),
            mpatches.Patch(facecolor="#2a9d8f", label=f"Hợp lệ importance cao (edge_mask >= {mask_threshold})"),
            mpatches.Patch(facecolor="#d90429", label=f"Fraud importance cao (edge_mask >= {mask_threshold})"),
            mpatches.Patch(facecolor="#adb5bd", label="Màu fallback khi thiếu y"),
            mpatches.Patch(facecolor="none", edgecolor="#333333", label="Nét đứt: target edge"),
            mlines.Line2D([], [], color="#333333", linewidth=4, label="Nét càng dày = importance cao"),
        ]
        ax.legend(handles=legends, loc="lower right", fontsize=8, framealpha=0.9)

        meta = getattr(sub_data, "metadata", {})
        src_acc = meta.get("src_account", str(target_u))
        dst_acc = meta.get("dst_account", str(target_v))
        ax.set_title(
            f"Đồ thị con giải thích giao dịch nghi vấn (NetworkX XAI)\n"
            f"Giao dịch: {src_acc} -> {dst_acc} | Nodes: {num_nodes} | "
            f"Edges: {num_edges} | Fraud edges: "
            f"{int(np.sum(edge_labels == 1)) if edge_labels is not None else 'N/A'}",
            fontsize=12, fontweight="bold", color="#1d3557", pad=12,
        )
        ax.axis("off")

        if output_path is None:
            out_dir = PROC_DIR / "figures_xai"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"subgraph_xai_{target_u}_{target_v}.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        return output_path


def visualize_subgraph(
    sub_data: Data,
    edge_mask: torch.Tensor | np.ndarray | None = None,
    output_path: Path | str | None = None,
    mask_threshold: float = 0.5,
    node_importance: torch.Tensor | np.ndarray | None = None,
) -> Path:
    """Hàm tiện ích gọi nhanh cho các thành viên trong nhóm."""
    viz = SubgraphXaiVisualizer()
    return viz.plot_subgraph_explanation(
        sub_data=sub_data,
        edge_mask=edge_mask,
        node_importance=node_importance,
        output_path=output_path,
        mask_threshold=mask_threshold,
    )


if __name__ == "__main__":
    print("=" * 65)
    print("Thành viên 2 (Graph Engineer): Kiểm thử module trực quan hóa")
    print("=" * 65)

    subgraph_files = list((PROC_DIR / "subgraphs").glob("subgraph_edge_*.pt"))
    if not subgraph_files:
        print("[!] Không tìm thấy file đồ thị con trong data/processed/subgraphs.")
        sys.exit(0)

    sample_sub = torch.load(subgraph_files[0], weights_only=False)
    viz = SubgraphXaiVisualizer()

    # Kiểm tra mapping ID
    mapping = viz.map_tensor_indices_to_accounts(sample_sub)
    print(f"[*] Đã ánh xạ thành công {len(mapping)} tài khoản trong subgraph.")
    for local_id in list(mapping.keys())[:3]:
        print(f"    Node {local_id} -> Account ID: {mapping[local_id]}")

    # Vẽ thử đồ thị mẫu
    out_file = PROC_DIR / "figures_xai" / "test_tv2_preview.png"
    saved_path = viz.plot_subgraph_explanation(
        sub_data=sample_sub,
        edge_mask=None,  # Tự động dùng điểm giả lập trước khi có TV3
        output_path=out_file,
    )
    print(f"[OK] Đã vẽ và lưu ảnh kiểm thử tại: {saved_path}")
