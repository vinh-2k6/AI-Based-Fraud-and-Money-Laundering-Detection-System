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
        Làm nổi bật các cạnh có trọng số edge_mask >= mask_threshold.
        """
        edge_index = sub_data.edge_index.cpu().numpy()
        num_edges = edge_index.shape[1]
        num_nodes = sub_data.num_nodes

        # Chuẩn bị trọng số edge_mask (nếu chưa có từ TV3, dùng giả lập kiểm thử)
        if edge_mask is None:
            edge_weights = np.ones(num_edges, dtype=np.float32) * 0.2
            target_idx = getattr(sub_data, "target_edge_idx_in_sub", 0)
            if 0 <= target_idx < num_edges:
                edge_weights[target_idx] = 0.95
        else:
            if isinstance(edge_mask, torch.Tensor):
                edge_weights = edge_mask.detach().cpu().numpy().astype(np.float32)
            else:
                edge_weights = np.asarray(edge_mask, dtype=np.float32)
            edge_weights = edge_weights.reshape(-1)
            if edge_weights.size != num_edges:
                raise ValueError(
                    f"edge_mask có {edge_weights.size} phần tử, cần {num_edges}"
                )

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

        # Xây dựng NetworkX DiGraph
        G = nx.DiGraph()
        for i in range(num_nodes):
            G.add_node(i, label=node_labels[i])

        target_sub_idx = getattr(sub_data, "target_edge_idx_in_sub", -1)
        for i in range(num_edges):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            w = float(edge_weights[i]) if i < len(edge_weights) else 0.0
            G.add_edge(u, v, weight=w, is_target=(i == target_sub_idx))

        # Phân loại màu cho nút trung tâm (Sender/Receiver mục tiêu)
        target_u = int(sub_data.target_edge_label_index[0, 0].item()) if hasattr(sub_data, "target_edge_label_index") else 0
        target_v = int(sub_data.target_edge_label_index[1, 0].item()) if hasattr(sub_data, "target_edge_label_index") else 1

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

        # Phân loại cạnh: Nổi bật (High Mask) vs Cạnh nền (Normal)
        high_edges = []
        high_widths = []
        high_colors = []
        bg_edges = []

        for u, v, d in G.edges(data=True):
            w = d.get("weight", 0.0)
            is_target = d.get("is_target", False)
            if is_target or w >= mask_threshold:
                high_edges.append((u, v))
                high_widths.append(2.0 + w * 3.0)
                high_colors.append("#d00000" if is_target else "#e85d04")
            else:
                bg_edges.append((u, v))

        # Tính bố cục hiển thị (Spring Layout)
        pos = nx.spring_layout(G, k=1.8 / np.sqrt(max(1, num_nodes)), iterations=70, seed=42)

        fig, ax = plt.subplots(figsize=figsize, facecolor="#fafafa")
        ax.set_facecolor("#fafafa")

        # Vẽ cạnh nền mờ
        if bg_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=bg_edges, edge_color="#adb5bd",
                width=0.8, alpha=0.35, arrows=True, arrowsize=10, ax=ax,
            )

        # Vẽ cạnh nổi bật có trọng số cao
        if high_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=high_edges, edge_color=high_colors,
                width=high_widths, alpha=0.95, arrows=True, arrowsize=16, ax=ax,
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
            mpatches.Patch(facecolor="#d00000", label=f"Cạnh quan trọng (edge_mask >= {mask_threshold})"),
            mpatches.Patch(facecolor="#adb5bd", label="Giao dịch bình thường"),
        ]
        ax.legend(handles=legends, loc="lower right", fontsize=8, framealpha=0.9)

        meta = getattr(sub_data, "metadata", {})
        src_acc = meta.get("src_account", str(target_u))
        dst_acc = meta.get("dst_account", str(target_v))
        ax.set_title(
            f"Đồ thị con giải thích giao dịch nghi vấn (NetworkX XAI)\n"
            f"Giao dịch: {src_acc} -> {dst_acc} | Nodes: {num_nodes} | Edges: {num_edges}",
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
