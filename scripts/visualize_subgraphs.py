"""
scripts/visualize_subgraphs.py
===============================
Script thực thi trực quan hóa hàng loạt đồ thị con XAI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.graph_visualizer import SubgraphXaiVisualizer


def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa đồ thị con giải thích gian lận (NetworkX XAI)")
    parser.add_argument(
        "--subgraphs-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "subgraphs",
        help="Thư mục chứa các file subgraph .pt",
    )
    parser.add_argument(
        "--explanations-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "xai_explanations",
        help="Thư mục chứa các file explanation .pt của GNNExplainer (tùy chọn)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "figures_xai",
        help="Thư mục xuất ảnh đồ thị PNG",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số lượng đồ thị con cần vẽ",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng điểm edge_mask để tô nổi bật nét đậm",
    )
    args = parser.parse_args()

    subgraph_files = sorted(list(args.subgraphs_dir.glob("subgraph_edge_*.pt")))
    if not subgraph_files:
        print(f"[!] Không tìm thấy file subgraph trong: {args.subgraphs_dir}")
        return

    if args.limit is not None:
        subgraph_files = subgraph_files[: args.limit]

    print("=" * 60)
    print("TRỰC QUAN HÓA ĐỒ THỊ CON GIẢI THÍCH (NETWORKX XAI) - TV2")
    print("=" * 60)
    print(f"[*] Tìm thấy {len(subgraph_files)} đồ thị con.")
    print(f"[*] Thư mục subgraphs    : {args.subgraphs_dir}")
    print(f"[*] Thư mục explanations : {args.explanations_dir}")
    print(f"[*] Thư mục lưu ảnh      : {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    viz = SubgraphXaiVisualizer()

    count = 0
    for idx, f_path in enumerate(subgraph_files, 1):
        try:
            sub = torch.load(f_path, weights_only=False)
            out_img = args.output_dir / f"{f_path.stem}_xai.png"

            # 1. Tìm file explanation tương ứng do GNNExplainer sinh ra
            edge_mask = None
            node_importance = None
            exp_file = args.explanations_dir / f"explanation_{f_path.stem}.pt"

            if exp_file.exists():
                exp_data = torch.load(exp_file, map_location="cpu", weights_only=False)
                edge_mask = exp_data.get("edge_mask")
                node_importance = exp_data.get("node_importance")
                prob_raw = exp_data.get("prediction_probability", 0.0)
                prob_val = float(prob_raw.reshape(-1)[0]) if isinstance(prob_raw, torch.Tensor) else float(prob_raw)
                exp_info = f"(GNNExplainer: P={prob_val:.4f})"
            else:
                edge_mask = getattr(sub, "edge_mask", None)
                node_importance = getattr(sub, "node_importance", None)
                exp_info = "(Fallback mask)"

            viz.plot_subgraph_explanation(
                sub_data=sub,
                edge_mask=edge_mask,
                node_importance=node_importance,
                output_path=out_img,
                mask_threshold=args.threshold,
            )
            print(f"  [{idx}/{len(subgraph_files)}] Đã vẽ: {f_path.name} -> {out_img.name} {exp_info}")
            count += 1
        except Exception as e:
            print(f"  [X] Lỗi vẽ {f_path.name}: {e}")

    print("=" * 60)
    print(f"[OK] Đã hoàn thành xuất {count}/{len(subgraph_files)} ảnh tại: {args.output_dir}")


if __name__ == "__main__":
    main()
