"""
src/subgraph_extractor.py
==========================
Module lọc và trích xuất đồ thị con (Subgraphs) phục vụ Explainable AI (Tuần 4 - Thành viên 1: Data Engineer).

Nhiệm vụ:
  1. Lọc các cạnh giao dịch trong tập Test bị đánh cờ gian lận (isFraud / Is_laundering = 1 hoặc do mô hình dự đoán).
  2. Trích xuất đồ thị con cục bộ (k-hop ego-subgraph) xung quanh các cạnh mục tiêu để tái hiện trọn vẹn ngữ cảnh giao dịch.
  3. Relabel các node trong subgraph, đồng thời lưu giữ bản đồ ánh xạ ngược về chỉ số gốc và ID tài khoản thực tế.
  4. Đóng gói đồ thị con thành cấu trúc PyTorch Geometric Data chuẩn, sẵn sàng cung cấp cho:
     - Thành viên 3 (AI Architect): Đưa vào GNNExplainer để trích xuất edge_mask và node_mask.
     - Thành viên 2 (Graph Engineer): Ánh xạ ngược và trực quan hóa mạng lưới qua NetworkX.
     - Thành viên 4 (MLOps & Reporter): Tích hợp vào luồng suy diễn thời gian thực (Real-time Inference XAI).
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph, to_undirected

# Đảm bảo in tiếng Việt an toàn trên console Windows (tránh lỗi charmap/cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class FraudSubgraphExtractor:
    """
    Bộ trích xuất đồ thị con gian lận từ đồ thị giao dịch PyTorch Geometric.

    Args:
        data_test (Data | Path | str): Đối tượng PyG Data của tập Test hoặc đường dẫn tới file .pt.
        mapping_path (Path | str | None): Đường dẫn tới account_mapping.pkl để giải mã ID tài khoản gốc.
        checkpoint_path (Path | str | None): Đường dẫn tới checkpoint GraphSAGE (nếu muốn lọc theo dự đoán).
        device (str | torch.device): Thiết bị tính toán ('cpu' hoặc 'cuda').
    """

    def __init__(
        self,
        data_test: Data | Path | str,
        mapping_path: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)

        # 1. Nạp đồ thị Test
        if isinstance(data_test, (str, Path)):
            test_path = Path(data_test)
            if not test_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file dữ liệu đồ thị Test tại: {test_path}")
            print(f"[SubgraphExtractor] Đang nạp đồ thị Test từ {test_path}...")
            self.data = torch.load(test_path, map_location="cpu", weights_only=False)
        elif isinstance(data_test, Data):
            self.data = data_test
        else:
            raise TypeError(f"data_test phải là PyG Data hoặc Path/str, nhận được: {type(data_test)}")

        self.num_nodes = self.data.num_nodes
        self.num_edges = self.data.edge_index.size(1)

        # 2. Xây dựng cấu trúc vô hướng phục vụ mở rộng láng giềng k-hop siêu tốc
        print(f"[SubgraphExtractor] Xây dựng đồ thị vô hướng tra cứu k-hop ({self.num_edges:,} cạnh)...")
        t0 = time.perf_counter()
        self.undirected_edge_index = to_undirected(self.data.edge_index)
        t_elapsed = time.perf_counter() - t0
        print(f"[SubgraphExtractor] Hoàn tất tra cứu vô hướng trong {t_elapsed:.3f}s.")

        # 3. Nạp từ điển ánh xạ Account ID
        self.account_to_idx: dict[int, int] = {}
        self.idx_to_account: dict[int, int] = {}
        if mapping_path is not None:
            mapping_p = Path(mapping_path)
            if mapping_p.exists():
                with open(mapping_p, "rb") as f:
                    mapping_raw = pickle.load(f)
                    if isinstance(mapping_raw, dict) and "account_to_idx" in mapping_raw:
                        self.account_to_idx = mapping_raw["account_to_idx"]
                    elif isinstance(mapping_raw, dict):
                        self.account_to_idx = mapping_raw
                self.idx_to_account = {idx: acc for acc, idx in self.account_to_idx.items()}
                print(f"[SubgraphExtractor] Đã nạp từ điển ánh xạ {len(self.idx_to_account):,} tài khoản.")
            else:
                print(f"[Cảnh báo] File mapping không tồn tại: {mapping_p}. Bỏ qua giải mã Account ID.")

        # 4. Nạp checkpoint mô hình (nếu có yêu cầu lọc theo dự đoán mô hình)
        self.model = None
        if checkpoint_path is not None:
            self._load_model_checkpoint(Path(checkpoint_path))

    def _load_model_checkpoint(self, checkpoint_path: Path) -> None:
        """Nạp trọng số mô hình FraudDetectionGNN phục vụ tính xác suất dự đoán."""
        if not checkpoint_path.exists():
            print(f"[Cảnh báo] Không tìm thấy checkpoint tại {checkpoint_path}. Bỏ qua nạp mô hình.")
            return

        try:
            from src.models.gnn_model import FraudDetectionGNN

            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            in_channels = ckpt.get("in_channels", self.data.x.shape[1]) if isinstance(ckpt, dict) else self.data.x.shape[1]
            hidden_channels = ckpt.get("hidden_channels", 64) if isinstance(ckpt, dict) else 64

            model = FraudDetectionGNN(in_channels=in_channels, hidden_channels=hidden_channels)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            print(f"[SubgraphExtractor] Đã nạp thành công mô hình GraphSAGE từ {checkpoint_path}.")
        except Exception as e:
            print(f"[Cảnh báo] Không thể khởi tạo mô hình từ checkpoint: {e}")
            self.model = None

    def find_fraud_edges(
        self,
        mode: Literal["ground_truth", "model_pred", "both"] = "ground_truth",
        threshold: float = 0.5,
        limit: int | None = None,
    ) -> list[int]:
        """
        Tìm và lọc danh sách các cạnh giao dịch bị đánh cờ gian lận trong tập Test.

        Args:
            mode:
                - 'ground_truth': Lọc các cạnh có nhãn Is_laundering == 1 (y == 1).
                - 'model_pred': Lọc các cạnh mà mô hình GraphSAGE dự đoán P(fraud) >= threshold.
                - 'both': Lọc các cạnh thoả mãn cả nhãn thật và dự đoán (True Positives).
            threshold: Ngưỡng xác suất phân loại gian lận (mặc định 0.5).
            limit: Giới hạn số lượng cạnh trả về (None = lấy tất cả).

        Returns:
            list[int]: Danh sách các chỉ số cạnh (edge indices) trong đồ thị Test.
        """
        if self.data.y is None:
            raise ValueError("Đồ thị Test không có thuộc tính nhãn data.y!")

        # 1. Chế độ Ground Truth (mặc định cho Thành viên 1)
        if mode == "ground_truth" or self.model is None:
            if mode != "ground_truth" and self.model is None:
                print("[Thông báo] Mô hình chưa được nạp, tự động chuyển về chế độ 'ground_truth'.")
            fraud_mask = (self.data.y == 1)
            fraud_indices = torch.where(fraud_mask)[0].tolist()
            if limit is not None:
                fraud_indices = fraud_indices[:limit]
            return fraud_indices

        # 2. Chế độ dự đoán bằng mô hình
        print(f"[SubgraphExtractor] Chạy suy diễn bằng mô hình GraphSAGE (threshold={threshold})...")
        probs = self._predict_edge_probabilities(threshold=threshold)
        pred_fraud_mask = (probs >= threshold)

        if mode == "model_pred":
            selected_indices = torch.where(pred_fraud_mask)[0].tolist()
        elif mode == "both":
            combined_mask = pred_fraud_mask & (self.data.y == 1)
            selected_indices = torch.where(combined_mask)[0].tolist()
        else:
            raise ValueError(f"Chế độ lọc không hợp lệ: {mode}")

        if limit is not None:
            selected_indices = selected_indices[:limit]

        return selected_indices

    def _predict_edge_probabilities(self, threshold: float = 0.5, batch_size: int = 4096) -> torch.Tensor:
        """Hàm phụ trợ tính toán xác suất gian lận cho toàn bộ cạnh tập Test."""
        from torch_geometric.loader import LinkNeighborLoader

        loader = LinkNeighborLoader(
            data=self.data,
            num_neighbors=[10, 10],
            edge_label_index=self.data.edge_index,
            edge_label=self.data.y,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        all_probs: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch.x, batch.edge_index, batch.edge_label_index)
                prob = torch.sigmoid(logits).cpu().reshape(-1)
                all_probs.append(prob)
        return torch.cat(all_probs, dim=0)

    def extract_subgraph(
        self,
        target_edge_idx: int,
        num_hops: int = 2,
        directed_expansion: bool = False,
    ) -> Data:
        """
        Trích xuất đồ thị con cục bộ xung quanh một cạnh giao dịch mục tiêu.

        Quy trình kỹ thuật:
          1. Lấy hai nút đầu mút (u: sender, v: receiver) của cạnh mục tiêu.
          2. Tìm tập hợp tất cả các nút lân cận trong bán kính num_hops (mặc định 2-hop).
             Sử dụng cấu trúc vô hướng (directed_expansion=False) để bao quát trọn vẹn
             các chuỗi bóc tách (Layering) và chu trình rửa tiền (Cycles) xung quanh hai đối tượng.
          3. Trích xuất các cạnh có hướng gốc giữa các nút trong đồ thị con và relabel về 0..N_sub-1.
          4. Xác định chính xác vị trí cạnh mục tiêu trong đồ thị con mới (target_edge_label_index).
          5. Bảo toàn các đặc trưng x (node), edge_attr (edge), nhãn y (fraud).
          6. Đóng gói bản đồ ánh xạ ngược và metadata chi tiết.

        Args:
            target_edge_idx (int): Chỉ số cạnh mục tiêu trong đồ thị Test gốc.
            num_hops (int): Bán kính láng giềng k-hop (mặc định 2, ứng với 2 tầng GraphSAGE).
            directed_expansion (bool): Nếu True, chỉ mở rộng theo chiều có hướng của giao dịch.

        Returns:
            Data: Đối tượng đồ thị con PyG hoàn chỉnh.
        """
        if target_edge_idx < 0 or target_edge_idx >= self.num_edges:
            raise IndexError(f"target_edge_idx {target_edge_idx} nằm ngoài khoảng [0, {self.num_edges - 1}]!")

        # 1. Xác định hai đầu mút của cạnh mục tiêu
        u = self.data.edge_index[0, target_edge_idx].item()
        v = self.data.edge_index[1, target_edge_idx].item()
        center_nodes = torch.tensor([u, v], dtype=torch.long)

        # 2. Mở rộng láng giềng k-hop
        expansion_edges = self.data.edge_index if directed_expansion else self.undirected_edge_index
        subset_nodes, _, _, _ = k_hop_subgraph(
            node_idx=center_nodes,
            num_hops=num_hops,
            edge_index=expansion_edges,
            relabel_nodes=False,
        )

        # 3. Trích xuất đồ thị con có hướng gốc và relabeling
        sub_edge_index, sub_edge_attr, edge_mask = subgraph(
            subset=subset_nodes,
            edge_index=self.data.edge_index,
            edge_attr=self.data.edge_attr if hasattr(self.data, "edge_attr") else None,
            relabel_nodes=True,
            return_edge_mask=True,
            num_nodes=self.num_nodes,
        )

        # 4. Trích xuất các đặc trưng và nhãn tương ứng
        sub_x = self.data.x[subset_nodes] if hasattr(self.data, "x") and self.data.x is not None else None
        sub_y = self.data.y[edge_mask] if hasattr(self.data, "y") and self.data.y is not None else None

        # 5. Xác định vị trí của cạnh mục tiêu trong đồ thị con mới
        orig_selected_edge_indices = torch.where(edge_mask)[0]
        match_idx = torch.where(orig_selected_edge_indices == target_edge_idx)[0]

        if len(match_idx) == 0:
            raise RuntimeError(
                f"Lỗi logic: Cạnh mục tiêu {target_edge_idx} (u={u}, v={v}) "
                f"không xuất hiện trong subgraph được trích xuất!"
            )
        target_edge_idx_in_sub = match_idx[0].item()

        # Tạo target_edge_label_index [2, 1] cho forward pass của mô hình
        target_edge_label_index = sub_edge_index[:, target_edge_idx_in_sub : target_edge_idx_in_sub + 1]

        # 6. Ánh xạ ngược về ID tài khoản thực tế
        subset_nodes_list = subset_nodes.tolist()
        orig_account_ids = [self.idx_to_account.get(nid, nid) for nid in subset_nodes_list]
        src_account = self.idx_to_account.get(u, u)
        dst_account = self.idx_to_account.get(v, v)

        # 7. Thống kê đặc trưng đồ thị con
        num_sub_nodes = len(subset_nodes)
        num_sub_edges = sub_edge_index.size(1)
        num_fraud_edges = int(sub_y.sum()) if sub_y is not None else 0
        fraud_ratio = float(num_fraud_edges / max(1, num_sub_edges))

        # 8. Dự đoán xác suất nếu mô hình khả dụng
        predicted_prob = None
        predicted_label = None
        if self.model is not None and sub_x is not None:
            with torch.no_grad():
                sub_x_dev = sub_x.to(self.device)
                sub_ei_dev = sub_edge_index.to(self.device)
                sub_eli_dev = target_edge_label_index.to(self.device)
                logit = self.model(sub_x_dev, sub_ei_dev, sub_eli_dev)
                prob = float(torch.sigmoid(logit).cpu().item())
                predicted_prob = prob
                predicted_label = int(prob >= 0.5)

        # 9. Đóng gói Metadata chi tiết
        metadata: dict[str, Any] = {
            "target_edge_idx_global": int(target_edge_idx),
            "target_edge_idx_sub": int(target_edge_idx_in_sub),
            "src_global_node": int(u),
            "dst_global_node": int(v),
            "src_account": src_account,
            "dst_account": dst_account,
            "ground_truth_label": int(self.data.y[target_edge_idx].item()) if self.data.y is not None else None,
            "predicted_prob": predicted_prob,
            "predicted_label": predicted_label,
            "num_nodes": int(num_sub_nodes),
            "num_edges": int(num_sub_edges),
            "num_fraud_edges": int(num_fraud_edges),
            "fraud_ratio": fraud_ratio,
            "num_hops": int(num_hops),
            "directed_expansion": bool(directed_expansion),
            "is_complex_laundering_ring": bool(num_fraud_edges > 1),
        }

        # 10. Tạo đối tượng Data hoàn chỉnh
        sub_data = Data(
            x=sub_x,
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            y=sub_y,
            num_nodes=num_sub_nodes,
        )

        # Đính kèm các trường nghiệp vụ phục vụ XAI & Visualization
        sub_data.target_edge_label_index = target_edge_label_index
        sub_data.target_edge_idx_in_sub = target_edge_idx_in_sub
        sub_data.orig_target_edge_idx = target_edge_idx
        sub_data.orig_node_indices = subset_nodes
        sub_data.orig_edge_indices = orig_selected_edge_indices
        sub_data.orig_account_ids = orig_account_ids
        sub_data.metadata = metadata

        return sub_data

    def batch_extract_and_save(
        self,
        target_edge_indices: list[int] | torch.Tensor,
        output_dir: Path | str,
        num_hops: int = 2,
        directed_expansion: bool = False,
    ) -> pd.DataFrame:
        """
        Trích xuất hàng loạt đồ thị con và lưu vào thư mục chỉ định kèm file manifest tổng hợp.

        Cấu trúc lưu trữ:
            <output_dir>/
              ├── subgraphs/
              │     ├── subgraph_edge_{idx}.pt
              │     └── ...
              ├── subgraphs_manifest.csv
              └── subgraphs_manifest.json

        Args:
            target_edge_indices: Danh sách chỉ số cạnh mục tiêu cần trích xuất.
            output_dir: Thư mục gốc lưu trữ kết quả.
            num_hops: Bán kính k-hop (mặc định 2).
            directed_expansion: Mở rộng có hướng hay vô hướng.

        Returns:
            pd.DataFrame: Bảng manifest tổng hợp tất cả đồ thị con đã trích xuất.
        """
        out_base = Path(output_dir)
        # Tránh lặp thư mục con nếu output_dir đã có tên là 'subgraphs'
        sub_dir = out_base if out_base.name == "subgraphs" else out_base / "subgraphs"
        sub_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(target_edge_indices, torch.Tensor):
            target_edge_indices = target_edge_indices.tolist()

        n_total = len(target_edge_indices)
        print(f"\n[SubgraphExtractor] Bắt đầu trích xuất hàng loạt {n_total} đồ thị con vào: {out_base}...")
        start_time = time.perf_counter()

        manifest_records: list[dict[str, Any]] = []

        for i, edge_idx in enumerate(target_edge_indices):
            sub_data = self.extract_subgraph(
                target_edge_idx=edge_idx,
                num_hops=num_hops,
                directed_expansion=directed_expansion,
            )

            file_name = f"subgraph_edge_{edge_idx}.pt"
            file_path = sub_dir / file_name

            # Lưu file .pt chuẩn PyG
            torch.save(sub_data, file_path)

            record = {
                "subgraph_id": f"subgraph_{i:04d}",
                "file_name": file_name,
                "file_path": str(file_path),
                **sub_data.metadata,
            }
            manifest_records.append(record)

            if (i + 1) % 50 == 0 or (i + 1) == n_total:
                print(f"  -> Tiến độ: {i + 1}/{n_total} subgraphs ({(i + 1) / n_total * 100:.1f}%)")

        total_elapsed = time.perf_counter() - start_time
        print(f"[SubgraphExtractor] Trích xuất thành công {n_total} subgraphs trong {total_elapsed:.2f}s "
              f"({total_elapsed / max(1, n_total) * 1000:.1f} ms/subgraph)!")

        # Xuất file manifest CSV và JSON
        manifest_df = pd.DataFrame(manifest_records)
        csv_path = out_base / "subgraphs_manifest.csv"
        json_path = out_base / "subgraphs_manifest.json"

        manifest_df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest_records, f, indent=2, ensure_ascii=False)

        print(f"[SubgraphExtractor] Đã lưu Manifest CSV  -> {csv_path}")
        print(f"[SubgraphExtractor] Đã lưu Manifest JSON -> {json_path}")

        return manifest_df


def load_extracted_subgraph(file_path: Path | str) -> Data:
    """Nạp một file đồ thị con .pt đã được lưu bởi FraudSubgraphExtractor."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file đồ thị con: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)
