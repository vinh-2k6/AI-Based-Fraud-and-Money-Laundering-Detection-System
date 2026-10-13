"""
tests/test_subgraph_extractor.py
=================================
Bộ kiểm thử tự động (Unit Tests) cho module Subgraph Extractor (Tuần 4 - Thành viên 1).
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Đảm bảo in tiếng Việt an toàn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch_geometric.data import Data

from src.models.gnn_model import FraudDetectionGNN
from src.subgraph_extractor import FraudSubgraphExtractor, load_extracted_subgraph


class TestFraudSubgraphExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parents[1]
        cls.test_graph_path = cls.base_dir / "data" / "processed" / "data_test.pt"
        cls.mapping_path = cls.base_dir / "data" / "processed" / "account_mapping.pkl"

        if not cls.test_graph_path.exists():
            raise unittest.SkipTest(f"Chưa có {cls.test_graph_path}, bỏ qua test.")

        cls.extractor = FraudSubgraphExtractor(
            data_test=cls.test_graph_path,
            mapping_path=cls.mapping_path,
            device="cpu",
        )

    def test_find_fraud_edges(self):
        """Kiểm tra tìm danh sách cạnh gian lận trong tập Test."""
        fraud_edges = self.extractor.find_fraud_edges(mode="ground_truth", limit=20)
        self.assertGreater(len(fraud_edges), 0)
        self.assertLessEqual(len(fraud_edges), 20)

        # Kiểm tra nhãn thực tế của cạnh đầu tiên phải là 1
        first_edge_idx = fraud_edges[0]
        self.assertEqual(int(self.extractor.data.y[first_edge_idx].item()), 1)

    def test_extract_single_subgraph(self):
        """Kiểm tra trích xuất một đồ thị con cục bộ 2-hop xung quanh cạnh gian lận."""
        fraud_edges = self.extractor.find_fraud_edges(mode="ground_truth", limit=1)
        target_edge_idx = fraud_edges[0]

        sub_data = self.extractor.extract_subgraph(
            target_edge_idx=target_edge_idx,
            num_hops=2,
            directed_expansion=False,
        )

        self.assertIsInstance(sub_data, Data)
        self.assertGreater(sub_data.num_nodes, 0)
        self.assertGreater(sub_data.edge_index.size(1), 0)

        # Kiểm tra tính hợp lệ của node relabeling
        max_node_idx = sub_data.edge_index.max().item()
        self.assertLess(max_node_idx, sub_data.num_nodes)

        # Kiểm tra target_edge_label_index
        self.assertEqual(sub_data.target_edge_label_index.shape, (2, 1))

        # Kiểm tra metadata và các trường ánh xạ
        self.assertIn("src_account", sub_data.metadata)
        self.assertIn("dst_account", sub_data.metadata)
        self.assertEqual(sub_data.metadata["ground_truth_label"], 1)
        self.assertEqual(len(sub_data.orig_account_ids), sub_data.num_nodes)
        self.assertEqual(len(sub_data.orig_node_indices), sub_data.num_nodes)

    def test_gnn_model_forward_compatibility(self):
        """Kiểm tra tính tương thích: Đồ thị con trích xuất phải chạy forward pass được trên FraudDetectionGNN."""
        fraud_edges = self.extractor.find_fraud_edges(mode="ground_truth", limit=1)
        target_edge_idx = fraud_edges[0]

        sub_data = self.extractor.extract_subgraph(target_edge_idx=target_edge_idx, num_hops=2)

        # Khởi tạo mô hình GraphSAGE
        in_channels = sub_data.x.shape[1]
        model = FraudDetectionGNN(in_channels=in_channels, hidden_channels=32)
        model.eval()

        with torch.no_grad():
            logit = model(sub_data.x, sub_data.edge_index, sub_data.target_edge_label_index)
            prob = torch.sigmoid(logit).item()

        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_batch_extract_and_save(self):
        """Kiểm tra xuất hàng loạt và đọc lại file đồ thị con .pt cùng manifest."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            fraud_edges = self.extractor.find_fraud_edges(mode="ground_truth", limit=5)
            manifest_df = self.extractor.batch_extract_and_save(
                target_edge_indices=fraud_edges,
                output_dir=temp_dir,
                num_hops=2,
            )

            self.assertEqual(len(manifest_df), 5)
            csv_path = temp_dir / "subgraphs_manifest.csv"
            json_path = temp_dir / "subgraphs_manifest.json"
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            # Thử đọc lại file đầu tiên bằng helper
            first_pt_path = Path(manifest_df.iloc[0]["file_path"])
            loaded_data = load_extracted_subgraph(first_pt_path)
            self.assertEqual(loaded_data.num_nodes, manifest_df.iloc[0]["num_nodes"])
            self.assertEqual(loaded_data.edge_index.size(1), manifest_df.iloc[0]["num_edges"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
