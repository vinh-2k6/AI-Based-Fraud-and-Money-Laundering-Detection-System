"""
tests/test_graph_utils.py
=========================
Unit tests cho module tiện ích đồ thị src/utils/graph_utils.py 
"""

from pathlib import Path
import unittest
import torch

from src.utils.graph_utils import (
    load_account_mapping,
    resolve_node_index,
    resolve_account_id,
    create_cold_start_node_features,
    build_edge_embedding,
    fast_extract_ego_subgraph,
)

BASE_DIR = Path(__file__).resolve().parents[1]
MAPPING_PATH = BASE_DIR / "data" / "processed" / "account_mapping.pkl"


class TestGraphUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MAPPING_PATH.exists():
            raise unittest.SkipTest(f"Không tìm thấy mapping tại {MAPPING_PATH}, bỏ qua test.")
        cls.acc_to_idx, cls.known_train, cls.idx_to_acc = load_account_mapping(MAPPING_PATH)

    def test_load_mapping_caching(self):
        """Kiểm tra tính năng cache in-memory của load_account_mapping."""
        m1, k1, r1 = load_account_mapping(MAPPING_PATH, use_cache=True)
        m2, k2, r2 = load_account_mapping(MAPPING_PATH, use_cache=True)
        self.assertIs(m1, m2, "Mapping phải được trả về từ cache bộ nhớ")
        self.assertGreater(len(m1), 0)
        self.assertGreater(len(r1), 0)

    def test_resolve_node_index(self):
        """Kiểm tra tra cứu node index và nhận diện tài khoản mới (cold-start)."""
        sample_acc = next(iter(self.known_train))
        node_idx, is_cold = resolve_node_index(sample_acc, self.acc_to_idx, self.known_train)
        self.assertGreaterEqual(node_idx, 0)
        self.assertFalse(is_cold, "Tài khoản từ tập train không được coi là cold-start")

        # Tài khoản lạ chưa từng xuất hiện
        fake_acc = 999999999999999
        fake_idx, fake_cold = resolve_node_index(fake_acc, self.acc_to_idx, self.known_train)
        self.assertEqual(fake_idx, -1)
        self.assertTrue(fake_cold)

    def test_resolve_account_id(self):
        """Kiểm tra ánh xạ ngược từ node index về Account ID gốc."""
        first_node_idx = 0
        acc_id = resolve_account_id(first_node_idx, self.idx_to_acc)
        self.assertIsNotNone(acc_id)
        self.assertNotEqual(str(acc_id), "")

    def test_create_cold_start_features(self):
        """Kiểm tra tạo vector đặc trưng cold-start 8 chiều."""
        feat = create_cold_start_node_features(num_nodes=3, feature_dim=8)
        self.assertEqual(feat.shape, (3, 8))
        # 7 chiều đầu phải bằng 0.0
        self.assertTrue(torch.all(feat[:, :7] == 0.0))
        # Chiều cuối is_new_node phải bằng 1.0
        self.assertTrue(torch.all(feat[:, 7] == 1.0))

    def test_build_edge_embedding(self):
        """Kiểm tra ghép nối vector nhúng giao dịch."""
        embeddings = torch.randn(10, 64)
        edge_feat = build_edge_embedding(embeddings, sender_idx=2, receiver_idx=5)
        self.assertEqual(edge_feat.shape, (128,))

        # Kiểm tra kiểm soát lỗi ngoài biên (IndexError)
        with self.assertRaises(IndexError):
            build_edge_embedding(embeddings, sender_idx=2, receiver_idx=20)

    def test_fast_extract_ego_subgraph(self):
        """Kiểm tra trích xuất nhanh ego-subgraph."""
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        nodes, sub_edges, r_map = fast_extract_ego_subgraph(
            sender_idx=0,
            receiver_idx=1,
            edge_index=edge_index,
            num_hops=1,
        )
        self.assertGreater(len(nodes), 0)
        self.assertGreater(sub_edges.size(1), 0)
        self.assertIn(0, r_map)
        self.assertIn(1, r_map)


if __name__ == "__main__":
    unittest.main()
