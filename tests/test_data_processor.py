"""
tests/test_data_processor.py
============================
Bộ kiểm thử tự động (Unit Tests & Integration Tests) cho API nội bộ DataProcessor (Tuần 5 - Thành viên 1).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

# Đảm bảo in tiếng Việt an toàn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch

from src.data_processor import (
    DataProcessor,
    ProcessedTransaction,
    BatchProcessedData,
    TOTAL_EDGE_FEATURE_DIMS,
    TOTAL_NODE_FEATURE_DIMS,
)


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parents[1]
        cls.scaler_path = cls.base_dir / "data" / "processed" / "scalers.pkl"
        cls.encoder_path = cls.base_dir / "data" / "processed" / "edge_encoders.pkl"
        cls.mapping_path = cls.base_dir / "data" / "processed" / "account_mapping.pkl"
        cls.test_csv_path = cls.base_dir / "data" / "processed" / "test_cleaned.csv"
        cls.test_pt_path = cls.base_dir / "data" / "processed" / "data_test.pt"

        if not (cls.scaler_path.exists() and cls.encoder_path.exists() and cls.mapping_path.exists()):
            raise unittest.SkipTest("Thiếu artifacts trong data/processed, bỏ qua kiểm thử.")

        cls.processor = DataProcessor(
            base_dir=cls.base_dir,
            scaler_path=cls.scaler_path,
            encoder_path=cls.encoder_path,
            mapping_path=cls.mapping_path,
            device="cpu",
        )

    def setUp(self):
        # Reset trạng thái sliding window trước mỗi test case
        self.processor.reset_state()

    def test_01_initialization(self):
        """Kiểm tra khởi tạo và tải thành công các artifact tiền xử lý."""
        self.assertGreater(len(self.processor.account_to_idx), 0)
        self.assertEqual(self.processor.total_categorical_dims, 69)
        self.assertIsNotNone(self.processor.scaler)
        self.assertIsNotNone(self.processor.ohe)

    def test_02_process_raw_transaction(self):
        """Kiểm tra xử lý một giao dịch thô (Raw Transaction) đầy đủ các trường."""
        raw_tx = {
            "Sender_account": 9768510209,
            "Receiver_account": 5764492978,
            "Amount": 1500.0,
            "Payment_currency": "UK pounds",
            "Received_currency": "UK pounds",
            "Sender_bank_location": "UK",
            "Receiver_bank_location": "UK",
            "Payment_type": "ACH",
            "Date": "2023-06-19",
            "Time": "22:08:52",
        }

        res = self.processor.process_transaction(raw_tx)
        self.assertIsInstance(res, ProcessedTransaction)
        self.assertEqual(res.edge_index.shape, (2, 1))
        self.assertEqual(res.edge_index.dtype, torch.long)
        self.assertEqual(res.edge_attr.shape, (1, TOTAL_EDGE_FEATURE_DIMS))
        self.assertEqual(res.edge_attr.dtype, torch.float32)
        self.assertGreater(res.processing_time_ms, 0.0)
        self.assertLess(res.processing_time_ms, 50.0)  # Phải hoàn thành cực nhanh

    def test_03_process_precleaned_transaction(self):
        """Kiểm tra xử lý giao dịch ở định dạng đã tiền xử lý sẵn (pre-cleaned format)."""
        if not self.test_csv_path.exists():
            raise unittest.SkipTest("test_cleaned.csv không tồn tại")

        df = pd.read_csv(self.test_csv_path, nrows=1)
        row_dict = df.iloc[0].to_dict()

        res = self.processor.process_transaction(row_dict)
        self.assertIsInstance(res, ProcessedTransaction)
        self.assertEqual(res.edge_attr.shape, (1, 80))
        self.assertEqual(res.sender_account, int(row_dict["Sender_account"]))
        self.assertEqual(res.receiver_account, int(row_dict["Receiver_account"]))

    def test_04_mathematical_consistency_with_saved_graph(self):
        """
        Kiểm tra tính nhất quán toán học tuyệt đối:
        So sánh kết quả edge_attr sinh ra bởi DataProcessor với data_test.pt (cạnh 0).
        """
        if not (self.test_csv_path.exists() and self.test_pt_path.exists()):
            raise unittest.SkipTest("Thiếu test_cleaned.csv hoặc data_test.pt")

        df = pd.read_csv(self.test_csv_path, nrows=1)
        row_dict = df.iloc[0].to_dict()

        test_data = torch.load(self.test_pt_path, map_location="cpu", weights_only=False)
        expected_edge_attr = test_data.edge_attr[0:1]

        res = self.processor.process_transaction(row_dict)

        # Kiểm tra 5 continuous features
        self.assertTrue(
            torch.allclose(res.edge_attr[:, :5], expected_edge_attr[:, :5], atol=1e-4),
            f"Lệch continuous features: {res.edge_attr[:, :5]} vs {expected_edge_attr[:, :5]}",
        )
        # Kiểm tra 6 cyclical features
        self.assertTrue(
            torch.allclose(res.edge_attr[:, 5:11], expected_edge_attr[:, 5:11], atol=1e-4),
            f"Lệch cyclical features: {res.edge_attr[:, 5:11]} vs {expected_edge_attr[:, 5:11]}",
        )
        # Kiểm tra toàn bộ 80 chiều
        self.assertTrue(
            torch.allclose(res.edge_attr, expected_edge_attr, atol=1e-4),
            "Toàn bộ 80 chiều edge_attr phải khớp nhau",
        )

    def test_05_causal_sliding_window_behavior(self):
        """
        Kiểm tra tính đúng đắn của cửa sổ trượt quá khứ 24h:
        - Giao dịch 1 tại t=0: sender_tx_count = 0, sender_amount_sum = 0
        - Giao dịch 2 tại t=3600 (sau 1h): sender_tx_count = 1, sender_amount_sum = amt_tx1, time_since_last_sender_tx = 3600
        - Giao dịch 3 tại t=90000 (sau 25h > 24h): tx1 bị loại bỏ, chỉ còn tx2
        """
        sender = 88888888
        receiver = 99999999
        t0 = 1700000000

        # Giao dịch 1
        s_cnt1, r_cnt1, s_sum1, dt1 = self.processor.update_sliding_window(
            sender=sender, receiver=receiver, amount=100.0, timestamp_epoch=t0
        )
        self.assertEqual(s_cnt1, 0)
        self.assertEqual(s_sum1, 0.0)
        self.assertEqual(dt1, float(self.processor.window_seconds))

        # Giao dịch 2 (sau 1h = 3600s)
        s_cnt2, r_cnt2, s_sum2, dt2 = self.processor.update_sliding_window(
            sender=sender, receiver=receiver, amount=200.0, timestamp_epoch=t0 + 3600
        )
        self.assertEqual(s_cnt2, 1)
        self.assertAlmostEqual(s_sum2, 100.0, places=2)
        self.assertEqual(r_cnt2, 1)
        self.assertEqual(dt2, 3600.0)

        # Giao dịch 3 (sau 25h = 90000s kể từ t0) -> tx1 quá 24h bị loại, chỉ còn tx2
        s_cnt3, r_cnt3, s_sum3, dt3 = self.processor.update_sliding_window(
            sender=sender, receiver=receiver, amount=50.0, timestamp_epoch=t0 + 90000
        )
        self.assertEqual(s_cnt3, 1)  # Chỉ còn tx2
        self.assertAlmostEqual(s_sum3, 200.0, places=2)
        self.assertEqual(dt3, 90000.0 - 3600.0)

    def test_06_inductive_new_node_cold_start(self):
        """Kiểm tra cơ chế đồ thị quy nạp (Inductive Graph) khi gặp tài khoản chưa từng xuất hiện."""
        unseen_sender = 9999999999991
        unseen_receiver = 9999999999992

        self.assertNotIn(unseen_sender, self.processor.account_to_idx)
        self.assertNotIn(unseen_receiver, self.processor.account_to_idx)

        raw_tx = {
            "Sender_account": unseen_sender,
            "Receiver_account": unseen_receiver,
            "Amount": 500.0,
            "Payment_currency": "Euro",
            "Received_currency": "Euro",
            "Sender_bank_location": "France",
            "Receiver_bank_location": "France",
            "Payment_type": "Credit Card",
            "Date": "2023-01-01",
            "Time": "12:00:00",
        }

        res = self.processor.process_transaction(raw_tx)
        self.assertTrue(res.is_new_sender)
        self.assertTrue(res.is_new_receiver)
        self.assertIn(unseen_sender, self.processor.account_to_idx)
        self.assertIn(unseen_receiver, self.processor.account_to_idx)

        # Kiểm tra node feature mặc định của node mới
        new_node_feat = self.processor.get_node_features(res.sender_idx)
        self.assertEqual(new_node_feat.shape, (TOTAL_NODE_FEATURE_DIMS,))
        self.assertEqual(float(new_node_feat[7].item()), 1.0)  # is_new_node cờ = 1.0

    def test_07_batch_processing(self):
        """Kiểm tra xử lý micro-batch nhiều giao dịch cùng lúc."""
        transactions = [
            {
                "Sender_account": 9768510209,
                "Receiver_account": 5764492978,
                "Amount": 100.0,
                "Payment_currency": "UK pounds",
                "Received_currency": "UK pounds",
                "Sender_bank_location": "UK",
                "Receiver_bank_location": "UK",
                "Payment_type": "ACH",
                "Date": "2023-06-19",
                "Time": "22:08:52",
            },
            {
                "Sender_account": 5764492978,
                "Receiver_account": 9768510209,
                "Amount": 500.0,
                "Payment_currency": "UK pounds",
                "Received_currency": "UK pounds",
                "Sender_bank_location": "UK",
                "Receiver_bank_location": "UK",
                "Payment_type": "ACH",
                "Date": "2023-06-19",
                "Time": "22:15:00",
            },
        ]

        batch_res = self.processor.process_batch(transactions)
        self.assertIsInstance(batch_res, BatchProcessedData)
        self.assertEqual(batch_res.batch_size, 2)
        self.assertEqual(batch_res.edge_index.shape, (2, 2))
        self.assertEqual(batch_res.edge_attr.shape, (2, TOTAL_EDGE_FEATURE_DIMS))

        pyg_data = batch_res.to_pyg_data()
        self.assertEqual(pyg_data.edge_index.shape, (2, 2))
        self.assertEqual(pyg_data.edge_attr.shape, (2, 80))

    def test_08_process_json(self):
        """Kiểm tra xử lý chuỗi JSON đầu vào từ luồng streaming."""
        tx_dict = {
            "Sender_account": 9768510209,
            "Receiver_account": 5764492978,
            "Amount": 2500.0,
            "Payment_currency": "UK pounds",
            "Received_currency": "UK pounds",
            "Sender_bank_location": "UK",
            "Receiver_bank_location": "UK",
            "Payment_type": "ACH",
            "Date": "2023-06-19",
            "Time": "22:08:52",
        }
        tx_json = json.dumps(tx_dict)

        res = self.processor.process_json(tx_json)
        self.assertEqual(res.sender_account, 9768510209)
        self.assertEqual(res.receiver_account, 5764492978)
        self.assertEqual(res.edge_attr.shape, (1, 80))


if __name__ == "__main__":
    unittest.main()
