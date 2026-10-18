"""
tests/test_model_freezing.py
============================
Unit tests cho quy trình đóng băng mô hình GraphSAGE 
"""

from pathlib import Path
import tempfile
import unittest
import torch

from src.models.gnn_model import FraudDetectionGNN
from scripts.freeze_model import freeze_and_package_model

BASE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = BASE_DIR / "data" / "processed" / "graphsage_model.pt"
FROZEN_PATH = BASE_DIR / "data" / "processed" / "graphsage_frozen.pt"


class TestModelFreezing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT_PATH.exists():
            raise unittest.SkipTest(f"Không tìm thấy checkpoint tại {CHECKPOINT_PATH}, bỏ qua test.")

    def test_freeze_parameters(self):
        """Kiểm tra toàn bộ tham số của mô hình được đóng băng (requires_grad = False)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_out = Path(tmp_dir) / "test_frozen.pt"
            frozen_model, saved_path = freeze_and_package_model(
                checkpoint_path=CHECKPOINT_PATH,
                output_path=tmp_out,
                device="cpu",
                benchmark_runs=10,
            )

            # Kiểm tra trạng thái eval
            self.assertFalse(frozen_model.training, "Mô hình phải ở chế độ eval()")

            # Kiểm tra requires_grad
            for name, param in frozen_model.named_parameters():
                self.assertFalse(param.requires_grad, f"Tham số {name} chưa được đóng băng!")

            # Kiểm tra file đã lưu
            self.assertTrue(saved_path.exists())
            payload = torch.load(saved_path, map_location="cpu", weights_only=False)
            self.assertTrue(payload.get("is_frozen", False))
            self.assertEqual(payload.get("trainable_parameters"), 0)

    def test_numerical_equivalence(self):
        """Kiểm tra kết quả dự đoán của mô hình đóng băng giống 100% mô hình gốc."""
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        in_c = int(checkpoint.get("in_channels", 8))
        hidden_c = int(checkpoint.get("hidden_channels", 64))
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        model_orig = FraudDetectionGNN(in_channels=in_c, hidden_channels=hidden_c)
        model_orig.load_state_dict(state_dict)
        model_orig.eval()

        if FROZEN_PATH.exists():
            frozen_ckpt = torch.load(FROZEN_PATH, map_location="cpu", weights_only=False)
            model_frozen = FraudDetectionGNN(in_channels=in_c, hidden_channels=hidden_c)
            model_frozen.load_state_dict(frozen_ckpt["model_state_dict"])
            model_frozen.eval()

            torch.manual_seed(123)
            dummy_x = torch.randn(15, in_c)
            dummy_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
            dummy_target = torch.tensor([[0], [1]], dtype=torch.long)

            with torch.no_grad():
                out1 = model_orig(dummy_x, dummy_edge_index, dummy_target)
                out2 = model_frozen(dummy_x, dummy_edge_index, dummy_target)
                diff = float((out1 - out2).abs().max())

            self.assertLess(diff, 1e-6, "Sai số giữa mô hình gốc và mô hình freeze phải < 1e-6")


if __name__ == "__main__":
    unittest.main()
