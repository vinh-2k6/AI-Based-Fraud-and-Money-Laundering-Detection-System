import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.gnn_explainer import FraudGNNExplainer
from src.models.gnn_model import FraudDetectionGNN


class TestFraudGNNExplainer(unittest.TestCase):
    def make_subgraph(self) -> Data:
        data = Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
            y=torch.tensor([1, 0, 0, 0]),
        )
        data.target_edge_label_index = data.edge_index[:, :1]
        data.target_edge_idx_in_sub = 0
        data.metadata = {"ground_truth_label": 1}
        return data

    def test_explain_and_round_trip(self) -> None:
        torch.manual_seed(42)
        data = self.make_subgraph()
        explainer = FraudGNNExplainer(
            FraudDetectionGNN(in_channels=3, hidden_channels=4),
            epochs=2,
        )

        result = explainer.explain_subgraph(data)

        self.assertEqual(result["edge_mask"].numel(), data.edge_index.shape[1])
        self.assertEqual(result["node_mask"].shape[0], data.num_nodes)
        self.assertEqual(result["node_importance"].numel(), data.num_nodes)
        probability = float(result["prediction_probability"].reshape(-1)[0])
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = explainer.save_explanation(
                result, Path(temporary_directory) / "explanation.pt"
            )
            loaded = explainer.load_explanation(path)
            self.assertEqual(
                loaded["edge_mask"].shape,
                result["edge_mask"].shape,
            )
            self.assertEqual(
                loaded["node_importance"].shape,
                result["node_importance"].shape,
            )

    def test_empty_graph_is_rejected(self) -> None:
        data = Data(x=torch.randn(2, 3), edge_index=torch.empty((2, 0), dtype=torch.long))
        data.target_edge_label_index = torch.empty((2, 0), dtype=torch.long)
        explainer = FraudGNNExplainer(
            FraudDetectionGNN(in_channels=3, hidden_channels=4),
            epochs=1,
        )
        with self.assertRaises(ValueError):
            explainer.explain_subgraph(data)


if __name__ == "__main__":
    unittest.main()
