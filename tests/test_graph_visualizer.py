import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.graph_visualizer import SubgraphXaiVisualizer


class TestGraphVisualizer(unittest.TestCase):
    def make_graph(self, labels: torch.Tensor) -> Data:
        data = Data(
            x=torch.ones(4, 2),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
            y=labels,
        )
        data.target_edge_label_index = data.edge_index[:, :1]
        data.target_edge_idx_in_sub = 0
        data.orig_account_ids = [10, 20, 30, 40]
        data.metadata = {"src_account": 10, "dst_account": 20, "num_hops": 2}
        return data

    def test_labels_are_independent_from_edge_importance(self) -> None:
        data = self.make_graph(torch.tensor([1, 0, 1, 0]))
        visualizer = SubgraphXaiVisualizer(mapping_path=None)
        self.assertEqual(
            visualizer._get_edge_labels(data, 4).tolist(),
            [1, 0, 1, 0],
        )
        importance = visualizer._normalize_edge_importance(
            torch.tensor([-2.0, float("nan"), 0.5, 1.0]), 4
        )
        self.assertGreater(importance[0], importance[2])
        self.assertEqual(importance[1], 0.0)

    def test_shape_variants_and_mismatches_fallback(self) -> None:
        visualizer = SubgraphXaiVisualizer(mapping_path=None)
        data = self.make_graph(torch.tensor([[1], [0], [1], [0]]))
        self.assertEqual(visualizer._get_edge_labels(data, 4).tolist(), [1, 0, 1, 0])
        self.assertIsNone(visualizer._get_edge_labels(data, 3))
        self.assertEqual(
            visualizer._normalize_edge_importance(torch.ones(2), 4).tolist(),
            [0.0, 0.0, 0.0, 0.0],
        )

    def test_png_is_created_with_label_colors_and_target(self) -> None:
        data = self.make_graph(torch.tensor([1, 0, 1, 0]))
        visualizer = SubgraphXaiVisualizer(mapping_path=None)
        with tempfile.TemporaryDirectory() as directory:
            output = visualizer.plot_subgraph_explanation(
                data,
                edge_mask=torch.tensor([0.1, 0.9, 0.3, 0.8]),
                output_path=Path(directory) / "graph.png",
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_missing_labels_does_not_crash(self) -> None:
        data = self.make_graph(torch.tensor([1, 0, 1, 0]))
        del data.y
        visualizer = SubgraphXaiVisualizer(mapping_path=None)
        with tempfile.TemporaryDirectory() as directory:
            output = visualizer.plot_subgraph_explanation(
                data,
                edge_mask=torch.tensor([float("nan"), -1.0, 0.2, 0.3]),
                output_path=Path(directory) / "fallback.png",
            )
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
