import unittest

import torch
from torch_geometric.data import Data

from src.reason_codes import ReasonCodeGenerator


class TestReasonCodeGenerator(unittest.TestCase):
    def test_structural_chain_code_contains_mask_evidence(self) -> None:
        graph = Data(
            x=torch.ones(3, 2),
            edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        )
        graph.target_edge_label_index = graph.edge_index[:, :1]
        explanation = {
            "edge_mask": torch.tensor([1.0, 0.8, 0.1]),
            "node_importance": torch.tensor([0.2, 1.0, 0.7]),
            "prediction_probability": torch.tensor([0.9]),
        }

        result = ReasonCodeGenerator().generate(explanation, graph)

        self.assertEqual(result["explanation_status"], "OK")
        self.assertEqual(result["reason_codes"][0]["code"], "RC-STRUCT-CHAIN")
        self.assertEqual(result["reason_codes"][0]["evidence_edges"], [0, 1])

    def test_low_signal_is_explicit(self) -> None:
        result = ReasonCodeGenerator().generate({
            "edge_mask": torch.zeros(2),
            "node_importance": torch.zeros(2),
        })
        self.assertEqual(result["reason_codes"][0]["code"], "RC-XAI-LOW-SIGNAL")
        self.assertEqual(result["explanation_status"], "XAI_LOW_SIGNAL")


if __name__ == "__main__":
    unittest.main()
