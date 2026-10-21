"""Map local GNN explanations to stable, auditable reason codes."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import torch
from torch_geometric.data import Data


class ReasonCodeGenerator:
    """Generate business-facing reason codes from one XAI explanation.

    The generator never treats a mask as a probability.  Masks are normalized
    within the current subgraph and are combined with observable graph or
    transaction evidence before a code is emitted.
    """

    def __init__(self, min_score: float = 0.5, max_codes: int = 3) -> None:
        if not 0.0 < min_score <= 1.0:
            raise ValueError("min_score phải nằm trong khoảng (0, 1]")
        if max_codes <= 0:
            raise ValueError("max_codes phải lớn hơn 0")
        self.min_score = min_score
        self.max_codes = max_codes

    @staticmethod
    def _normalized(values: torch.Tensor) -> torch.Tensor:
        values = values.detach().float().abs().reshape(-1)
        maximum = values.max() if values.numel() else torch.tensor(0.0)
        return values / maximum if float(maximum) > 0 else torch.zeros_like(values)

    @staticmethod
    def _metadata(explanation: Mapping[str, Any], subgraph: Data | None) -> Mapping[str, Any]:
        metadata = explanation.get("metadata", {})
        if isinstance(metadata, Mapping):
            return metadata
        if subgraph is not None and isinstance(getattr(subgraph, "metadata", None), Mapping):
            return subgraph.metadata
        return {}

    @staticmethod
    def _edge_nodes(subgraph: Data | None, index: int) -> tuple[int, int] | None:
        if subgraph is None or subgraph.edge_index.shape[1] <= index:
            return None
        return (
            int(subgraph.edge_index[0, index].item()),
            int(subgraph.edge_index[1, index].item()),
        )

    def generate(
        self,
        explanation: Mapping[str, Any],
        subgraph: Data | None = None,
        transaction_features: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return ranked reason codes and the evidence used to create them."""
        edge_mask = explanation.get("edge_mask")
        node_importance = explanation.get("node_importance")
        if not isinstance(edge_mask, torch.Tensor) or not isinstance(
            node_importance, torch.Tensor
        ):
            raise ValueError("Explanation phải chứa edge_mask và node_importance dạng tensor")

        edge_scores = self._normalized(edge_mask)
        node_scores = self._normalized(node_importance)
        selected_edges = [
            index for index, score in enumerate(edge_scores.tolist())
            if score >= self.min_score
        ]
        selected_nodes = [
            index for index, score in enumerate(node_scores.tolist())
            if score >= self.min_score
        ]
        metadata = self._metadata(explanation, subgraph)
        features = transaction_features or {}
        candidates: list[dict[str, Any]] = []

        edge_pairs = [
            pair for index in selected_edges
            if (pair := self._edge_nodes(subgraph, index)) is not None
        ]
        degrees = Counter(node for pair in edge_pairs for node in pair)
        target_nodes: set[int] = set()
        if subgraph is not None and hasattr(subgraph, "target_edge_label_index"):
            target = subgraph.target_edge_label_index
            target_nodes = {int(target[0, 0]), int(target[1, 0])}

        if len(edge_pairs) >= 2 and any(
            node not in target_nodes for node in degrees
        ):
            score = min(1.0, sum(float(edge_scores[i]) for i in selected_edges[:3]) / 3)
            candidates.append({
                "code": "RC-STRUCT-CHAIN",
                "label": "Chuỗi giao dịch liên kết bất thường",
                "score": round(score, 6),
                "evidence_edges": selected_edges[:3],
                "evidence_nodes": selected_nodes[:3],
                "basis": "Các cạnh liên kết trong subgraph có edge score cao",
            })

        if any(count >= 2 for count in degrees.values()) and len(edge_pairs) >= 3:
            # A repeated endpoint is a conservative local cycle/hub signal.
            score = max((float(edge_scores[i]) for i in selected_edges), default=0.0)
            candidates.append({
                "code": "RC-STRUCT-HUB",
                "label": "Tài khoản trung tâm có nhiều liên kết",
                "score": round(score, 6),
                "evidence_edges": selected_edges[:3],
                "evidence_nodes": [
                    node for node, count in degrees.most_common(3) if count >= 2
                ],
                "basis": "Một node xuất hiện trên nhiều cạnh có importance cao",
            })

        if metadata.get("is_complex_laundering_ring") or (
            metadata.get("num_fraud_edges", 0) > 1 and len(selected_edges) >= 2
        ):
            score = min(1.0, sum(float(edge_scores[i]) for i in selected_edges[:3]) / 3)
            candidates.append({
                "code": "RC-STRUCT-CYCLE",
                "label": "Cụm giao dịch gian lận liên kết/vòng",
                "score": round(score, 6),
                "evidence_edges": selected_edges[:3],
                "evidence_nodes": selected_nodes[:3],
                "basis": "Metadata hoặc nhiều cạnh gian lận cùng nổi bật trong subgraph",
            })

        count_24h = max(
            float(features.get("sender_tx_count_24h", 0) or 0),
            float(features.get("receiver_tx_count_24h", 0) or 0),
        )
        if count_24h > 0 and count_24h >= 10 and selected_nodes:
            candidates.append({
                "code": "RC-TEMPORAL-BURST",
                "label": "Tần suất giao dịch tăng cao trong 24 giờ",
                "score": round(float(node_scores[selected_nodes[0]]), 6),
                "evidence_edges": selected_edges[:3],
                "evidence_nodes": selected_nodes[:3],
                "basis": "Số giao dịch trong cửa sổ 24h vượt ngưỡng 10",
            })

        if features.get("is_new_sender") or features.get("is_new_receiver"):
            candidates.append({
                "code": "RC-ACCOUNT-NEW",
                "label": "Tài khoản mới hoặc chưa có lịch sử",
                "score": round(max(node_scores.tolist(), default=0.0), 6),
                "evidence_edges": selected_edges[:3],
                "evidence_nodes": selected_nodes[:3],
                "basis": "DataProcessor đánh dấu tài khoản cold-start",
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        codes = [item for item in candidates if item["score"] >= self.min_score]
        if not codes:
            codes = [{
                "code": "RC-XAI-LOW-SIGNAL",
                "label": "Không đủ bằng chứng XAI nổi bật",
                "score": round(max(edge_scores.max().item() if edge_scores.numel() else 0.0,
                                   node_scores.max().item() if node_scores.numel() else 0.0), 6),
                "evidence_edges": selected_edges[:3],
                "evidence_nodes": selected_nodes[:3],
                "basis": "Không có bằng chứng đạt ngưỡng chuẩn hóa",
            }]

        probability = explanation.get("prediction_probability")
        probability_value = (
            float(probability.reshape(-1)[0]) if isinstance(probability, torch.Tensor)
            else probability
        )
        return {
            "reason_codes": codes[: self.max_codes],
            "explanation_status": "OK" if codes[0]["code"] != "RC-XAI-LOW-SIGNAL" else "XAI_LOW_SIGNAL",
            "fraud_probability": probability_value,
            "normalization": "abs(mask) / max(abs(mask)) within subgraph",
            "min_score": self.min_score,
        }
