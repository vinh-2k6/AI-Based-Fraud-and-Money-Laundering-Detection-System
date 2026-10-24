"""GNNExplainer integration for FraudDetectionGNN edge predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from src.models.gnn_model import FraudDetectionGNN


class FraudGNNExplainer:
    """Generate and persist explanations for extracted fraud subgraphs."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device = "cpu",
        epochs: int = 200,
        lr: float = 0.01,
        node_mask_type: str = "attributes",
        edge_mask_type: str = "object",
    ) -> None:
        if epochs <= 0:
            raise ValueError("epochs phải lớn hơn 0")
        if lr <= 0:
            raise ValueError("lr phải lớn hơn 0")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=epochs, lr=lr),
            explanation_type="model",
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
            model_config={
                "mode": "binary_classification",
                "task_level": "edge",
                "return_type": "raw",
            },
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        in_channels: int,
        device: str | torch.device = "cpu",
        epochs: int = 200,
        lr: float = 0.01,
    ) -> "FraudGNNExplainer":
        """Create an explainer from a GraphSAGE checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Định dạng checkpoint không hợp lệ: {path}")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint không chứa state dict: {path}")
        checkpoint_channels = int(checkpoint.get("in_channels", in_channels))
        hidden_channels = int(checkpoint.get("hidden_channels", 64))
        if checkpoint_channels != in_channels:
            raise ValueError(
                "Checkpoint và graph không cùng số chiều node feature: "
                f"{checkpoint_channels} != {in_channels}"
            )
        model = FraudDetectionGNN(
            in_channels=checkpoint_channels,
            hidden_channels=hidden_channels,
        )
        model.load_state_dict(state_dict)
        return cls(model, device=device, epochs=epochs, lr=lr)

    @staticmethod
    def _validate_subgraph(sub_data: Data) -> None:
        for field in ("x", "edge_index", "target_edge_label_index"):
            if not hasattr(sub_data, field) or getattr(sub_data, field) is None:
                raise ValueError(f"Subgraph thiếu trường bắt buộc: {field}")
        if sub_data.x.ndim != 2:
            raise ValueError("sub_data.x phải là tensor 2 chiều")
        if sub_data.edge_index.ndim != 2 or sub_data.edge_index.shape[0] != 2:
            raise ValueError("sub_data.edge_index phải có shape [2, num_edges]")
        if sub_data.edge_index.shape[1] == 0:
            raise ValueError("Không thể giải thích subgraph không có cạnh")
        target = sub_data.target_edge_label_index
        if target.ndim != 2 or target.shape[0] != 2 or target.shape[1] == 0:
            raise ValueError(
                "target_edge_label_index phải có shape [2, num_target_edges]"
            )
        if target.min().item() < 0 or target.max().item() >= sub_data.num_nodes:
            raise ValueError("target_edge_label_index chứa node index ngoài phạm vi")

    def explain_subgraph(self, sub_data: Data) -> dict[str, Any]:
        """Generate masks and prediction metadata for one PyG subgraph."""
        self._validate_subgraph(sub_data)
        data = sub_data.to(self.device)
        target_edge_idx = getattr(data, "target_edge_idx_in_sub", None)

        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_label_index=data.target_edge_label_index,
        )
        edge_mask = explanation.edge_mask
        node_mask = explanation.node_mask
        if edge_mask is None or node_mask is None:
            raise RuntimeError("GNNExplainer không trả về edge_mask/node_mask")
        if edge_mask.numel() != data.edge_index.shape[1]:
            raise RuntimeError("edge_mask không khớp số lượng cạnh")

        if node_mask.ndim == 1:
            node_importance = node_mask.abs()
        else:
            node_importance = node_mask.abs().mean(dim=-1)
        if node_importance.numel() != data.num_nodes:
            raise RuntimeError("node_importance không khớp số lượng node")

        with torch.no_grad():
            logit = self.model(
                data.x,
                data.edge_index,
                data.target_edge_label_index,
            ).reshape(-1)
            probability = torch.sigmoid(logit)

        return {
            "edge_mask": edge_mask.detach().cpu(),
            "node_mask": node_mask.detach().cpu(),
            "node_importance": node_importance.detach().cpu(),
            "prediction_logit": logit.detach().cpu(),
            "prediction_probability": probability.detach().cpu(),
            "target_edge_label_index": data.target_edge_label_index.detach().cpu(),
            "target_edge_idx_in_sub": target_edge_idx,
            "metadata": getattr(sub_data, "metadata", {}),
        }

    @staticmethod
    def save_explanation(
        explanation: dict[str, Any],
        output_path: Path | str,
        source_subgraph: Path | str | None = None,
    ) -> Path:
        """Save an explanation dictionary as a CPU-loadable PyTorch artifact."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(explanation)
        if source_subgraph is not None:
            payload["source_subgraph"] = str(source_subgraph)
        torch.save(payload, path)
        return path

    @staticmethod
    def load_explanation(path: Path | str) -> dict[str, Any]:
        """Load a previously saved explanation."""
        explanation_path = Path(path)
        if not explanation_path.exists():
            raise FileNotFoundError(f"Không tìm thấy explanation: {explanation_path}")
        payload = torch.load(
            explanation_path, map_location="cpu", weights_only=False
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Explanation không có dạng dictionary: {explanation_path}")
        return payload

    def explain_file(
        self,
        subgraph_path: Path | str,
        output_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Explain one saved subgraph and optionally persist its result."""
        source_path = Path(subgraph_path)
        sub_data = torch.load(source_path, map_location=self.device, weights_only=False)
        result = self.explain_subgraph(sub_data)
        if output_path is not None:
            self.save_explanation(result, output_path, source_subgraph=source_path)
        return result

    def explain_directory(
        self,
        input_dir: Path | str,
        output_dir: Path | str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Explain all subgraph files, retaining per-file errors in the results."""
        source_dir = Path(input_dir)
        files = sorted(source_dir.glob("subgraph_edge_*.pt"))
        if limit is not None:
            files = files[:limit]
        results: list[dict[str, Any]] = []
        for source_path in files:
            output_path = (
                Path(output_dir) / f"explanation_{source_path.stem}.pt"
                if output_dir is not None
                else None
            )
            try:
                result = self.explain_file(source_path, output_path)
                results.append(
                    {
                        "source_subgraph": str(source_path),
                        "output_explanation": str(output_path) if output_path else None,
                        "status": "success",
                        "result": result,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "source_subgraph": str(source_path),
                        "output_explanation": str(output_path) if output_path else None,
                        "status": "error",
                        "error_message": str(exc),
                    }
                )
        return results
