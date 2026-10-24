"""CLI for generating GNNExplainer artifacts for fraud subgraphs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.gnn_explainer import FraudGNNExplainer
from src.reason_codes import ReasonCodeGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sinh explanation bằng GNNExplainer")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/processed/graphsage_model.pt"))
    parser.add_argument("--subgraphs-dir", type=Path, default=Path("data/processed/subgraphs"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/xai_explanations"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--reason-codes-dir",
        type=Path,
        default=Path("data/processed/reason_codes"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã chọn CUDA nhưng CUDA không khả dụng")
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    files = sorted(args.subgraphs_dir.glob("subgraph_edge_*.pt"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"Không tìm thấy subgraph trong {args.subgraphs_dir}")
    sample = torch.load(files[0], map_location="cpu", weights_only=False)
    explainer = FraudGNNExplainer.from_checkpoint(
        args.checkpoint,
        in_channels=int(sample.x.shape[1]),
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reason_codes_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    reason_code_generator = ReasonCodeGenerator()
    success = 0
    for index, source_path in enumerate(files, 1):
        output_path = args.output_dir / f"explanation_{source_path.stem}.pt"
        row: dict[str, object] = {
            "source_subgraph": str(source_path),
            "output_explanation": str(output_path),
        }
        try:
            result = explainer.explain_file(source_path, output_path)
            sub_data = torch.load(source_path, map_location="cpu", weights_only=False)
            reason_codes = reason_code_generator.generate(result, sub_data)
            result["reason_codes"] = reason_codes
            FraudGNNExplainer.save_explanation(result, output_path, source_subgraph=source_path)
            transaction_id = (
                f"edge_{result['metadata'].get('target_edge_idx_global', source_path.stem)}"
            )
            reason_document = reason_code_generator.build_document(
                result,
                subgraph=sub_data,
                transaction_id=transaction_id,
                threshold=args.threshold,
                top_k=5,
            )
            reason_code_path = args.reason_codes_dir / f"transaction_{transaction_id}.json"
            reason_code_path.write_text(
                json.dumps(reason_document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            row.update(
                {
                    "num_nodes": int(result["node_importance"].numel()),
                    "num_edges": int(result["edge_mask"].numel()),
                    "prediction_probability": float(
                        result["prediction_probability"].reshape(-1)[0]
                    ),
                    "ground_truth_label": result["metadata"].get("ground_truth_label"),
                    "target_edge_idx_in_sub": result["target_edge_idx_in_sub"],
                    "max_edge_importance": float(result["edge_mask"].abs().max()),
                    "max_node_importance": float(result["node_importance"].max()),
                    "reason_codes": "|".join(
                        item["code"] for item in reason_codes["reason_codes"]
                    ),
                    "explanation_status": reason_codes["explanation_status"],
                    "reason_code_json": str(reason_code_path),
                    "status": "success",
                    "error_message": "",
                }
            )
            success += 1
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error_message": str(exc),
                }
            )
        manifest_rows.append(row)
        print(f"[{index}/{len(files)}] {row['status']}: {source_path.name}")
    manifest_path = args.output_dir / "manifest.csv"
    fields = sorted({key for row in manifest_rows for key in row})
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Hoàn tất: {success} thành công, {len(files) - success} thất bại")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
