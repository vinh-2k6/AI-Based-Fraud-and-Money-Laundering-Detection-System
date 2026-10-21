"""Generate optional Gemini reports from local reason-code JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.gemini_client import GeminiClient
from src.transaction_reporter import DEFAULT_PROMPT, TransactionReporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh báo cáo Gemini từ reason code JSON")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/reason_codes"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/transaction_reports"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-prompt-chars", type=int, default=12000)
    parser.add_argument("--max-string-length", type=int, default=500)
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("*.json"))
    if args.limit is not None:
        files = files[: args.limit]
    reporter = TransactionReporter(
        GeminiClient(model=args.gemini_model, timeout=args.timeout, retry_count=args.retry_count),
        prompt_file=args.prompt_file,
        top_k=args.top_k,
        max_prompt_chars=args.max_prompt_chars,
        max_string_length=args.max_string_length,
    )
    counts = {"success": 0, "skipped": 0, "failed": 0}
    for path in files:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            result = reporter.report_document(document, args.output_dir, force=args.force)
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            detail = result.get("error_type", "")
            suffix = f" ({detail})" if detail else ""
            print(f"{result['status']}: {path.name}{suffix}")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            counts["failed"] += 1
            print(f"failed: {path.name}: {exc}")
    print(
        f"Hoàn tất: {counts['success']} thành công, "
        f"{counts['skipped']} bỏ qua, {counts['failed']} thất bại"
    )


if __name__ == "__main__":
    main()
