"""Privacy-preserving, bounded transaction report generation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from src.gemini_client import GeminiClient, GeminiClientError


DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "config" / "prompts" / "fraud_transaction_report_vi.txt"


class TransactionReporter:
    def __init__(
        self,
        client: GeminiClient,
        prompt_file: Path | str = DEFAULT_PROMPT,
        top_k: int = 5,
        max_prompt_chars: int = 12000,
        max_string_length: int = 500,
    ) -> None:
        if top_k <= 0 or max_prompt_chars <= 0 or max_string_length <= 0:
            raise ValueError("Các giới hạn payload phải lớn hơn 0")
        self.client = client
        self.prompt_file = Path(prompt_file)
        self.top_k = top_k
        self.max_prompt_chars = max_prompt_chars
        self.max_string_length = max_string_length
        self.prompt_version = "v1"
        self._aliases: dict[str, str] = {}

    def _alias(self, value: Any, prefix: str) -> str:
        raw = str(value)
        if not raw:
            return raw
        if raw not in self._aliases:
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
            self._aliases[raw] = f"{prefix}_{digest}"
        return self._aliases[raw]

    def _sanitize(self, value: Any, key: str = "") -> Any:
        if isinstance(value, Mapping):
            return {
                key_name: self._sanitize(item, key_name)
                for key_name, item in value.items()
                if key_name not in {"source_account", "destination_account", "src_account", "dst_account"}
            }
        if isinstance(value, list):
            return [self._sanitize(item, key) for item in value[: self.top_k]]
        if isinstance(value, str):
            if "account" in key.lower() or key.lower() in {"transaction_id", "source_subgraph"}:
                prefix = "Account" if "account" in key.lower() else "Transaction"
                return self._alias(value, prefix)
            return value[: self.max_string_length]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)[: self.max_string_length]

    def build_payload(self, document: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        payload = self._sanitize(document)
        if isinstance(payload.get("reason_codes"), list):
            payload["reason_codes"] = payload["reason_codes"][: self.top_k]
        explanation = payload.get("explanation")
        if isinstance(explanation, dict):
            explanation["top_edges"] = explanation.get("top_edges", [])[: self.top_k]
            explanation["top_nodes"] = explanation.get("top_nodes", [])[: self.top_k]
            explanation["top_k"] = self.top_k
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        truncated = len(encoded) > self.max_prompt_chars
        if truncated:
            payload = {
                "transaction_id": payload.get("transaction_id", "unknown"),
                "prediction": payload.get("prediction", {}),
                "reason_codes": payload.get("reason_codes", [])[: self.top_k],
                "explanation": {
                    "top_edges": payload.get("explanation", {}).get("top_edges", [])[: self.top_k],
                    "top_nodes": payload.get("explanation", {}).get("top_nodes", [])[: self.top_k],
                },
                "payload_truncated": True,
            }
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            while len(encoded) > self.max_prompt_chars and payload["reason_codes"]:
                payload["reason_codes"].pop()
                encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            while len(encoded) > self.max_prompt_chars and (
                payload["explanation"]["top_edges"] or payload["explanation"]["top_nodes"]
            ):
                if payload["explanation"]["top_edges"]:
                    payload["explanation"]["top_edges"].pop()
                if len(encoded) > self.max_prompt_chars and payload["explanation"]["top_nodes"]:
                    payload["explanation"]["top_nodes"].pop()
                encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if len(encoded) > self.max_prompt_chars:
                payload = {"transaction_id": "unknown", "payload_truncated": True}
                encoded = json.dumps(payload, separators=(",", ":"))
            if len(encoded) > self.max_prompt_chars:
                raise ValueError("max_prompt_chars quá nhỏ để tạo payload hợp lệ")
        return {"json": encoded, "payload_truncated": truncated}, truncated

    def report_document(self, document: Mapping[str, Any], output_dir: Path, force: bool = False) -> dict[str, Any]:
        payload, truncated = self.build_payload(document)
        transaction_id = str(document.get("transaction_id", "unknown"))
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", transaction_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"transaction_{safe_id}.json"
        md_path = output_dir / f"transaction_{safe_id}.md"
        source_hash = hashlib.sha256(
            f"{payload['json']}|{self.prompt_version}|{self.client.model}".encode("utf-8")
        ).hexdigest()
        if not force and json_path.exists() and md_path.exists():
            try:
                cached = json.loads(json_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") == source_hash and cached.get("status") == "success":
                    return {"status": "skipped", "transaction_id": transaction_id}
            except (OSError, json.JSONDecodeError):
                pass
        try:
            instruction = self.prompt_file.read_text(encoding="utf-8")
            prompt = f"<transaction_data>\n{payload['json']}\n</transaction_data>"
            report = self.client.generate(prompt, instruction)
            result = {
                "transaction_id": transaction_id,
                "status": "success",
                "model": self.client.model,
                "prompt_version": self.prompt_version,
                "payload_truncated": truncated,
                "cache_key": source_hash,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "report": {"text": report},
            }
            md_path.write_text(report, encoding="utf-8")
        except GeminiClientError as exc:
            result = {
                "transaction_id": transaction_id,
                "status": "failed",
                "error_type": exc.error_type,
                "payload_truncated": truncated,
                "source_reason_code": [
                    item.get("code")
                    for item in document.get("reason_codes", [])
                    if isinstance(item, Mapping)
                ],
            }
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
