"""Small, lazy Gemini REST client used only by the optional report pipeline."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import error, request


class GeminiClientError(RuntimeError):
    """An actionable Gemini API failure without exposing request secrets."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        retry_count: int = 2,
        opener: Any = request.urlopen,
    ) -> None:
        if timeout <= 0 or retry_count < 0:
            raise ValueError("timeout phải > 0 và retry_count không âm")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.timeout = timeout
        self.retry_count = retry_count
        self._opener = opener

    def generate(self, prompt: str, system_instruction: str) -> str:
        if not self.api_key:
            raise GeminiClientError("missing_api_key", "Thiếu biến môi trường GEMINI_API_KEY")
        if not prompt.strip():
            raise GeminiClientError("invalid_request", "Prompt rỗng")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }).encode("utf-8")
        last_error: GeminiClientError | None = None
        for attempt in range(self.retry_count + 1):
            try:
                req = request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self._opener(req, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = self._extract_text(payload)
                if not text:
                    raise GeminiClientError("empty_response", "Gemini trả về response rỗng")
                return text
            except error.HTTPError as exc:
                status = exc.code
                response_detail = ""
                try:
                    response_detail = exc.read().decode("utf-8", errors="replace")
                except OSError:
                    pass
                error_type = {
                    400: "http_400",
                    401: "http_401",
                    403: "http_403",
                    404: "http_404_model_not_found",
                    429: "http_429",
                }.get(status, "http_5xx" if status >= 500 else "http_error")
                last_error = GeminiClientError(
                    error_type,
                    self._http_error_message(status, response_detail),
                )
                if status not in (429,) and status < 500:
                    break
            except (TimeoutError, error.URLError, json.JSONDecodeError) as exc:
                last_error = GeminiClientError("timeout_or_transport", type(exc).__name__)
            except GeminiClientError as exc:
                last_error = exc
                break
            if attempt < self.retry_count:
                time.sleep(min(2**attempt, 4))
        raise last_error or GeminiClientError("unknown", "Gemini request thất bại")

    @staticmethod
    def _http_error_message(status: int, response_detail: str) -> str:
        """Return actionable diagnostics without returning request data or secrets."""
        detail = ""
        try:
            parsed = json.loads(response_detail)
            message = parsed.get("error", {}).get("message", "")
            if isinstance(message, str):
                detail = message[:300]
        except json.JSONDecodeError:
            pass
        suffix = f": {detail}" if detail else ""
        return f"Gemini HTTP {status}{suffix}"

    @staticmethod
    def _extract_text(payload: Any) -> str:
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        return "\n".join(
            str(part["text"]) for part in parts if isinstance(part, dict) and part.get("text")
        ).strip()
