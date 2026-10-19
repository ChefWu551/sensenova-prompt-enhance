"""OpenAI-compatible LLM client for prompt enhancement."""
from __future__ import annotations

import base64
import logging
import mimetypes
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import httpx

logger = logging.getLogger("pe_enhance")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PE] %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str = ""
    api_style: Literal["chat_completions", "responses"] = "chat_completions"
    api_key_env: str | None = None
    timeout: float = 60.0
    max_tokens: int = 4096
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be nonempty")


def load_system_prompt(name: Literal["text_to_image", "image_edit"]) -> str:
    """Load an optional SenseNova U1.5 prompt; selection is explicit."""
    files = {
        "text_to_image": "sensenova_u1_5_text_to_image_prompt.txt",
        "image_edit": "sensenova_u1_5_image_edit_prompt.txt",
    }
    if name not in files:
        raise ValueError(f"Unknown system prompt: {name}")
    return resources.files("pe_enhance").joinpath(files[name]).read_text(encoding="utf-8").strip()


class PromptEnhancer:
    """Call a PE LLM and return its complete JSON response."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @staticmethod
    def _image_url(image: str) -> str:
        value = os.fspath(image)
        if value.startswith(("http://", "https://", "data:")):
            return value
        path = Path(value).expanduser()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _request(
        self,
        prompt: str,
        system_prompt: str | None,
        images: Sequence[str],
        request_body: Mapping[str, Any] | None,
        body_fields: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty")
        headers: dict[str, str] = {}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        urls = [self._image_url(image) for image in images]
        if self.config.api_style == "responses":
            content: str | list[dict[str, Any]] = prompt
            if urls:
                content = [{"type": "input_image", "image_url": url} for url in urls]
                content.append({"type": "input_text", "text": prompt})
            body = {
                "input": content if isinstance(content, str) else [{"role": "user", "content": content}],
                "max_output_tokens": self.config.max_tokens,
                "model": self.config.model,
            }
            if system_prompt:
                body["instructions"] = system_prompt
            body.update(self.config.extra_body)
            body.update(request_body or {})
            body.update(body_fields)
            self._check_model(body)
            return "responses", body, headers
        content = prompt
        if urls:
            content = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
            content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        body = {
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "model": self.config.model,
        }
        body.update(self.config.extra_body)
        body.update(request_body or {})
        body.update(body_fields)
        self._check_model(body)
        return "chat/completions", body, headers

    @staticmethod
    def _check_model(body: Mapping[str, Any]) -> None:
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("request body model must be nonempty")

    @staticmethod
    def _output_prompt(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        try:
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        except (KeyError, IndexError, TypeError):
            pass
        parts = []
        for item in payload.get("output", []):
            if isinstance(item, dict):
                parts.extend(part.get("text", "") for part in item.get("content", []) if isinstance(part, dict) and part.get("type") == "output_text")
        return "".join(parts) if parts else None

    @staticmethod
    def _debug_enabled() -> bool:
        return os.getenv("PE_ENHANCE_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _log_request(prompt: str, body: Mapping[str, Any], debug: bool) -> None:
        if not debug:
            return
        logger.info("input_prompt:\n%s", prompt)
        system_prompt = body.get("instructions")
        if system_prompt is None:
            messages = body.get("messages")
            system_prompt = next(
                (item.get("content") for item in messages
                 if isinstance(item, dict) and item.get("role") == "system"),
                None,
            ) if isinstance(messages, list) else None
        logger.info("system_prompt:\n%s", system_prompt or "<not sent>")

    @classmethod
    def _finish(cls, response: httpx.Response, debug: bool) -> Any:
        if not response.is_success:
            logger.error("HTTP %s: %s", response.status_code, response.text)
            response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            logger.error("Invalid JSON response: %s", response.text)
            raise
        if debug:
            logger.info("output_prompt:\n%s", cls._output_prompt(payload) or "<unavailable>")
        return payload

    def enhance(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        images: Sequence[str] = (),
        request_body: Mapping[str, Any] | None = None,
        **body_fields: Any,
    ) -> Any:
        """Return the LLM's complete JSON body without parsing or changing it."""
        path, body, headers = self._request(prompt, system_prompt, images, request_body, body_fields)
        debug = self._debug_enabled()
        self._log_request(prompt, body, debug)
        try:
            with httpx.Client(base_url=self.config.base_url.rstrip("/") + "/", timeout=self.config.timeout, trust_env=False) as client:
                response = client.post(path, json=body, headers=headers)
        except httpx.RequestError as exc:
            logger.error("Request error: %s", exc)
            raise
        return self._finish(response, debug)

    async def aenhance(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        images: Sequence[str] = (),
        request_body: Mapping[str, Any] | None = None,
        **body_fields: Any,
    ) -> Any:
        """Async variant of enhance."""
        path, body, headers = self._request(prompt, system_prompt, images, request_body, body_fields)
        debug = self._debug_enabled()
        self._log_request(prompt, body, debug)
        try:
            async with httpx.AsyncClient(base_url=self.config.base_url.rstrip("/") + "/", timeout=self.config.timeout, trust_env=False) as client:
                response = await client.post(path, json=body, headers=headers)
        except httpx.RequestError as exc:
            logger.error("Request error: %s", exc)
            raise
        return self._finish(response, debug)
