"""OpenAI-compatible LLM client for prompt enhancement."""
from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
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
    stream: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be nonempty")
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")


def load_system_prompt(name: Literal["text_to_image", "image_edit"]) -> str:
    """Load an optional image prompt enhancer template; selection is explicit."""
    files = {
        "text_to_image": "universal_image_prompt_v1_1.txt",
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
    def _user_message(prompt: str, size: str | None) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty")
        if size is None:
            return prompt
        match = (
            re.fullmatch(r"([1-9]\d*)\s*[xX×*]\s*([1-9]\d*)", size.strip())
            if isinstance(size, str) else None
        )
        if match is None:
            raise ValueError("size must be WIDTHxHEIGHT with positive integer dimensions")
        width, height = match.groups()
        return (
            "画布参数（用于规划布局，不是要绘制在图片上的文字；"
            "如与原始需求中的尺寸冲突，以此处为准）：\n"
            f"宽：{width} 像素\n高：{height} 像素\n\n"
            f"用户原始需求：\n{prompt}"
        )

    def _request(
        self,
        prompt: str,
        system_prompt: str | None,
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
        if self.config.api_style == "responses":
            body = {
                "input": prompt,
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
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        body = {
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "stream": self.config.stream,
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

    @classmethod
    def _finish_stream(cls, response: httpx.Response, debug: bool) -> dict[str, Any]:
        if not response.is_success:
            logger.error("HTTP %s: %s", response.status_code, response.read().decode("utf-8", errors="replace"))
            response.raise_for_status()
        chunks: list[str] = []
        first: dict[str, Any] | None = None
        finish_reason = None
        usage = None
        data_lines: list[str] = []
        done = False

        def consume(data: str) -> None:
            nonlocal first, finish_reason, usage, done
            if data == "[DONE]":
                done = True
                return
            try:
                event = json.loads(data)
            except ValueError as exc:
                raise ValueError("Invalid JSON stream event") from exc
            if not isinstance(event, dict):
                raise ValueError("Stream event must be a JSON object")
            if "error" in event:
                raise ValueError(f"LLM stream error: {event['error']}")
            choices = event.get("choices")
            if not isinstance(choices, list):
                raise ValueError("Stream event has invalid choices")
            if first is None and choices:
                first = event
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("index") != 0:
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    raise ValueError("Stream choice has invalid delta")
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        raise ValueError("Stream content must be text")
                    chunks.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]

        for line in response.iter_lines():
            if not line:
                if data_lines:
                    consume("\n".join(data_lines))
                    data_lines.clear()
                if done:
                    break
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        if data_lines and not done:
            consume("\n".join(data_lines))
        if first is None or not chunks:
            raise ValueError("Stream contained no prompt content")
        if finish_reason != "stop":
            raise ValueError(f"Stream did not finish successfully: {finish_reason!r}")
        if not done:
            raise ValueError("Stream ended without [DONE]")
        payload = {
            "id": first.get("id"),
            "object": "chat.completion",
            "created": first.get("created"),
            "model": first.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(chunks)},
                         "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        if debug:
            logger.info("output_prompt:\n%s", payload["choices"][0]["message"]["content"])
        return payload

    def enhance(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        size: str | None = None,
        request_body: Mapping[str, Any] | None = None,
        **body_fields: Any,
    ) -> Any:
        """Return a complete JSON response; assemble streamed chat content when enabled."""
        user_message = self._user_message(prompt, size)
        path, body, headers = self._request(user_message, system_prompt, request_body, body_fields)
        if not isinstance(body.get("stream", False), bool):
            raise ValueError("stream must be a boolean")
        if body.get("stream") and self.config.api_style != "chat_completions":
            raise ValueError("stream is supported only for chat_completions")
        debug = self._debug_enabled()
        self._log_request(user_message, body, debug)
        try:
            with httpx.Client(base_url=self.config.base_url.rstrip("/") + "/", timeout=self.config.timeout, trust_env=False) as client:
                if body.get("stream"):
                    with client.stream("POST", path, json=body, headers=headers) as response:
                        try:
                            return self._finish_stream(response, debug)
                        except ValueError as exc:
                            logger.error("Invalid stream response: %s", exc)
                            raise
                response = client.post(path, json=body, headers=headers)
        except httpx.RequestError as exc:
            logger.error("Request error: %s", exc)
            raise
        return self._finish(response, debug)
