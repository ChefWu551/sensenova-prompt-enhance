import json

import httpx
import pytest

from pe_enhance import LLMConfig, PromptEnhancer, load_system_prompt


def mock_sync(monkeypatch, handler):
    original = httpx.Client.__init__

    def init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", init)


def test_chat_returns_complete_response_and_logs_only_when_enabled(monkeypatch, caplog):
    expected = {
        "id": "chatcmpl-1", "object": "chat.completion", "model": "pe-v1",
        "usage": {"total_tokens": 37},
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "改写后的提示词"}}],
    }
    seen = []
    logs_at_request = []

    def respond(request):
        logs_at_request.append([record.getMessage() for record in caplog.records])
        seen.append((str(request.url), request.headers.get("authorization"), json.loads(request.content)))
        return httpx.Response(200, json=expected)

    mock_sync(monkeypatch, respond)
    monkeypatch.setenv("PE_API_KEY", "secret")
    monkeypatch.delenv("PE_ENHANCE_DEBUG", raising=False)
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "pe-v1", api_key_env="PE_API_KEY"))
    assert sdk.enhance("输入的提示词", system_prompt="系统提示词") == expected
    assert not caplog.records
    assert seen[0][0] == "https://example.com/v1/chat/completions"
    assert seen[0][1] == "Bearer secret"
    assert seen[0][2]["messages"] == [
        {"role": "system", "content": "系统提示词"},
        {"role": "user", "content": "输入的提示词"},
    ]
    monkeypatch.setenv("PE_ENHANCE_DEBUG", "true")
    assert sdk.enhance("输入的提示词", system_prompt="系统提示词") == expected
    messages = [record.getMessage() for record in caplog.records]
    assert "input_prompt:\n输入的提示词" in messages
    assert "system_prompt:\n系统提示词" in messages
    assert "output_prompt:\n改写后的提示词" in messages
    assert "secret" not in caplog.text
    assert logs_at_request[0] == []
    assert logs_at_request[1] == [
        "input_prompt:\n输入的提示词",
        "system_prompt:\n系统提示词",
    ]


def test_http_failure_always_logs_error(monkeypatch, caplog):
    mock_sync(monkeypatch, lambda request: httpx.Response(429, json={"error": {"message": "rate limit"}}))
    monkeypatch.delenv("PE_ENHANCE_DEBUG", raising=False)
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "pe-v1"))
    with pytest.raises(httpx.HTTPStatusError):
        sdk.enhance("hello", system_prompt="rewrite")
    assert "HTTP 429" in caplog.text
    assert "rate limit" in caplog.text
    assert all(record.levelname == "ERROR" for record in caplog.records)


def test_request_prompts_log_before_failed_http_response(monkeypatch, caplog):
    monkeypatch.setenv("PE_ENHANCE_DEBUG", "1")
    mock_sync(monkeypatch, lambda request: httpx.Response(503, text="unavailable"))
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "pe-v1"))
    with pytest.raises(httpx.HTTPStatusError):
        sdk.enhance("输入", system_prompt="系统指令")
    assert [record.getMessage() for record in caplog.records] == [
        "input_prompt:\n输入",
        "system_prompt:\n系统指令",
        "HTTP 503: unavailable",
    ]


def test_image_edit_prompt_uses_text_only(monkeypatch):
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "修改原图中的天空，保持其他内容不变"}}]})

    mock_sync(monkeypatch, respond)
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "text-only-pe"))
    sdk.enhance("把天空改成晚霞", system_prompt=load_system_prompt("image_edit"))
    assert seen[0]["messages"][1]["content"] == "把天空改成晚霞"
    assert "看不到底图或参考图" in seen[0]["messages"][0]["content"]


@pytest.mark.parametrize("api_style", ["chat_completions", "responses"])
@pytest.mark.parametrize("template", ["text_to_image", "image_edit"])
def test_canvas_size_and_original_prompt_share_one_user_message(monkeypatch, api_style, template):
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "result"})

    mock_sync(monkeypatch, respond)
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "pe-v1", api_style=api_style))
    system_prompt = load_system_prompt(template)
    sdk.enhance("生成一张海报", system_prompt=system_prompt, size="2048x4096")
    body = seen[0]
    if api_style == "responses":
        content = body["input"]
        assert body["instructions"] == system_prompt
    else:
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
        assert body["messages"][0]["content"] == system_prompt
        content = body["messages"][1]["content"]
    assert "宽：2048 像素\n高：4096 像素" in content
    assert content.endswith("用户原始需求：\n生成一张海报")
    assert "size" not in body


@pytest.mark.parametrize("size", ["2048", "0x4096", "2048x", "auto", 2048])
def test_canvas_size_requires_two_positive_dimensions(size):
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "pe-v1"))
    with pytest.raises(ValueError, match="size"):
        sdk.enhance("生成一张海报", size=size)


@pytest.mark.parametrize("size", ["2048x4096", "2048×4096", "2048*4096"])
def test_canvas_size_accepts_common_separators(size):
    content = PromptEnhancer._user_message("生成一张海报", size)
    assert "宽：2048 像素\n高：4096 像素" in content


def test_responses_returns_complete_response(monkeypatch, caplog):
    expected = {"id": "resp-1", "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "clear scene"}]}], "usage": {"total_tokens": 24}}
    seen = []
    logs_at_request = []

    def respond(request):
        logs_at_request.append([record.getMessage() for record in caplog.records])
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json=expected)

    mock_sync(monkeypatch, respond)
    monkeypatch.setenv("PE_ENHANCE_DEBUG", "1")
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "pe-v1", api_style="responses"))
    assert sdk.enhance("brief", system_prompt="rewrite", top_p=0.6,
                       chat_template_kwargs={"enable_thinking": False}) == expected
    assert seen[0][0] == "https://example.com/v1/responses"
    assert seen[0][1]["instructions"] == "rewrite"
    assert seen[0][1]["input"] == "brief"
    assert seen[0][1]["top_p"] == 0.6
    assert seen[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert logs_at_request == [["input_prompt:\nbrief", "system_prompt:\nrewrite"]]
    assert "output_prompt:\nclear scene" in [record.getMessage() for record in caplog.records]


def test_only_prompt_url_and_model_are_required(monkeypatch):
    monkeypatch.delenv("PE_API_KEY", raising=False)
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "pe-v1", api_key_env="PE_API_KEY"))
    seen = []

    def respond(request):
        seen.append((request.headers.get("authorization"), json.loads(request.content)))
        return httpx.Response(200, json={"choices": []})

    mock_sync(monkeypatch, respond)
    assert sdk.enhance("hello") == {"choices": []}
    assert seen[0][0] is None
    assert seen[0][1]["model"] == "pe-v1"
    assert seen[0][1]["messages"] == [{"role": "user", "content": "hello"}]
    assert "temperature" not in seen[0][1]

    with pytest.raises(ValueError, match="prompt"):
        sdk.enhance("  ")
    with pytest.raises(ValueError, match="base_url"):
        LLMConfig(" ", "pe-v1")
    with pytest.raises(ValueError, match="model"):
        PromptEnhancer(LLMConfig("https://example.com/v1", " ")).enhance("hello")


def test_optional_request_fields_are_not_required(monkeypatch):
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "raw-response"})

    mock_sync(monkeypatch, respond)
    sdk = PromptEnhancer(
        LLMConfig("https://example.com/v1", "pe-v1", max_tokens=0,
                  extra_body={"model": "ignored", "custom": "value"})
    )
    assert sdk.enhance("hello", system_prompt="") == {"id": "raw-response"}
    assert seen[0]["model"] == "ignored"
    assert seen[0]["custom"] == "value"
    assert seen[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_per_request_fields_pass_through_without_filtering(monkeypatch):
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "complete", "usage": {"total_tokens": 7}})

    mock_sync(monkeypatch, respond)
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "default-model", extra_body={"top_p": 0.8}))
    custom_messages = [{"role": "user", "content": "custom request"}]
    result = sdk.enhance(
        "原始提示词",
        model="chosen-model",
        top_p=0.3,
        temperature=0.2,
        response_format={"type": "json_object"},
        chat_template_kwargs={"enable_thinking": True},
        request_body={"messages": custom_messages, "vendor_option": {"enabled": True}},
    )
    assert result == {"id": "complete", "usage": {"total_tokens": 7}}
    assert seen[0]["model"] == "chosen-model"
    assert seen[0]["top_p"] == 0.3
    assert seen[0]["temperature"] == 0.2
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert seen[0]["vendor_option"] == {"enabled": True}
    assert seen[0]["messages"] == custom_messages


def test_final_model_must_be_nonempty_even_with_overrides():
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1"))
    with pytest.raises(ValueError, match="model"):
        sdk.enhance("hello", model="")


def test_stream_assembles_content_and_logs_once(monkeypatch, caplog):
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        events = [
            {"id": "stream-1", "object": "chat.completion.chunk", "created": 1,
             "model": "pe-v1", "choices": [{"index": 0, "delta": {"reasoning": "thinking", "content": "改写"}, "finish_reason": None}]},
            {"id": "stream-1", "object": "chat.completion.chunk", "created": 1,
             "model": "pe-v1", "choices": [{"index": 0, "delta": {"content": "后的提示词"}, "finish_reason": "stop"}]},
            {"id": "stream-1", "choices": [], "usage": {"total_tokens": 12}},
        ]
        data = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=data)

    mock_sync(monkeypatch, respond)
    monkeypatch.setenv("PE_ENHANCE_DEBUG", "1")
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "pe-v1", stream=True))
    result = sdk.enhance("原文")
    assert seen[0]["stream"] is True
    assert result["choices"][0]["message"]["content"] == "改写后的提示词"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {"total_tokens": 12}
    assert [r.getMessage() for r in caplog.records].count("output_prompt:\n改写后的提示词") == 1


@pytest.mark.parametrize("events, message", [
    ([{"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}], "did not finish"),
    ([{"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": "length"}]}], "did not finish"),
    ([{"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": "stop"}]}], "without \\[DONE\\]"),
    ([{"error": {"message": "server failed"}}], "server failed"),
    ([{"choices": [{"index": 0, "delta": {"reasoning": "hidden"}, "finish_reason": "stop"}]}], "no prompt content"),
])
def test_stream_rejects_incomplete_or_failed_result(monkeypatch, events, message):
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    mock_sync(monkeypatch, lambda request: httpx.Response(200, text=body))
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "pe-v1", stream=True))
    with pytest.raises(ValueError, match=message):
        sdk.enhance("hello")


def test_stream_override_and_http_failure(monkeypatch, caplog):
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(503, text="unavailable")

    mock_sync(monkeypatch, respond)
    sdk = PromptEnhancer(LLMConfig("https://example.com/v1", "pe-v1"))
    with pytest.raises(httpx.HTTPStatusError):
        sdk.enhance("hello", stream=True)
    assert seen[0]["stream"] is True
    assert "HTTP 503" in caplog.text
