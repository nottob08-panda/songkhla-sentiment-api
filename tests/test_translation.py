"""ทดสอบลำดับการแปลภาษา โดยจำลองคำตอบของ Typhoon API (รูปแบบเดียวกับ OpenAI chat completions)"""
import json

import httpx
import pytest

from sentiment_core import language
from sentiment_core.language import TranslationError, _typhoon, translate_to_thai


def typhoon_reply(content, status=200):
    def handler(request):
        body = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer test-key"
        assert body["temperature"] == 0 and body["messages"][1]["content"]
        if status != 200:
            return httpx.Response(status, json={"error": "rate limited"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})
    return httpx.MockTransport(handler)


def test_typhoon_success():
    out = _typhoon("Beautiful beach", 5, "test-key", transport=typhoon_reply("ชายหาดสวยงาม"))
    assert out == "ชายหาดสวยงาม"


def test_typhoon_strips_think_and_quotes():
    out = _typhoon("x", 5, "test-key", transport=typhoon_reply('<think>reasoning</think>\n"ทะเลสะอาดมาก"'))
    assert out == "ทะเลสะอาดมาก"


def test_typhoon_rejects_non_thai():
    with pytest.raises(TranslationError):
        _typhoon("x", 5, "test-key", transport=typhoon_reply("Beautiful beach"))


def test_typhoon_rate_limit_raises():
    with pytest.raises(httpx.HTTPStatusError):
        _typhoon("x", 5, "test-key", transport=typhoon_reply("", status=429))


def test_order_typhoon_first(monkeypatch):
    called = []
    monkeypatch.setattr(language, "_typhoon", lambda *a, **k: called.append("typhoon") or "ชายหาดสวย")
    monkeypatch.setattr(language, "_google", lambda *a, **k: called.append("google") or "ไม่ควรถูกเรียก")
    assert translate_to_thai("Beautiful beach", "en", typhoon_api_key="k") == ("ชายหาดสวย", "typhoon")
    assert called == ["typhoon"]


def test_falls_back_when_typhoon_fails(monkeypatch):
    def boom(*a, **k):
        raise TranslationError("429")
    monkeypatch.setattr(language, "_typhoon", boom)
    monkeypatch.setattr(language, "_google", lambda *a, **k: "ชายหาดสวย")
    assert translate_to_thai("Beautiful beach", "en", typhoon_api_key="k") == ("ชายหาดสวย", "google")


def test_no_key_skips_typhoon(monkeypatch):
    monkeypatch.setattr(language, "_typhoon", lambda *a, **k: pytest.fail("should not call Typhoon"))
    monkeypatch.setattr(language, "_google", lambda *a, **k: "ชายหาดสวย")
    assert translate_to_thai("Beautiful beach", "en") == ("ชายหาดสวย", "google")


def test_all_fail_error_lists_every_provider(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("429 Too Many Requests")
    for name in ("_typhoon", "_google", "_mymemory"):
        monkeypatch.setattr(language, name, boom)
    with pytest.raises(TranslationError) as e:
        translate_to_thai("Beautiful beach", "en", typhoon_api_key="k")
    assert all(p in str(e.value) for p in ("typhoon", "google", "mymemory"))
