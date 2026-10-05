"""ทดสอบ API โดยจำลองฐานข้อมูล Supabase และบริการแปลภาษา (ไม่ต้องต่ออินเทอร์เน็ต)"""
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from api import app as appmod
from sentiment_core.language import TranslationError

SECRET = "test-secret"


class FakeSupabase:
    def __init__(self):
        self.rows = {}
        self.patches = []

    def handler(self, request):
        if request.method == "GET":
            rows = [r for r in self.rows.values() if r["analysis_status"] in ("pending", "failed")]
            return httpx.Response(200, json=[{"reviewId": r["reviewId"], "text": r["text"]} for r in rows])
        if request.method == "PATCH":
            rid = request.url.params["reviewId"].removeprefix("eq.")
            body = json.loads(request.content)
            self.patches.append((rid, body))
            if self.rows.get(rid, {}).get("analysis_status") in ("pending", "failed"):
                self.rows[rid].update(body)
            return httpx.Response(204)
        return httpx.Response(405)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    fake = FakeSupabase()

    def fake_translator(text, lang, **kw):
        if "FAIL" in text:
            raise TranslationError("simulated outage")
        return {"Beautiful view, very impressive": "วิวสวยงาม ประทับใจมาก",
                "Dirty toilets, not recommended": "ห้องน้ำสกปรก ไม่แนะนำ"}[text], "fake"

    with TestClient(appmod.app) as client:            # เปิด lifespan -> โหลดโมเดลจริง
        appmod.analyzer._translator = fake_translator
        appmod.repo = appmod.SupabaseRepo("https://fake.supabase.co", "key",
                                          transport=httpx.MockTransport(fake.handler))
        yield client, fake


def test_health(env):
    client, _ = env
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_version"] == "nb-v1" and r["lexicon_version"] == "lexicon-1"


def test_analyze_thai(env):
    client, _ = env
    t = time.time()
    r = client.post("/v1/analyze", json={"text": "วิวสวยมาก อาหารอร่อย แต่ห้องน้ำไม่ค่อยสะอาด"}).json()
    assert time.time() - t < 3.0
    assert r["status"] == "done" and r["originalLanguage"] == "th"
    assert r["sentiment"] in ("positive", "neutral", "negative")
    assert abs(r["prob_positive"] + r["prob_neutral"] + r["prob_negative"] - 1) < 1e-3
    assert r["sentiment_text"] == "วิวสวย อาหารอร่อย ห้องน้ำไม่สะอาด"
    assert r["model_version"] == "nb-v1/lexicon-1"


def test_analyze_symbols_only(env):
    client, _ = env
    r = client.post("/v1/analyze", json={"text": "❤️❤️ ^_^"}).json()
    assert r["status"] == "done" and r["analyzable"] == "no" and r["originalLanguage"] is None
    assert r["sentiment"] is None and r["sentiment_text"] is None


def test_webhook_english_review_end_to_end(env):
    client, fake = env
    fake.rows["R007202"] = {"reviewId": "R007202", "text": "Beautiful view, very impressive",
                            "analysis_status": "pending"}
    payload = {"type": "INSERT", "table": "reviews", "record": fake.rows["R007202"]}
    r = client.post("/v1/webhook/reviews", json=payload, headers={"x-webhook-secret": SECRET})
    assert r.status_code == 202
    row = fake.rows["R007202"]                       # background task runs before TestClient returns
    assert row["analysis_status"] == "done" and row["analysis_error"] is None
    assert row["originalLanguage"] == "en" and row["processed_text"] == "วิวสวยงาม ประทับใจมาก"
    assert row["sentiment"] == "positive"
    assert "translation_provider" not in row          # mapped to null in column_map.json -> not written


def test_webhook_rejects_bad_secret(env):
    client, _ = env
    r = client.post("/v1/webhook/reviews", json={"record": {"reviewId": "X"}},
                    headers={"x-webhook-secret": "wrong"})
    assert r.status_code == 401


def test_translation_failure_then_retry(env):
    client, fake = env
    fake.rows["R007203"] = {"reviewId": "R007203", "text": "FAIL this english text",
                            "analysis_status": "pending"}
    client.post("/v1/webhook/reviews", json={"record": fake.rows["R007203"]},
                headers={"x-webhook-secret": SECRET})
    assert fake.rows["R007203"]["analysis_status"] == "failed"
    assert "translation failed" in fake.rows["R007203"]["analysis_error"]
    # บริการแปลกลับมาใช้ได้ -> ตัวสำรองตามเวลาประมวลผลซ้ำ
    fake.rows["R007203"]["text"] = "Dirty toilets, not recommended"
    r = client.post("/v1/process-pending", headers={"x-webhook-secret": SECRET}).json()
    assert r["processed"] == 1 and r["done"] == 1
    assert fake.rows["R007203"]["analysis_status"] == "done"
    assert fake.rows["R007203"]["sentiment"] == "negative"


def test_done_rows_are_not_overwritten(env):
    client, fake = env
    fake.rows["R000001"] = {"reviewId": "R000001", "text": "วิวสวย", "analysis_status": "done",
                            "sentiment": "positive"}
    client.post("/v1/webhook/reviews", json={"record": {"reviewId": "R000001", "text": "แย่มาก"}},
                headers={"x-webhook-secret": SECRET})
    assert fake.rows["R000001"]["sentiment"] == "positive"
