"""ทดสอบการสกัดวลีด้วย LLM โดยจำลองคำตอบของ Typhoon API (ไม่ต้องต่ออินเทอร์เน็ต)"""
import json
from pathlib import Path

import httpx
import pytest

from sentiment_core import typhoon
from sentiment_core.llm_extractor import (LLMExtractionError, LLMPhraseExtractor, compose, normalize_phrase,
                                          parse_response)
from sentiment_core.pipeline import ReviewAnalyzer

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    monkeypatch.setattr(typhoon, "MIN_INTERVAL", 0)


def replies(*contents, status=200):
    """ตอบตามลำดับ; status != 200 = ตอบ error ทุกครั้ง"""
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["messages"][-1]["content"])
        if status != 200:
            return httpx.Response(status, json={"error": "x"})
        content = contents[min(len(calls) - 1, len(contents) - 1)]
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    return httpx.MockTransport(handler), calls


def op(aspect, opinion, polarity="positive"):
    return {"aspect": aspect, "opinion": opinion, "polarity": polarity}


def as_json(*ops):
    return json.dumps({"opinions": list(ops)}, ensure_ascii=False)


def test_compose_and_normalize():
    assert compose("วิว", "สวย") == "วิวสวย"
    assert compose("ที่นี่", "สวย") == "สวย"                 # หัวข้อกว้าง -> ตัดทิ้ง
    assert compose("ข้าวผัด", "ชอบ") == "ชอบข้าวผัด"          # คำความรู้สึกนำหน้า
    assert compose("สวนร่มรื่น", "ร่มรื่น") == "สวนร่มรื่น"     # ไม่ซ้ำคำ (พบใน v8 รอบแรก)
    assert normalize_phrase("วิวสวยงามมากที่สุด") == "วิวสวย"
    assert normalize_phrase("ห้องน้ำไม่ค่อยสะอาด") == "ห้องน้ำไม่สะอาด"


def test_parse_handles_think_and_code_fence():
    raw = "<think>x</think>```json\n" + as_json(op("วิว", "สวย")) + "\n```"
    assert parse_response(raw) == [{"phrase": "วิวสวย", "polarity": "positive", "opinion_like": True}]
    assert parse_response("not json") is None


def test_extracts_phrases_outside_lexicon():
    text = "มีอาหารให้เลือกมากมาย แต่ห้องน้ำค่อนข้างเก่าและที่จอดรถหายาก"
    transport, _ = replies(as_json(op("อาหาร", "มากมาย"), op("ห้องน้ำ", "เก่า", "negative"),
                                   op("ที่จอดรถ", "หายาก", "negative")))
    out = LLMPhraseExtractor("k", transport=transport).analyze(text)
    assert out["positive_text"] == "อาหารมากมาย"
    assert out["negative_text"] == "ห้องน้ำเก่า ที่จอดรถหายาก"
    assert out["n_sentiment_terms"] == 3


def test_drops_hallucinated_phrases():
    transport, _ = replies(as_json(op("วิว", "สวย"), op("อาหาร", "อร่อย")))   # รีวิวไม่ได้พูดถึงอาหาร
    kept, dropped = LLMPhraseExtractor("k", transport=transport).extract("วิวสวยมาก")
    assert [k["phrase"] for k in kept] == ["วิวสวย"]
    assert [d["phrase"] for d in dropped] == ["อาหารอร่อย"]


def test_empty_opinions_is_valid_result():
    transport, _ = replies(as_json())
    out = LLMPhraseExtractor("k", transport=transport).analyze("เปิดทุกวัน 9 โมง")
    assert out == {"sentiment_text": None, "positive_text": None, "negative_text": None, "n_sentiment_terms": 0}


def test_retries_bad_json_then_raises():
    transport, calls = replies("oops")
    with pytest.raises(LLMExtractionError):
        LLMPhraseExtractor("k", transport=transport).analyze("วิวสวย")
    assert len(calls) == 2


def test_long_review_falls_back_to_chunks():
    text = ("วิวสวยมาก " * 40) + "แต่ห้องน้ำสกปรก " * 20
    transport, calls = replies("broken", "broken", as_json(op("วิว", "สวย")), as_json(op("ห้องน้ำ", "สกปรก", "negative")))
    out = LLMPhraseExtractor("k", transport=transport).analyze(text)
    assert "วิวสวย" in out["sentiment_text"] and len(calls) > 2


def test_pipeline_uses_llm_and_records_version():
    transport, _ = replies(as_json(op("น้ำ", "ใสแจ๋ว")))
    a = ReviewAnalyzer(MODEL_DIR, translate=False, phrase_method="llm",
                       llm_extractor=LLMPhraseExtractor("k", transport=transport))
    r = a.analyze("น้ำทะเลใสแจ๋วมาก")
    assert r.status == "done"
    assert r.fields["sentiment_text"] == "น้ำใสแจ๋ว"
    assert r.fields["model_version"] == "nb-v1/llm-typhoon-v8.1"


def test_pipeline_falls_back_to_lexicon_when_typhoon_down():
    transport, _ = replies(status=429)
    a = ReviewAnalyzer(MODEL_DIR, translate=False, phrase_method="llm",
                       llm_extractor=LLMPhraseExtractor("k", transport=transport))
    r = a.analyze("วิวสวยมาก บรรยากาศดี")
    assert r.status == "done"
    assert r.fields["sentiment_text"] == "วิวสวย บรรยากาศดี"
    assert r.fields["model_version"] == "nb-v1/lexicon-1"


def test_llm_mode_requires_key():
    with pytest.raises(ValueError):
        ReviewAnalyzer(MODEL_DIR, translate=False, phrase_method="llm")


# ---------------------------------------------------------------- กฎของ v8 (จากผลตรวจด้วยมือ)
def test_v8_drops_existence_only_and_function_words():
    text = "ในสวนมีสระน้ำ มีของให้ซื้อมากมาย ที่สุดเลยวัดนี้"
    transport, _ = replies(as_json(op("สระน้ำ", "มี"), op("ของให้ซื้อ", "มากมาย"), op("", "ที่")))
    kept, dropped = LLMPhraseExtractor("k", transport=transport).extract(text)
    assert [k["phrase"] for k in kept] == ["ของให้ซื้อมากมาย"]
    assert {d["phrase"] for d in dropped} == {"สระน้ำมี", "ที่"}


def test_v8_keeps_absence_complaint():
    transport, _ = replies(as_json(op("รถประจำทาง", "ไม่มี", "negative")))
    out = LLMPhraseExtractor("k", transport=transport).analyze("ไม่มีรถประจำทางเลย")
    assert out["negative_text"] == "รถประจำทางไม่มี"


def test_v8_allows_six_words():
    text = "ลานจอดรถกว้าง จอดได้หลายคัน"
    transport, _ = replies(as_json(op("ลานจอดรถ", "จอดได้หลายคัน")))
    out = LLMPhraseExtractor("k", transport=transport).analyze(text)
    assert out["positive_text"] == "ลานจอดรถจอดได้หลายคัน"


def test_v8_grounding_accepts_one_letter_typo():
    # รีวิวสะกด "เสีบดาย" LLM ตอบ "เสียดาย" -> v7 ตัดทิ้ง, v8 เก็บไว้
    transport, _ = replies(as_json(op("", "เสียดาย", "negative")))
    out = LLMPhraseExtractor("k", transport=transport).analyze("เสีบดายมีเวลาน้อย")
    assert out["negative_text"] == "เสียดาย"


def test_v8_grounding_still_rejects_paraphrase():
    transport, _ = replies(as_json(op("แม่ค้า", "เยาะเย้ย", "negative")))
    kept, dropped = LLMPhraseExtractor("k", transport=transport).extract("แม่ค้าทำหน้า เหยียดลูกค้า")
    assert kept == [] and dropped[0]["phrase"] == "แม่ค้าเยาะเย้ย"


def test_two_phase_analyze():
    transport, _ = replies(as_json(op("วิว", "สวย")))
    a = ReviewAnalyzer(MODEL_DIR, translate=False, phrase_method="llm",
                       llm_extractor=LLMPhraseExtractor("k", transport=transport))
    r1 = a.analyze("วิวสวยมาก", with_phrases=False)
    assert r1.status == "scored" and r1.fields["sentiment"] == "positive" and "sentiment_text" not in r1.fields
    r2 = a.complete(r1)
    assert r2.status == "done" and r2.fields["sentiment_text"] == "วิวสวย" and r2.fields["sentiment"] == "positive"
