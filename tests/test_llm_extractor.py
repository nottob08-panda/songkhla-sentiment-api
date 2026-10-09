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


# ---------------------------------------------------------------- v9: คัดลอกข้อความจากรีวิว (span)
from sentiment_core.llm_extractor import LLMPhraseExtractorV9, make_extractor  # noqa: E402


def sp(text, polarity="positive", aspect="", opinion=None):
    return {"text": text, "opinion": text if opinion is None else opinion, "aspect": aspect, "polarity": polarity}


def v9(*items):
    return json.dumps({"opinions": list(items)}, ensure_ascii=False)


def test_v9_keeps_adjacent_aspect_and_normalizes():
    transport, _ = replies(v9(sp("บรรยากาศดีมากกก"), sp("สะอาด"), sp("สวยงาม")))
    out = LLMPhraseExtractorV9("k", transport=transport).analyze("สะอาด สวยงาม บรรยากาศดีมากกก")
    assert out["positive_text"] == "บรรยากาศดี สะอาด สวย"


def test_v9_distant_aspect_from_field():
    text = "ห้องน้ำที่อยู่ด้านหลังค่อนข้างสกปรก"
    transport, _ = replies(v9(sp("ค่อนข้างสกปรก", "negative", "ห้องน้ำ")))
    assert LLMPhraseExtractorV9("k", transport=transport).analyze(text)["negative_text"] == "ห้องน้ำสกปรก"


def test_v9_rejects_text_not_in_review():
    transport, _ = replies(v9(sp("บรรยากาศดี"), sp("อาหารอร่อย")))
    kept, dropped = LLMPhraseExtractorV9("k", transport=transport).extract("บรรยากาศดี")
    assert [k["phrase"] for k in kept] == ["บรรยากาศดี"] and dropped[0]["phrase"] == "อาหารอร่อย"


def test_v9_ignores_ungrounded_aspect_but_never_invents_words():
    transport, _ = replies(v9(sp("สะอาด", aspect="ห้องน้ำ")))      # รีวิวไม่ได้พูดถึงห้องน้ำ
    assert LLMPhraseExtractorV9("k", transport=transport).analyze("สะอาดมาก")["positive_text"] == "สะอาด"


def test_v9_accepts_one_letter_typo_and_verb_first():
    transport, _ = replies(v9(sp("เสียดาย", "negative"), sp("ชอบ", aspect="ข้าวผัด")))
    out = LLMPhraseExtractorV9("k", transport=transport).analyze("เสีบดายเวลาน้อย ข้าวผัดอร่อย ชอบ")
    assert out["negative_text"] == "เสียดาย" and out["positive_text"] == "ชอบข้าวผัด"


def test_v9_rejects_existence_and_too_long():
    text = "มีสระน้ำ ห้องน้ำแยกสำหรับชายและหญิงที่สะอาดและกว้างขวางมาก"
    transport, _ = replies(v9(sp("สระน้ำมี"), sp("ห้องน้ำแยกสำหรับชายและหญิงที่สะอาดและกว้างขวาง")))
    kept, dropped = LLMPhraseExtractorV9("k", transport=transport).extract(text)
    assert kept == [] and len(dropped) == 2


def test_make_extractor_and_pipeline_prompt_switch():
    assert make_extractor("v9", "k").version == "llm-typhoon-v9.3"
    assert make_extractor("v8.1", "k").version == "llm-typhoon-v8.1"
    with pytest.raises(ValueError):
        make_extractor("v10", "k")
    transport, _ = replies(v9(sp("บรรยากาศดี")))
    a = ReviewAnalyzer(MODEL_DIR, translate=False, phrase_method="llm",
                       llm_extractor=LLMPhraseExtractorV9("k", transport=transport))
    r = a.analyze("บรรยากาศดี")
    assert r.fields["sentiment_text"] == "บรรยากาศดี" and r.fields["model_version"] == "nb-v1/llm-typhoon-v9.3"
    assert a.versions["llm_version"] == "llm-typhoon-v9.3"


def test_v91_requires_opinion_inside_text():
    transport, _ = replies(v9(sp("ผ้าทอเกาะยอ", opinion=""), sp("เงียบสงบ", aspect="วัด"),
                              sp("ค่าเข้าชม50บาท", opinion="แพง", polarity="negative")))
    kept, dropped = LLMPhraseExtractorV9("k", transport=transport).extract("มีผ้าทอเกาะยอ เป็นวัดที่เงียบสงบ ค่าเข้าชม 50 บาท")
    assert [k["phrase"] for k in kept] == ["วัดเงียบสงบ"]
    assert {d["phrase"] for d in dropped} == {"ผ้าทอเกาะยอ", "ค่าเข้าชม50บาท"}


def test_v91_strips_particles_and_zero_width():
    transport, _ = replies(v9(sp("บรรยากาศดีมากค่ะ", opinion="ดี"), sp("สวยงามมากครับ", opinion="สวยงาม"),
                              sp("น่าเที่ยวมากคร้า", opinion="น่าเที่ยว")))
    out = LLMPhraseExtractorV9("k", transport=transport).analyze(
        "บรรยากาศ\u200bดีมากค่ะ สวยงามมากครับ น่าเที่ยวมากคร้า")
    assert out["positive_text"] == "บรรยากาศดี สวย น่าเที่ยว"


# ---- v9.2: กู้คืนความเห็นจริงที่ v9.1 ทิ้ง (กรณีจากชุดทดสอบจริง) ----
def _it(t, o, a="", pol="negative"):
    return {"span": t, "opinion": o, "aspect": a, "polarity": pol}


def _kept(text, items):
    from sentiment_core.llm_extractor import process_v9
    return [k["phrase"] for k in process_v9(text, items)[0]]


def test_v92_salvage_long_or_inexact_span():
    assert _kept("แม้ว่ารสชาติอาหารจะอยู่ในระดับปานกลาง",
                 [_it("รสชาติอาหารอยู่ในระดับปานกลาง", "ปานกลาง", "รสชาติอาหาร")]) == ["รสชาติอาหารปานกลาง"]
    assert _kept("ประสบการณ์เยี่ยมและอาหารก็อร่อยทุกอย่าง",
                 [_it("อาหารอร่อย", "อร่อย", "อาหาร", "positive")]) == ["อาหารอร่อย"]
    assert _kept("ไม่มีของให้เลือกซื้อมากนัก",
                 [_it("ไม่มีของให้เลือกซื้อมากนัก", "ไม่มีของให้เลือกซื้อมากนัก", "ของให้เลือกซื้อ")]) == \
        ["ไม่มีของให้เลือกซื้อมากนัก"]


def test_v92_drop_thi_only_between_aspect_and_opinion():
    assert _kept("ราคาที่คุ้มค่ามาก", [_it("ราคาที่คุ้มค่า", "คุ้มค่า", "ราคา", "positive")]) == ["ราคาคุ้มค่า"]
    assert _kept("สถานที่ดีมาก", [_it("สถานที่ดี", "ดี", "สถานที่", "positive")]) == ["สถานที่ดี"]
    assert _kept("อยู่ในที่ร่ม", [_it("อยู่ในที่ร่ม", "ร่ม")]) == ["อยู่ในที่ร่ม"]


def test_v92_distant_aspect_and_rejections():
    assert _kept("ยังคงเป็นประสบการณ์ที่ยอดเยี่ยม", [_it("ยอดเยี่ยม", "ยอดเยี่ยม", "ประสบการณ์", "positive")]) == \
        ["ประสบการณ์ยอดเยี่ยม"]
    assert _kept("ลดการใช้พลาสติกได้อย่างดี", [_it("ลดการใช้พลาสติกได้อย่าง", "ลด", "", "positive")]) == []
    assert _kept("มีอาหารฮาลาล", [_it("อาหารฮาลาล", "", "อาหาร", "positive")]) == []
    assert _kept("ดีมาก", [_it("ดี", "ดี", "", "neutral")]) == []


# ---- v9.3: กรณีจากคำตอบดิบของ v9.2 บนชุดทดสอบ ----
def test_v93_no_duplicate_aspect_and_not_opinions():
    assert _kept("มื้อเช้าเป็นการเริ่มต้นที่ดี เริ่มต้นสมบูรณ์แบบมาก",
                 [_it("เริ่มต้นสมบูรณ์แบบ", "สมบูรณ์แบบ", "การเริ่มต้น", "positive")]) == ["เริ่มต้นสมบูรณ์แบบ"]
    assert _kept("บรรยากาศภายในสวนสัตว์", [_it("บรรยากาศภายในสวนสัตว์", "บรรยากาศ", "บรรยากาศ", "positive")]) == []
    assert _kept("จุดเด่นที่สุด", [_it("จุดเด่นที่สุด", "ที่สุด", "จุดเด่น", "positive")]) == []
    assert _kept("ไม่แนะนำให้วางของทิ้งไว้ท้ายรถ",
                 [_it("ไม่แนะนำให้วางของทิ้งไว้ท้ายรถ", "ไม่แนะนำ", "การวางของ")]) == []


def test_v93_normalization_fixes():
    assert _kept("รถใหญ่เข้าไม่ได้", [_it("รถใหญ่เข้าไม่ได้", "เข้าไม่ได้", "รถ")]) == ["รถใหญ่เข้าไม่ได้"]
    assert _kept("ทางขึ้นชัน รถไม่ค่อยมี", [_it("รถไม่ค่อยมี", "ไม่ค่อยมี", "รถ", "positive")]) == ["รถไม่มี"]
    assert _kept("ถ่ายรูปก็สวยด้วย", [_it("ถ่ายรูปก็สวย", "สวย", "", "positive")]) == ["ถ่ายรูปสวย"]
    assert _kept("เก็บค่าเข้าชมแพง", [_it("เก็บค่าเข้าชมแพง", "แพง", "ค่าเข้าชม")]) == ["เก็บค่าเข้าชมแพง"]
    assert _kept("เป็นประสบการณ์ที่ดีมาก", [_it("เป็นประสบการณ์ที่ดี", "ดี", "ประสบการณ์", "positive")]) == \
        ["ประสบการณ์ดี"]


def test_v93_salvage_rejects_bare_verbs():
    assert _kept("เหมาะกับการพาเด็กๆ มาเล่นน้ำที่น้ำตกชั้น 1",
                 [_it("เหมาะกับการพาเด็กมาเล่นน้ำที่น้ำตก", "เหมาะ", "น้ำตกชั้น1", "positive")]) == []
