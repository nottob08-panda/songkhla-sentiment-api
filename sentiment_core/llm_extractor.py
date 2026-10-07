"""สกัดวลีแสดงความรู้สึกด้วย LLM (Typhoon API) -- ทางเลือกแทนพจนานุกรม (lexicon)

ต่อยอดจาก prompt v7 (สคริปต์ extract_phrases_with_typhoon.py ที่เคยรันผ่าน Ollama) เป็น v8 ตามผลตรวจด้วยมือ 2026-10-07:
  - คัดลอกคำจากรีวิวตรงตัว ห้ามถอดความ (v7 ตอบ "เยาะเย้ย" แทน "เหยียดลูกค้า" แล้วถูก grounding ตัดทิ้ง)
  - นิยาม "มี X": มีเฉยๆ ไม่ใช่ความคิดเห็น / มี X + คำประเมินจำนวนหรือคุณภาพ เป็นความคิดเห็น
  - คำแนะนำ/คำสั่ง (เลือกร้านที่สะอาด) ไม่ใช่ความคิดเห็น
  - วลียาวได้ถึง 6 คำ, ยอมรับคำสะกดผิด 1 ตัวอักษร, ตัดวลี "X มี" และวลีเพี้ยน

ขั้นตอน
  1. LLM ตอบ JSON: [{aspect, opinion, polarity}]   (หัวข้อ / คำประเมิน / บวก-ลบ)
  2. โค้ดประกอบวลีเอง เช่น วิว + สวย -> "วิวสวย"
  3. ทำให้เป็นรูปแบบเดียวกัน (ตัดคำขยาย "มาก" "ที่สุด", สวยงาม -> สวย)
  4. ตรวจว่าวลีมาจากรีวิวจริง (grounding) และไม่ยาวเกิน 4 คำ -- กัน LLM แต่งคำขึ้นเอง
ผลลัพธ์ใช้ชื่อเดียวกับ PhraseExtractor.analyze() จึงสลับใช้แทนกันได้ทันที
"""
import json
import re

from pythainlp.tokenize import word_tokenize

from . import typhoon

VERSION = "llm-typhoon-v8.1"   # v8.1 = แก้คำซ้ำตอนประกอบวลี (สวนร่มรื่นร่มรื่น)
MAX_CHARS = 1500       # รีวิวยาวมากตัดท้ายทิ้ง (เท่ากับสคริปต์เดิม)
CHUNK_CHARS = 300      # ใช้แบ่งรีวิวเฉพาะเมื่อ JSON เสีย
MAX_TOKENS = 6         # วลียาวเกินกี่คำถือว่าไม่ใช่วลีสั้น (v7 = 4 ตัดวลีที่ผู้ตรวจต้องการทิ้ง)
MAX_RETRIES = 2


class LLMExtractionError(Exception):
    pass


SYSTEM_PROMPT = """คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์ความคิดเห็นจากรีวิวสถานที่ท่องเที่ยวภาษาไทย
หน้าที่: หา "ความคิดเห็น" ให้ครบทุกจุดในรีวิว (รีวิวยาวมักมีหลายจุด) แต่ละจุดตอบเป็น 3 ช่อง

- opinion  = คำประเมิน 1-2 คำ ที่บอกคุณภาพหรือความรู้สึก คัดลอกจากรีวิวตรงตัว
             เช่น ดี สวย อร่อย สะอาด ใส ร่มรื่น น่ารัก ยิ้มแย้ม คุ้มค่า ชอบ แนะนำ ประทับใจ น่าสนใจ ฟรี
                  แพง สกปรก แออัด ร้อน รก รุงรัง น่าเบื่อ เสียดาย ผิดหวัง เหม็น
             ถ้ามีคำปฏิเสธให้ใส่ "ไม่" นำหน้า เช่น ไม่สะอาด ไม่แพง (คง "ไม่มี" ไว้ เช่น ไม่มีขยะ)
             ห้ามใส่คำขยายระดับ (มาก ที่สุด ค่อนข้าง เกินไป จริงๆ)
- aspect   = สิ่งที่ถูกชมหรือติ 1-2 คำ คัดลอกจากรีวิว เช่น วิว อาหาร ห้องน้ำ เจ้าหน้าที่ ต้นไม้ ชายหาด ราคา
             ถ้าคำนามอยู่ติดหน้าคำประเมิน ให้ใส่เป็น aspect เสมอ เช่น "คุณภาพดี" -> aspect คุณภาพ, "ราคาประหยัด" -> aspect ราคา
             "เหมาะกับครอบครัว" -> aspect ครอบครัว opinion เหมาะกับ
             ถ้าไม่ระบุ หรือเป็นคำกว้าง (ที่นี่ สถานที่ ทุกอย่าง โดยรวม) ให้ใส่ ""
- polarity = "positive" (ชม) หรือ "negative" (ติ/บ่น)

ถ้าข้อความไม่มีคำประเมิน แปลว่าไม่ใช่ความคิดเห็น ห้ามใส่ เช่น บอกแค่ว่ามีอะไร (มีสระน้ำ มีร้านค้า มีที่จอดรถ), ชื่ออาหาร/กิจกรรม,
การกระทำ (ขับรถผ่าน ช่วยกันทำความสะอาด ใช้เวลาทั้งวัน), ตำแหน่งที่ตั้ง (อยู่ริมถนน), เวลาเปิด-ปิด, ค่าเข้า,
คำแนะนำหรือคำสั่งถึงผู้อ่าน (เลือกร้านที่สะอาดนะ อย่าลืมแวะ) และความคิดเห็นต่อสถานที่อื่นที่ไม่ใช่ที่รีวิว
แต่ถ้า "มี X" ตามด้วยคำประเมินจำนวนหรือคุณภาพ ให้นับเป็นความคิดเห็น โดย aspect = X, opinion = คำประเมิน
  เช่น "มีของให้ซื้อมากมาย" -> aspect ของให้ซื้อ opinion มากมาย, "ลานจอดรถจอดได้หลายคัน" -> aspect ลานจอดรถ opinion จอดได้หลายคัน
opinion ต้องเป็นคำบอกคุณภาพ/ความรู้สึก ไม่ใช่กริยาการกระทำ เช่น ห้ามใช้ ทำความสะอาด ดูแลรักษา เดินขึ้นไป เป็น opinion
aspect และ opinion ต้องคัดลอกคำจากรีวิวตรงตัว ห้ามถอดความหรือเปลี่ยนเป็นคำอื่น แม้รีวิวจะสะกดผิด
  เช่น รีวิวเขียน "แม่ค้าเหยียดลูกค้า" ต้องตอบ opinion "เหยียดลูกค้า" ห้ามตอบ "เยาะเย้ย"
ตอบ {"opinions": []} เฉพาะเมื่อรีวิวไม่มีคำประเมินเลยจริงๆ

ตอบเป็น JSON เท่านั้น รูปแบบ:
{"opinions": [{"aspect": "วิว", "opinion": "สวย", "polarity": "positive"}]}"""


def _o(aspect, opinion, polarity="positive"):
    return {"aspect": aspect, "opinion": opinion, "polarity": polarity}


# ตัวอย่างแต่งขึ้นเอง (ไม่ใช้รีวิวจริงจากชุดข้อมูล)
FEW_SHOTS = [
    ("ที่นี่สวยงามมาก วิวดี อากาศดี เงียบสงบ",
     {"opinions": [_o("", "สวยงาม"), _o("วิว", "ดี"), _o("อากาศ", "ดี"), _o("", "เงียบสงบ")]}),
    ("อาหารอร่อย ราคาไม่แพง แต่ห้องน้ำไม่ค่อยสะอาด ที่จอดรถหายาก คนเยอะเกินไป",
     {"opinions": [_o("อาหาร", "อร่อย"), _o("ราคา", "ไม่แพง"), _o("ห้องน้ำ", "ไม่สะอาด", "negative"),
                   _o("ที่จอดรถ", "หายาก", "negative"), _o("คน", "เยอะ", "negative")]}),
    ("เป็นอุทยานที่น่ามาเที่ยวมาก ต้นไม้ร่มรื่นดี น้ำในลำธารใสมากๆ เจ้าหน้าที่ใจดียิ้มแย้ม มีโซนกางเต็นท์ ขับรถขึ้นไปได้ "
     "ส่วนตัวชอบข้าวผัดร้านหน้าอุทยาน แต่ทางเดินค่อนข้างรก มีกลิ่นเหม็นจากขยะ",
     {"opinions": [_o("", "น่ามาเที่ยว"), _o("ต้นไม้", "ร่มรื่น"), _o("น้ำ", "ใส"), _o("เจ้าหน้าที่", "ใจดี"),
                   _o("เจ้าหน้าที่", "ยิ้มแย้ม"), _o("ข้าวผัด", "ชอบ"), _o("ทางเดิน", "รก", "negative"),
                   _o("กลิ่น", "เหม็น", "negative")]}),
    ("ร้านขายของฝากคุณภาพดีและราคาประหยัด เหมาะกับครอบครัว ใช้เวลาเดินได้ทั้งวัน อยู่ติดถนนใหญ่",
     {"opinions": [_o("คุณภาพ", "ดี"), _o("ราคา", "ประหยัด"), _o("ครอบครัว", "เหมาะกับ")]}),
    ("เปิดทุกวัน 9.00-17.00 น. ค่าเข้าชม 50 บาท จอดรถได้ที่ลานด้านหน้า",
     {"opinions": []}),
    ("ในสวนมีสระน้ำ มีเรือปั่นให้เช่า มีของฝากให้เลือกมากมาย ลานจอดรถกว้าง จอดได้หลายคัน "
     "ถ้ามาช่วงเที่ยงแนะนำให้พกร่ม แม่ค้าบางร้านพูดจาไม่ดี",
     {"opinions": [_o("ของฝาก", "มากมาย"), _o("ลานจอดรถ", "กว้าง"), _o("ลานจอดรถ", "จอดได้หลายคัน"),
                   _o("แม่ค้า", "พูดจาไม่ดี", "negative")]}),
]

GENERIC_ASPECTS = {"ที่นี่", "สถานที่", "ทุกอย่าง", "โดยรวม", "ภาพรวม", "ที่", "นี่"}
VERB_FIRST = {"ชอบ", "ไม่ชอบ", "แนะนำ", "รัก", "เกลียด", "เหมาะกับ", "เหมาะสำหรับ", "เหมาะ"}
COMPACT_NORMALIZE = {"ความสวยงาม": "สวย", "สวยงาม": "สวย", "สะอาดสะอ้าน": "สะอาด", "ชิลล์": "ชิล",
                     "ความสงบ": "สงบ", "น่าเสียดาย": "เสียดาย", "กลิ่นเหม็น": "เหม็น", "ดีเยี่ยม": "ดี", "ดีมาก": "ดี"}
TRAILING_INTENSIFIERS = ["มากมาก", "มาก", "ที่สุด", "สุดสุด", "สุด", "เกินไป", "จริงจริง", "จริง", "ไปหน่อย", "หน่อย"]
NEGATORS = ["ไม่มี", "ไม่ค่อย", "ไม่ได้", "ไม่", "ไร้"]


def compose(aspect, opinion):
    aspect = re.sub(r"\s+", "", str(aspect or ""))
    opinion = re.sub(r"\s+", "", str(opinion or ""))
    if aspect in GENERIC_ASPECTS or aspect == opinion or aspect in opinion:
        aspect = ""
    if opinion in VERB_FIRST:
        return opinion + aspect
    if aspect and opinion and aspect.endswith(opinion):    # aspect "สวนร่มรื่น" + opinion "ร่มรื่น" -> ไม่ซ้ำคำ
        return aspect
    return aspect + opinion


def build_messages(text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for review, answer in FEW_SHOTS:
        msgs.append({"role": "user", "content": f'รีวิว: "{review}"'})
        msgs.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    msgs.append({"role": "user", "content": f'รีวิว: "{text[:MAX_CHARS]}"'})
    return msgs


def normalize_phrase(p):
    p = re.sub(r"\s+", "", str(p)).replace("ๆ", "")
    p = re.sub(r"[^฀-๿a-zA-Z0-9]", "", p)
    p = p.replace("ค่อนข้าง", "")
    changed = True
    while changed:                                   # สวยมากที่สุด -> สวย
        changed = False
        for w in TRAILING_INTENSIFIERS:
            if p.endswith(w) and len(p) > len(w) + 1:
                p, changed = p[: -len(w)], True
    p = p.replace("ไม่ค่อย", "ไม่").replace("ไม่ได้", "ไม่")
    for k, v in sorted(COMPACT_NORMALIZE.items(), key=lambda kv: -len(kv[0])):
        if p.endswith(k):
            p = p[: -len(k)] + v
            break
    return p


def _near(tok, flat):
    """คำนี้อยู่ในข้อความไหม โดยยอมให้ต่างกัน 1 ตัวอักษร (รีวิวสะกดผิด เช่น เสีบดาย) -- เฉพาะคำยาว >= 4 ตัว"""
    if tok in flat:
        return True
    n = len(tok)
    if n < 4:
        return False
    return any(sum(a != b for a, b in zip(tok, flat[i:i + n])) <= 1 for i in range(len(flat) - n + 1))


def is_grounded(phrase, text):
    """วลีต้องมาจากรีวิวจริง: ทุกคำ (ยกเว้นคำปฏิเสธ) ต้องพบในข้อความ"""
    flat = re.sub(r"\s+", "", str(text)).replace("ๆ", "")
    if phrase in flat:
        return True
    toks = [t for t in word_tokenize(phrase, engine="newmm") if t.strip() and t not in NEGATORS]
    return bool(toks) and all(_near(t, flat) for t in toks)


# วลีที่ไม่ใช่ความคิดเห็นแน่ๆ: บอกแค่ว่ามี (สระน้ำมี) หรือเหลือแต่คำเชื่อม/คำขยาย (ที่ นิด มากที่)
EXISTENCE_ONLY = {"มี", "มีให้", "มีบริการ"}
FUNCTION_WORDS = {"ที่", "นิด", "มาก", "มากที่", "สุด", "ที่สุด", "เลย", "จริง", "ก็", "แต่", "และ", "ได้"}


def is_opinion_like(opinion, phrase):
    o = re.sub(r"\s+", "", str(opinion or ""))
    return o not in EXISTENCE_ONLY and phrase not in FUNCTION_WORDS


def parse_response(raw):
    """คืนรายการ {phrase, polarity} หรือ None ถ้า JSON อ่านไม่ได้"""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    items = data.get("opinions", []) if isinstance(data, dict) else []
    out = []
    for it in items:
        if (isinstance(it, dict) and it.get("polarity") in ("positive", "negative")
                and str(it.get("opinion", "")).strip()):
            phrase = compose(it.get("aspect", ""), it["opinion"])
            out.append({"phrase": phrase, "polarity": it["polarity"],
                        "opinion_like": is_opinion_like(it["opinion"], phrase)})
    return out


def split_chunks(text, size=CHUNK_CHARS):
    text = str(text)[:MAX_CHARS]
    if len(text) <= size:
        return [text]
    parts = [p for p in re.split(r"(?:\s{2,}|\n|\.{2,}|(?<=[^\s])\s(?=[^\s]))", text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        if chunks and len(cur) < 80:
            chunks[-1] = f"{chunks[-1]} {cur}"
        else:
            chunks.append(cur)
    return chunks


def process(text, items):
    """normalize + grounding + ตัดวลียาว + ตัดซ้ำ (คงลำดับ) -> (kept, dropped)"""
    kept, dropped, seen = [], [], set()
    for it in items:
        p = normalize_phrase(it["phrase"])
        if not p or p in seen:
            continue
        seen.add(p)
        too_long = len([t for t in word_tokenize(p, engine="newmm") if t.strip()]) > MAX_TOKENS
        ok = (is_grounded(p, text) and not too_long and it.get("opinion_like", True)
              and p not in FUNCTION_WORDS)
        (kept if ok else dropped).append({"phrase": p, "polarity": it["polarity"]})
    return kept, dropped


class LLMPhraseExtractor:
    version = VERSION

    def __init__(self, api_key, model=None, timeout=15.0, transport=None):
        self.api_key, self.model, self.timeout, self.transport = api_key, model, timeout, transport

    def _ask(self, text):
        last = None
        for _ in range(MAX_RETRIES):
            try:
                raw = typhoon.chat(build_messages(text), self.api_key, self.model, timeout=self.timeout,
                                   max_tokens=1200, transport=self.transport)
            except Exception as e:                   # 429 / หมดเวลา / เครือข่าย -> ลองใหม่ 1 ครั้ง
                last = f"{type(e).__name__}: {e}"
                continue
            parsed = parse_response(raw)
            if parsed is not None:
                return parsed
            last = "invalid JSON"
        raise LLMExtractionError(last)

    def extract(self, text):
        """คืน (kept, dropped) -- raise LLMExtractionError ถ้า Typhoon ใช้ไม่ได้"""
        text = str(text or "")
        try:
            items = self._ask(text)
        except LLMExtractionError:
            if len(text) <= CHUNK_CHARS:
                raise
            items = []                               # JSON เสียกับรีวิวยาว -> ถามทีละช่วง
            for ch in split_chunks(text):
                items += self._ask(ch)
        return process(text, items)

    def analyze(self, text):
        kept, _ = self.extract(text)
        pos = [k["phrase"] for k in kept if k["polarity"] == "positive"]
        neg = [k["phrase"] for k in kept if k["polarity"] == "negative"]
        return {"sentiment_text": " ".join(k["phrase"] for k in kept) or None,
                "positive_text": " ".join(pos) or None,
                "negative_text": " ".join(neg) or None,
                "n_sentiment_terms": len(kept)}
