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
            if p.endswith(w) and len(p) > len(w) + 1 and not p[: -len(w)].endswith(("ไม่", "ไม่ค่อย")):
                p, changed = p[: -len(w)], True                # ไม่ตัด "มาก" ใน "ไม่มาก" (ความหมายกลับ)
    p = re.sub("ไม่ได้(?!$)", "ไม่", p.replace("ไม่ค่อย", "ไม่"))   # ไม่ได้สวย -> ไม่สวย แต่ "เข้าไม่ได้" คงไว้
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
    """v8.1: LLM ตอบ aspect/opinion แยกช่อง แล้วโค้ดประกอบวลี"""
    version = VERSION

    def __init__(self, api_key, model=None, timeout=15.0, transport=None):
        self.api_key, self.model, self.timeout, self.transport = api_key, model, timeout, transport

    # จุดที่รุ่นใหม่เปลี่ยนได้
    def build_messages(self, text):
        return build_messages(text)

    def parse(self, raw):
        return parse_response(raw)

    def process(self, text, items):
        return process(text, items)

    def _ask(self, text):
        last = None
        for _ in range(MAX_RETRIES):
            try:
                raw = typhoon.chat(self.build_messages(text), self.api_key, self.model, timeout=self.timeout,
                                   max_tokens=1200, transport=self.transport)
            except Exception as e:                   # 429 / หมดเวลา / เครือข่าย -> ลองใหม่ 1 ครั้ง
                last = f"{type(e).__name__}: {e}"
                continue
            parsed = self.parse(raw)
            if parsed is not None:
                return parsed
            last = "invalid JSON"
        raise LLMExtractionError(last)

    def extract(self, text):
        """คืน (kept, dropped) -- raise LLMExtractionError ถ้า Typhoon ใช้ไม่ได้"""
        text = str(text or "")
        return self.process(text, self.extract_items(text))

    def extract_items(self, text):
        """คำตอบดิบของ LLM (หลังอ่าน JSON) ก่อนตรวจ -- เก็บไว้เพื่อปรับกฎตรวจแล้ววัดผลซ้ำได้โดยไม่ต้องเรียก API"""
        text = str(text or "")
        try:
            items = self._ask(text)
        except LLMExtractionError:
            if len(text) <= CHUNK_CHARS:
                raise
            items = []                               # JSON เสียกับรีวิวยาว -> ถามทีละช่วง
            for ch in split_chunks(text):
                items += self._ask(ch)
        return items

    def analyze(self, text):
        kept, _ = self.extract(text)
        pos = [k["phrase"] for k in kept if k["polarity"] == "positive"]
        neg = [k["phrase"] for k in kept if k["polarity"] == "negative"]
        return {"sentiment_text": " ".join(k["phrase"] for k in kept) or None,
                "positive_text": " ".join(pos) or None,
                "negative_text": " ".join(neg) or None,
                "n_sentiment_terms": len(kept)}


# =====================================================================================
# v9: LLM คัดลอก "ข้อความ" จากรีวิวตรงตัว (span) แทนการแยก aspect/opinion
#   เหตุผล: v8.1 รันทั้งชุดแล้ว LLM ปล่อยช่อง aspect ว่างบ่อย ("บรรยากาศดี" -> "ดี" ใน 40% ของกรณี)
#   การคัดลอกช่วงข้อความทำให้หัวข้อที่อยู่ติดกันติดมาเองโดยธรรมชาติ
#   ช่อง aspect ใช้เฉพาะเมื่อหัวข้ออยู่ไกลจากคำประเมิน ("ห้องน้ำ...ค่อนข้างสกปรก")
#   โค้ดหลังจากนี้ทำแค่ "ตรวจ" (ข้อความต้องอยู่ในรีวิวจริง, ยาวไม่เกิน 6 คำ) และจัดรูปแบบ ไม่ได้เติมคำให้
# =====================================================================================
VERSION_V9 = "llm-typhoon-v9.3"
# v9.1 (หลังวัดผล v9 บนชุดทดสอบ 97 รีวิว): v9 แก้หัวข้อหายได้ (หัวข้อหาย 20 -> 2 วลี) แต่
#   - คัดลอกนามวลีที่ไม่มีคำประเมินมาด้วย (ผ้าทอเกาะยอ, ค่าเข้าชม50บาท, มีลิง) -> เพิ่มช่อง opinion บังคับ
#   - ติดคำลงท้าย (ดีค่ะ, สวยงามมากครับ, น่าเที่ยวมากคร้า) -> ตัดคำลงท้ายตอนจัดรูปแบบ
#   - คัดลอกทั้งประโยค ("น้ำตกชั้น3ทางขึ้นชันและอันตราย") -> ตัวอย่างการแยกเป็นหลายข้อ
#   - "วัดที่สวย" "ประสบการณ์ที่ดี" -> ใช้ช่อง aspect เมื่อมี "ที่" คั่น
#   - ตัวอักษรล่องหน (\u200b) ในรีวิวทำให้ตรวจไม่ผ่าน -> ลบก่อนตรวจ
# v9.2 (หลังวัดผล v9.1: ถูกปฏิเสธ 15.4% -> 5.2%, เก็บหัวข้อ 90.9%) อ่านทีละรีวิวแล้วพบว่า
#   ความเห็นจริงยังหายเพราะถูก "ทิ้งทั้งข้อ" เมื่อ text ยาวเกิน/ไม่ตรงตัว ซึ่งเป็นคำติที่สำคัญหลายข้อ
#     "คุณภาพน้ำอาจจะไม่ใสสะอาดนัก" "รสชาติอาหารอยู่ในระดับปานกลาง" "อาหารก็อร่อย" (LLM ตัด "ก็" ทิ้ง)
#   -> ช่อง aspect ใส่ทุกครั้ง ถ้า text ใช้ไม่ได้ ใช้ aspect+opinion แทน (ทั้งสองต้องอยู่ในรีวิวจริง) = "กู้คืน"
#   -> นับความยาวโดยให้หัวข้อนับเป็น 1 คำ ("อาหารริมทาง" = 1 หน่วย ไม่ใช่ 3 คำ)
#   - วลีขาดท้าย ("ลดการใช้พลาสติกได้อย่าง") -> ปฏิเสธวลีที่จบด้วยคำเชื่อม
#   - "ประสบการณ์ที่ดี" "ราคาที่คุ้มค่า" -> ตัด "ที่" เฉพาะเมื่อคั่นระหว่างหัวข้อกับคำประเมินพอดี
#   - หัวข้อที่อยู่ไกล ("ร้านโจ๊กหมูอร่อยๆ" -> "อร่อย") -> ช่อง aspect ใส่ทุกครั้งช่วยให้ได้ "โจ๊กหมูอร่อย"
# v9.3 (หลังวัดผล v9.2: เก็บหัวข้อ 97.5%, ถูกปฏิเสธ 2.7%, มีวลี 91.8%) แก้เฉพาะกฎในโค้ด ตรวจกับคำตอบดิบ 97 รีวิวแล้ว
#   - หัวข้อซ้ำกับข้อความ: "การเริ่มต้น"+"เริ่มต้นสมบูรณ์แบบ", "เวลาจัด"+"จัดได้ลงตัว" -> ไม่เติมหัวข้อเมื่อมีคำซ้ำกัน
#   - คำประเมินไม่ใช่คำประเมิน: opinion = หัวข้อเอง ("บรรยากาศ"), "มี", "ที่สุด", "อย่างมาก" -> ไม่ใช่ความคิดเห็น
#   - คำแนะนำถึงผู้อ่าน: "แนะนำให้...", "ไม่แนะนำให้วางของ...", "อย่าลืม..." -> ไม่ใช่ความคิดเห็น (ตามนิยามที่ทีมเลือก)
#   - กู้คืนเฉพาะเมื่อคำประเมินสั้น (<= 3 คำ) และไม่ใช่กริยาลอยๆ (เหมาะ ชอบ) -> กัน "เหมาะน้ำตกชั้น1"
#   - "เข้าไม่ได้" ถูกตัดเป็น "เข้าไม่" / "ไม่ค่อยมี" ถูกปฏิเสธเพราะจบด้วย "มี" -> แก้
#   - ตัด "ก็" (ถ่ายรูปก็สวย -> ถ่ายรูปสวย) และ "เป็น" หน้าหัวข้อ (เป็นประสบการณ์ดี -> ประสบการณ์ดี)
#   - LLM ตอบว่างกับรีวิวยาว (เล่าตำนานยาวแล้วแทรกความเห็น) -> ถามใหม่ทีละช่วง

SYSTEM_PROMPT_V9 = """คุณดึง "ความคิดเห็น" จากรีวิวสถานที่ท่องเที่ยวภาษาไทย ให้ครบทุกจุด ตอบเป็น JSON เท่านั้น

แต่ละความคิดเห็นมี 4 ช่อง (คัดลอกจากรีวิวตรงตัวทุกตัวอักษร ห้ามแก้คำ ห้ามตัดคำกลางข้อความ แม้สะกดผิด)
- text     = ข้อความสั้น 1-6 คำ ที่มีคำประเมิน และถ้าสิ่งที่ถูกประเมินอยู่ติดกัน ต้องคัดลอกมาด้วย
             เช่น "บรรยากาศดี" "ห้องน้ำไม่ค่อยสะอาด" "ราคาถูก" "ของให้เลือกมากมาย"
- opinion  = คำประเมินที่อยู่ใน text รวมคำปฏิเสธ เช่น ดี สวยงาม ไม่ค่อยสะอาด ไม่ใส ถูก มากมาย แพง แออัด ชอบ
             ถ้าหาคำประเมินไม่ได้ แปลว่าไม่ใช่ความคิดเห็น ห้ามใส่ข้อนั้น
- aspect   = สิ่งที่ถูกประเมิน (คำนามสั้นๆ จากรีวิว) ใส่ทุกครั้ง ไม่ว่าจะอยู่ใน text แล้วหรืออยู่ห่างออกไป
             ใส่ "" เฉพาะเมื่อชมหรือติสถานที่โดยรวม (ที่นี่ สถานที่นี้)
- polarity = "positive" (ชม) หรือ "negative" (ติ/บ่น)

กฎ
1. ห้ามตัดคำที่อยู่ติดหน้าคำประเมินทิ้ง: รีวิว "บรรยากาศดี" ต้องตอบ text "บรรยากาศดี" ห้ามตอบแค่ "ดี"
2. หนึ่งข้อต่อหนึ่งความคิดเห็น: "ทางขึ้นชันและอันตราย" = 2 ข้อ ("ทางขึ้นชัน" และ "อันตราย" โดย aspect "ทางขึ้น")
3. ไม่ใช่ความคิดเห็น ห้ามใส่: ชื่อสิ่งของ/อาหาร/สินค้า/กิจกรรมที่ไม่มีคำประเมิน, บอกแค่ว่ามีอะไร (มีสระน้ำ มีลิง มีอาหารฮาลาล),
   ราคาหรือเวลาเป็นตัวเลข, คำแนะนำหรือคำชวนถึงผู้อ่าน (อย่าลืมแวะ เลือกร้านที่สะอาดนะ), ความเห็นต่อสถานที่อื่น
4. "มี X" + คำประเมินจำนวนหรือคุณภาพ เป็นความคิดเห็น: "ของให้เลือกมากมาย" "จอดได้หลายคัน"
5. คำติสำคัญเท่ากับคำชม ต้องเก็บให้ครบ แม้จะพูดอ้อมๆ (อาจจะไม่ค่อยสะอาด อยู่ในระดับธรรมดา)
ถ้ารีวิวไม่มีความคิดเห็นเลย ตอบ {"opinions": []}

รูปแบบ: {"opinions": [{"text": "บรรยากาศดี", "opinion": "ดี", "aspect": "บรรยากาศ", "polarity": "positive"}]}"""


def _s(text, opinion, polarity="positive", aspect=""):
    return {"text": text, "opinion": opinion, "aspect": aspect, "polarity": polarity}


# ตัวอย่างแต่งขึ้นเอง (ไม่ใช้รีวิวจริงจากชุดข้อมูล) ครอบคลุมจุดที่ v8.1/v9/v9.1 พลาด
FEW_SHOTS_V9 = [
    ("บรรยากาศดีมากค่ะ",
     {"opinions": [_s("บรรยากาศดี", "ดี", aspect="บรรยากาศ")]}),
    ("สะอาด สวยงาม เป็นวัดที่เงียบสงบมากกก",
     {"opinions": [_s("สะอาด", "สะอาด"), _s("สวยงาม", "สวยงาม"), _s("เงียบสงบ", "เงียบสงบ", aspect="วัด")]}),
    ("อาหารอร่อย ราคาไม่แพง แต่ห้องน้ำไม่ค่อยสะอาด ที่จอดรถหายาก คนเยอะเกินไป",
     {"opinions": [_s("อาหารอร่อย", "อร่อย", aspect="อาหาร"), _s("ราคาไม่แพง", "ไม่แพง", aspect="ราคา"),
                   _s("ห้องน้ำไม่ค่อยสะอาด", "ไม่ค่อยสะอาด", "negative", "ห้องน้ำ"),
                   _s("ที่จอดรถหายาก", "หายาก", "negative", "ที่จอดรถ"),
                   _s("คนเยอะเกินไป", "เยอะเกินไป", "negative", "คน")]}),
    ("ห้องน้ำที่อยู่ด้านหลังอาคารใหญ่ค่อนข้างสกปรก ทางขึ้นจุดชมวิวชันและอันตราย ส่วนเจ้าหน้าที่ยิ้มแย้มดี "
     "เดินเล่นแถวนั้นเจอร้านข้าวผัดอร่อยๆ",
     {"opinions": [_s("ค่อนข้างสกปรก", "สกปรก", "negative", "ห้องน้ำ"), _s("ชัน", "ชัน", "negative", "ทางขึ้น"),
                   _s("อันตราย", "อันตราย", "negative", "ทางขึ้น"),
                   _s("เจ้าหน้าที่ยิ้มแย้มดี", "ยิ้มแย้มดี", aspect="เจ้าหน้าที่"),
                   _s("ข้าวผัดอร่อย", "อร่อย", aspect="ข้าวผัด")]}),
    ("ห้องน้ำก็สะอาดดี ขนมพื้นเมืองมีให้เลือกเยอะมาก แต่รสชาติก๋วยเตี๋ยวอยู่ในระดับธรรมดา "
     "น้ำทะเลอาจจะไม่ค่อยใสนัก เป็นประสบการณ์ที่ดีค่ะ",
     {"opinions": [_s("สะอาดดี", "สะอาดดี", aspect="ห้องน้ำ"),
                   _s("ขนมพื้นเมืองมีให้เลือกเยอะ", "มีให้เลือกเยอะ", aspect="ขนมพื้นเมือง"),
                   _s("ธรรมดา", "ธรรมดา", "negative", "รสชาติก๋วยเตี๋ยว"),
                   _s("ไม่ค่อยใส", "ไม่ค่อยใส", "negative", "น้ำทะเล"),
                   _s("ประสบการณ์ที่ดี", "ดี", aspect="ประสบการณ์")]}),
    ("ในสวนมีสระน้ำ มีเรือปั่นให้เช่า มีของฝากให้เลือกมากมาย ลานจอดรถกว้าง จอดได้หลายคัน มีขนมพื้นบ้าน ผ้าทอมือ "
     "มีอาหารฮาลาล ค่าเข้า 30 บาท ถ้ามาช่วงเที่ยงแนะนำให้พกร่ม อย่าลืมแวะร้านกาแฟหน้าทางเข้านะคะ",
     {"opinions": [_s("ของฝากให้เลือกมากมาย", "มากมาย", aspect="ของฝาก"),
                   _s("ลานจอดรถกว้าง", "กว้าง", aspect="ลานจอดรถ"),
                   _s("จอดได้หลายคัน", "จอดได้หลายคัน", aspect="ลานจอดรถ")]}),
    ("เปิดทุกวัน 9.00-17.00 น. ค่าเข้าชม 50 บาท มีร้านค้าและห้องน้ำบริการ",
     {"opinions": []}),
]


def build_messages_v9(text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_V9}]
    for review, answer in FEW_SHOTS_V9:
        msgs.append({"role": "user", "content": f'รีวิว: "{review}"'})
        msgs.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    msgs.append({"role": "user", "content": f'รีวิว: "{text[:MAX_CHARS]}"'})
    return msgs


ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u2028\u2029]")


def _flat(s):
    return re.sub(r"\s+", "", ZERO_WIDTH.sub("", str(s or ""))).replace("ๆ", "")


# คำลงท้าย/คำเสริมที่ไม่มีเนื้อหา ตัดออกตอนจัดรูปแบบ (ดีค่ะ -> ดี, สวยงามมากครับ -> สวย)
# ตรวจกับวลีจริง 7,159 คำจากการรันทั้งชุดแล้ว: ไม่ใส่ "ค่า" (คุ้มค่า), "นัก" (หนัก), "มากนัก" (ไม่มากนัก),
# "เหมือนกัน" (ของขายเหมือนกัน = ติ) และไม่ตัดเมื่อเป็นส่วนของคำ (ละเลย, เต็มไปด้วย)
TRAILING_PARTICLES = ["นะคะ", "นะครับ", "ค่ะ", "คะ", "ครับ", "คับ", "ครัฟ", "จ้า", "จ้ะ", "จร้า", "คร้า", "นะ",
                      "เลย", "ด้วย", "ทีเดียว", "แล้ว"]
PROTECTED_ENDINGS = ("ละเลย", "เฉยเลย", "ไปด้วย", "ประกอบด้วย", "ช่วยด้วย", "หายนะ", "ชนะ", "ภาชนะ", "พาหนะ", "สว่างจ้า")
THAI_MARKS = "ะาำิีึืุูเแโใไๅ่้๊๋็์ั"
# วลีที่จบด้วยคำเหล่านี้ = ประโยคขาดกลางทาง ("ลดการใช้พลาสติกได้อย่าง")
DANGLING = {"อย่าง", "และ", "แต่", "ที่", "ของ", "ให้", "เป็น", "ใน", "กับ", "ว่า", "จะ", "ซึ่ง", "โดย", "หรือ",
            "ก็", "ความ", "การ", "เพราะ", "ถ้า", "มี"}
# "ที่" หน้าคำประเมิน ไม่ตัดเมื่อคำก่อนหน้าทำให้ "ที่" มีความหมายอื่น (อยู่ในที่ร่ม, ตรงกับที่คาดไว้, สิ่งที่ดี)
KEEP_THI_AFTER = {"ใน", "กับ", "ไป", "มี", "ไม่มี", "อยู่", "ตรง", "สิ่ง", "ผู้", "คน", "ชั้น", "ด่าน", "หา"}
ADVICE_STARTS = ("แนะนำให้", "ไม่แนะนำให้", "อย่าลืม", "ขอแนะนำให้")
NOT_OPINIONS = {"มี", "เป็น", "อย่างมาก", "ได้อย่างมาก", "ที่สุด", "เต็มไปด้วย"}
ASPECT_MAX_CHARS = 20     # หัวข้อยาวกว่านี้ไม่นำมาประกอบ ("ประสบการณ์การขับรถเที่ยวสวนสัตว์")
PHRASE_MAX_CHARS = 40


def parse_response_v9(raw):
    """คืนรายการ {span, opinion, aspect, polarity} หรือ None ถ้า JSON อ่านไม่ได้
    (ข้อที่ polarity ผิดรูปแบบยังคืนมา ให้ process ทิ้งพร้อมบอกเหตุผล -- ดูย้อนหลังได้)"""
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
        if isinstance(it, dict) and _flat(it.get("text")):
            out.append({"span": _flat(it["text"]), "opinion": _flat(it.get("opinion")),
                        "aspect": _flat(it.get("aspect")), "polarity": str(it.get("polarity") or "")})
    return out


def normalize_v9(p):
    """จัดรูปแบบเท่านั้น: ยุบตัวอักษรซ้ำ (มากกก -> มาก), ตัดคำลงท้าย (ค่ะ ครับ เลย ด้วย),
    แล้วใช้กฎเดียวกับ v8 (ตัดคำขยายท้าย, สวยงาม -> สวย)"""
    p = _flat(p)
    p = re.sub(f"([{THAI_MARKS}])\\1{{2,}}", r"\1", p)        # สระ/วรรณยุกต์ซ้ำ: เยอะะะ -> เยอะ
    p = re.sub(r"(.)\1{3,}", r"\1", p)                        # ซ้ำ 4 ตัวขึ้นไป: สุดดดดด -> สุด (ทางงง/ระบบบริการ ไม่โดน)
    p = re.sub(r"มากก+$", "มาก", re.sub(r"สุดด+$", "สุด", p))  # มากกก -> มาก (ท้ายวลี)
    changed = True
    while changed:
        changed = False
        for w in TRAILING_PARTICLES:
            if p.endswith(w) and len(p) > len(w) + 1 and not p.endswith(PROTECTED_ENDINGS):
                p, changed = p[: -len(w)], True
        for w in TRAILING_INTENSIFIERS:
            if p.endswith(w) and len(p) > len(w) + 1 and not p[: -len(w)].endswith(("ไม่", "ไม่ค่อย")):
                p, changed = p[: -len(w)], True
    if "ก็" in p:                                              # ถ่ายรูปก็สวย -> ถ่ายรูปสวย (ตัดเฉพาะคำ "ก็" ไม่โดน "เก็บ")
        rest = "".join(t[2:] if t.startswith("ก็") and len(t) > 2 else t
                       for t in word_tokenize(p, engine="newmm") if t != "ก็")
        p = rest or p
    return normalize_phrase(p)


def span_in_review(span, flat):
    """ข้อความต้องอยู่ในรีวิวจริง (ยอมต่าง 1 ตัวอักษรสำหรับข้อความยาว >= 4 ตัว)"""
    return _near(span, flat)


def _toks(p):
    return [t for t in word_tokenize(p, engine="newmm") if t.strip()]


_THAI_WORDS = None


def _thai_words():
    global _THAI_WORDS
    if _THAI_WORDS is None:
        from pythainlp.corpus import thai_words
        _THAI_WORDS = set(thai_words())
    return _THAI_WORDS


def drop_thi(p, opinion):
    """"ประสบการณ์ที่ดี" -> "ประสบการณ์ดี": ตัด "ที่" เฉพาะเมื่อตามด้วยคำประเมินจนจบวลีพอดี"""
    if not opinion:
        return p
    for o in {opinion, normalize_v9(opinion)}:
        if o and p.endswith("ที่" + o):
            head = p[: -len("ที่" + o)]
            ht = _toks(head)
            # "ที่" ต้องไม่ใช่ท้ายคำ: คำสุดท้าย+ที่ เป็นคำในพจนานุกรม (สถาน+ที่, พื้น+ที่, หน้า+ที่) -> ไม่ตัด
            if ht and ht[-1] not in KEEP_THI_AFTER and (ht[-1] + "ที่") not in _thai_words():
                return head + o
    return p


def units(p, aspect):
    """ความยาววลีเป็น "คำ" โดยนับหัวข้อเป็น 1 หน่วย ("อาหารริมทางมีให้เลือกมากมาย" = 1 + 4)"""
    if aspect and aspect in p:
        rest = p.replace(aspect, "", 1)
        return 1 + len(_toks(rest)) if rest else 1
    return len(_toks(p))


def _overlap(a, p):
    """หัวข้อกับข้อความมีคำเดียวกัน (ไม่นับ การ/ความ) -> ไม่ต้องเติมหัวข้อ ("การเริ่มต้น" + "เริ่มต้นสมบูรณ์แบบ")"""
    skip = {"การ", "ความ", "ที่", "ของ", "ใน"}
    ta = {t for t in _toks(a) if len(t) >= 2 and t not in skip}
    return bool(ta & {t for t in _toks(p) if len(t) >= 2})


def _join(a, o):
    if not a or _overlap(a, o):
        return o
    if o in VERB_FIRST or any(o.startswith(v) for v in ("ชอบ", "ไม่ชอบ", "แนะนำ")):
        return o + a
    return a if a.endswith(o) else (o if o.startswith(a) else a + o)


def _bad(p, aspect):
    tk = _toks(p)
    return (not p or p in FUNCTION_WORDS or not tk or (tk[-1] in DANGLING and not p.endswith("ไม่มี"))
            or units(p, aspect) > MAX_TOKENS or len(p) > PHRASE_MAX_CHARS)


def process_v9(text, items):
    """ตรวจ + จัดรูปแบบ -> (kept, dropped)  ทุกข้อมี how (span/กู้คืน) หรือ reason (เหตุผลที่ทิ้ง)

    1) ใช้ text ของ LLM ถ้าอยู่ในรีวิวจริง สั้นพอ และมีคำประเมิน (เติมหัวข้อหน้าถ้าหัวข้ออยู่ไกล)
    2) ถ้า text ใช้ไม่ได้ (ยาวเกิน / ไม่ตรงตัว) แต่ opinion และ aspect อยู่ในรีวิวจริง -> ใช้ aspect+opinion ("กู้คืน")
    3) ไม่มีคำประเมิน หรือ polarity ผิด -> ทิ้ง (ไม่ใช่ความคิดเห็น)"""
    flat = _flat(text)
    kept, dropped, seen = [], [], set()

    def drop(phrase, pol, reason):
        dropped.append({"phrase": phrase, "polarity": pol, "reason": reason})

    for it in items:
        span, aspect, pol = it["span"], it.get("aspect", ""), it.get("polarity", "")
        opinion = it.get("opinion", "")
        if pol not in ("positive", "negative"):
            drop(span, pol, "polarity"); continue
        if (not opinion or not (opinion in span or _near(opinion, span) or _near(opinion, flat))
                or normalize_v9(opinion) in FUNCTION_WORDS or opinion in NOT_OPINIONS
                or (aspect and (opinion == aspect or opinion in aspect))):
            drop(span, pol, "ไม่มีคำประเมิน"); continue
        if span.startswith(ADVICE_STARTS):
            drop(span, pol, "คำแนะนำ"); continue
        a = normalize_v9(aspect) if (aspect and aspect not in GENERIC_ASPECTS and len(aspect) <= ASPECT_MAX_CHARS
                                     and _near(aspect, flat)) else ""
        a = "" if a in GENERIC_ASPECTS else a
        p, how = drop_thi(normalize_v9(span), opinion), "span"
        if (a and p.startswith("เป็น" + a)) or p.startswith("เป็นสถานที่"):   # เป็นประสบการณ์ดี / เป็นสถานที่ดี
            p = p[len("เป็น"):]
        if a and aspect not in span:
            p = _join(a, p)
        tk = _toks(p)
        if tk and tk[-1] in DANGLING and not p.endswith("ไม่มี"):   # ข้อความขาดกลางประโยค = LLM คัดลอกพัง ไม่กู้คืน
            drop(p, pol, "ขาดท้าย"); continue
        opinion_in_span = opinion in span or _near(opinion, span)
        if not span_in_review(span, flat) or not opinion_in_span or _bad(p, a):
            reason = ("ไม่อยู่ในรีวิว" if not span_in_review(span, flat) else
                      "คำประเมินไม่อยู่ใน text" if not opinion_in_span else "ยาวเกิน")
            o = normalize_v9(opinion)
            p, how = _join(a, o), "กู้คืน"
            if (not _near(opinion, flat) or _bad(p, a) or len(_toks(o)) > 3
                    or o in VERB_FIRST or o in ("เหมาะ", "ไม่เหมาะ")):
                drop(normalize_v9(span), pol, reason); continue
        if p in seen:
            continue
        seen.add(p)
        kept.append({"phrase": p, "polarity": pol, "how": how, "aspect": a})
    return kept, dropped


class LLMPhraseExtractorV9(LLMPhraseExtractor):
    version = VERSION_V9

    def build_messages(self, text):
        return build_messages_v9(text)

    def parse(self, raw):
        return parse_response_v9(raw)

    def process(self, text, items):
        return process_v9(text, items)

    def extract_items(self, text):
        items = super().extract_items(text)
        text = str(text or "")
        if not items and len(text) > CHUNK_CHARS:            # รีวิวยาวที่ LLM ตอบว่าง -> ถามทีละช่วง
            for ch in split_chunks(text):
                items += self._ask(ch)
        return items


PROMPTS = {"v8.1": LLMPhraseExtractor, "v9": LLMPhraseExtractorV9}


def make_extractor(prompt, api_key, model=None, **kw):
    if prompt not in PROMPTS:
        raise ValueError(f"LLM prompt must be one of {list(PROMPTS)}, got {prompt!r}")
    return PROMPTS[prompt](api_key, model, **kw)
