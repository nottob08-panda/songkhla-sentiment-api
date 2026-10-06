"""ตรวจภาษาและแปลเป็นภาษาไทย

โมเดลเทรนจากข้อความภาษาไทย (รีวิวต่างภาษาใช้คำแปลของ Google) รีวิวใหม่จึงต้องผ่านขั้นตอนเดียวกัน
- ภาษาไทย  -> ใช้ข้อความเดิม
- ภาษาอื่น -> แปลเป็นไทย ลองตามลำดับ:
    1. Typhoon API (ต้องมี API key ฟรีจาก opentyphoon.ai) -- นับโควตาต่อ key จึงไม่ติดปัญหา IP ร่วม
    2. Google (deep-translator, ไม่ต้องใช้ key)       -- นับโควตาต่อ IP มักถูกบล็อกบนโฮสต์ฟรี
    3. MyMemory (ไม่ต้องใช้ key)                     -- นับโควตาต่อ IP เช่นกัน
ถ้าแปลไม่สำเร็จทุกตัว จะ raise TranslationError แล้วรีวิวจะถูกบันทึกเป็น failed เพื่อลองใหม่ภายหลัง
"""
import re

import httpx
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0          # ให้ผลตรวจภาษาเหมือนเดิมทุกครั้ง

THAI_CHAR = re.compile(r"[ก-๙]")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# รหัสภาษาของ langdetect -> รูปแบบเดียวกับข้อมูลเดิมจาก Google Maps (เช่น zh-Hans)
LANG_ALIASES = {"zh-cn": "zh-Hans", "zh-tw": "zh-Hant"}
MYMEMORY_MAX = 450                # MyMemory รับไม่เกินประมาณ 500 bytes ต่อครั้ง
TYPHOON_URL = "https://api.opentyphoon.ai/v1/chat/completions"
TYPHOON_DEFAULT_MODEL = "typhoon-v2.5-30b-a3b-instruct"
TYPHOON_PROMPT = ("You are a professional translator. Translate the user's tourist review into natural Thai. "
                  "Keep the meaning and the sentiment exactly as written. "
                  "Output only the Thai translation, without quotes, notes, or explanations.")


class TranslationError(Exception):
    pass


def has_letters(text):
    return bool(LETTER.search(text or ""))


def detect_language(text):
    """คืนรหัสภาษา เช่น 'th', 'en', 'zh-Hans' หรือ None ถ้าไม่มีตัวอักษรเลย (เช่น อีโมจิล้วน)"""
    letters = LETTER.findall(text or "")
    if not letters:
        return None
    if sum(bool(THAI_CHAR.match(c)) for c in letters) / len(letters) >= 0.5:
        return "th"
    try:
        code = detect(text)
    except Exception:
        return "und"
    return LANG_ALIASES.get(code, code)


def _typhoon(text, timeout, api_key, model=None, transport=None):
    with httpx.Client(timeout=timeout, transport=transport) as client:
        r = client.post(TYPHOON_URL, headers={"Authorization": f"Bearer {api_key}"}, json={
            "model": model or TYPHOON_DEFAULT_MODEL,
            "messages": [{"role": "system", "content": TYPHOON_PROMPT},
                         {"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": min(4096, 64 + 3 * len(text)),
        })
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"] or ""
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().strip('"“”').strip()
    if not THAI_CHAR.search(out):                # กันกรณีโมเดลไม่ได้ตอบเป็นภาษาไทย
        raise TranslationError(f"Typhoon returned non-Thai text: {out[:80]!r}")
    return out


def _google(text, timeout):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="auto", target="th", timeout=timeout).translate(text)


def _chunks(text, size):
    parts, buf = [], ""
    for piece in re.split(r"(?<=[.!?。！？\n])\s*", text):
        while len(piece) > size:                 # ประโยคที่ยาวเกิน ตัดตามช่องว่าง
            cut = piece.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            parts.append(piece[:cut]); piece = piece[cut:].lstrip()
        if len(buf) + len(piece) + 1 > size:
            parts.append(buf); buf = piece
        else:
            buf = (buf + " " + piece).strip()
    if buf:
        parts.append(buf)
    return [p for p in parts if p]


def _mymemory(text, source_lang, timeout, email=None):
    src = (source_lang or "en").split("-")[0] if source_lang not in ("zh-Hans", "zh-Hant") else \
        {"zh-Hans": "zh-CN", "zh-Hant": "zh-TW"}[source_lang]
    out = []
    with httpx.Client(timeout=timeout) as client:
        for chunk in _chunks(text, MYMEMORY_MAX):
            params = {"q": chunk, "langpair": f"{src}|th"}
            if email:
                params["de"] = email            # ใส่อีเมลเพื่อเพิ่มโควตาฟรีเป็น 50,000 ตัวอักษร/วัน
            r = client.get("https://api.mymemory.translated.net/get", params=params)
            r.raise_for_status()
            data = r.json()
            if str(data.get("responseStatus")) != "200":
                raise TranslationError(f"MyMemory: {data.get('responseDetails')}")
            out.append(data["responseData"]["translatedText"])
    return " ".join(out)


def translate_to_thai(text, source_lang=None, timeout=4.0, mymemory_email=None,
                      typhoon_api_key=None, typhoon_model=None):
    """คืน (คำแปล, ผู้ให้บริการที่ใช้)"""
    errors = []
    providers = []
    if typhoon_api_key:                          # LLM ใช้เวลาตอบนานกว่า จึงให้เวลารอมากกว่า
        providers.append(("typhoon", lambda: _typhoon(text, max(timeout, 10.0), typhoon_api_key, typhoon_model)))
    providers += [("google", lambda: _google(text, timeout)),
                  ("mymemory", lambda: _mymemory(text, source_lang, timeout, mymemory_email))]
    for name, fn in providers:
        try:
            result = fn()
            if result and result.strip():
                return result.strip(), name
            errors.append(f"{name}: empty result")
        except Exception as e:                   # บริการฟรีอาจล่ม/จำกัดโควตาได้ทุกเมื่อ
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise TranslationError(" | ".join(errors))
