"""ตรวจภาษาและแปลเป็นภาษาไทย

โมเดลเทรนจากข้อความภาษาไทย (รีวิวต่างภาษาใช้คำแปลของ Google) รีวิวใหม่จึงต้องผ่านขั้นตอนเดียวกัน
- ภาษาไทย  -> ใช้ข้อความเดิม
- ภาษาอื่น -> แปลเป็นไทย: Google (deep-translator) เป็นหลัก ถ้าไม่สำเร็จใช้ MyMemory สำรอง
ถ้าแปลไม่สำเร็จทั้งคู่ จะ raise TranslationError แล้วรีวิวจะถูกบันทึกเป็น failed เพื่อลองใหม่ภายหลัง
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


def translate_to_thai(text, source_lang=None, timeout=4.0, mymemory_email=None):
    """คืน (คำแปล, ผู้ให้บริการที่ใช้)"""
    errors = []
    for name, fn in (("google", lambda: _google(text, timeout)),
                     ("mymemory", lambda: _mymemory(text, source_lang, timeout, mymemory_email))):
        try:
            result = fn()
            if result and result.strip():
                return result.strip(), name
            errors.append(f"{name}: empty result")
        except Exception as e:                   # บริการฟรีอาจล่ม/จำกัดโควตาได้ทุกเมื่อ
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise TranslationError(" | ".join(errors))
