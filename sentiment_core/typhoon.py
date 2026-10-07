"""เรียก Typhoon API (รูปแบบเดียวกับ OpenAI chat completions) ใช้ร่วมกันทั้งการแปลภาษาและการสกัดวลี

มีตัวคุมจังหวะ (throttle) กลาง: โควตา Typhoon นับต่อ API key (5 ครั้ง/วินาที, 200 ครั้ง/นาที)
การแปลและการสกัดวลีใช้ key เดียวกัน จึงต้องเว้นระยะร่วมกัน ค่าเริ่มต้น 0.35 วินาที (~170 ครั้ง/นาที)
"""
import os
import re
import threading
import time

import httpx

TYPHOON_URL = "https://api.opentyphoon.ai/v1/chat/completions"
TYPHOON_DEFAULT_MODEL = "typhoon-v2.5-30b-a3b-instruct"
MIN_INTERVAL = float(os.getenv("TYPHOON_MIN_INTERVAL", "0.35"))

_lock = threading.Lock()
_last_call = 0.0


def _throttle():
    global _last_call
    with _lock:
        wait = _last_call + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def chat(messages, api_key, model=None, timeout=10.0, max_tokens=1024, transport=None):
    """ส่งข้อความไป Typhoon แล้วคืนคำตอบ (ตัดส่วน <think> ออกแล้ว) -- error ของ HTTP จะ raise ออกไป"""
    _throttle()
    with httpx.Client(timeout=timeout, transport=transport) as client:
        r = client.post(TYPHOON_URL, headers={"Authorization": f"Bearer {api_key}"}, json={
            "model": model or TYPHOON_DEFAULT_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        })
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
