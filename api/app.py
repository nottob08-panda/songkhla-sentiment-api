"""Sentiment API สำหรับเว็บไซต์

Endpoints
  GET  /health                 ตรวจสถานะ + รุ่นโมเดล (ใช้ ping กันเซิร์ฟเวอร์หลับ)
  POST /v1/analyze             วิเคราะห์ข้อความ คืนผลโดยไม่บันทึกฐานข้อมูล
  POST /v1/webhook/reviews     รับ Database Webhook จาก Supabase เมื่อมีรีวิวใหม่ แล้วเขียนผลกลับ
  POST /v1/process-pending     วิเคราะห์รีวิวที่ค้างสถานะ pending/failed (เรียกจาก GitHub Actions)

ตัวแปรสภาพแวดล้อม (ตั้งใน Render > Environment)
  SUPABASE_URL                 เช่น https://xxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY    service_role key (เป็นความลับ ห้ามใส่ในโค้ดเว็บ)
  WEBHOOK_SECRET               รหัสลับที่ Supabase / GitHub Actions ต้องส่งมาใน header x-webhook-secret
  MODEL_DIR                    โฟลเดอร์ไฟล์โมเดล (ค่าเริ่มต้น model)
  MODEL_VERSION                ชื่อรุ่นโมเดล (ค่าเริ่มต้น nb-v1)
  MYMEMORY_EMAIL               (ไม่บังคับ) เพิ่มโควตาแปลภาษาสำรอง
"""
import hmac
import json
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from sentiment_core.pipeline import ReviewAnalyzer

log = logging.getLogger("sentiment-api")
logging.basicConfig(level=logging.INFO)

BASE = Path(__file__).resolve().parent.parent
COLUMN_MAP = json.loads((Path(__file__).parent / "column_map.json").read_text(encoding="utf-8"))
MAX_TEXT_LEN = 5000


# ---------------------------------------------------------------- ฐานข้อมูล
class SupabaseRepo:
    """อ่าน/เขียนตาราง reviews ผ่าน Supabase REST API (PostgREST)"""

    def __init__(self, url, key, transport=None):
        self.base = f"{url.rstrip('/')}/rest/v1/{COLUMN_MAP['table']}"
        self.client = httpx.Client(
            timeout=10.0, transport=transport,
            headers={"apikey": key, "Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})

    def fetch_pending(self, limit):
        idc, txt, st = COLUMN_MAP["id_column"], COLUMN_MAP["text_column"], COLUMN_MAP["status_column"]
        r = self.client.get(self.base, params={
            "select": f"{idc},{txt}", st: "in.(pending,failed)", "limit": str(limit)})
        r.raise_for_status()
        return r.json()

    def update(self, review_id, payload):
        # เขียนเฉพาะแถวที่ยังไม่ done กันการเขียนทับผลที่ใหม่กว่า
        r = self.client.patch(self.base, json=payload, params={
            COLUMN_MAP["id_column"]: f"eq.{review_id}",
            COLUMN_MAP["status_column"]: "in.(pending,failed)"},
            headers={"Prefer": "return=minimal"})
        r.raise_for_status()


def to_db_payload(result):
    payload = {COLUMN_MAP["status_column"]: result.status, COLUMN_MAP["error_column"]: result.error}
    for name, value in result.fields.items():
        col = COLUMN_MAP["fields"].get(name)
        if col:
            payload[col] = value
    return payload


# ---------------------------------------------------------------- แอป
analyzer = None
repo = None


def startup():
    global analyzer, repo
    analyzer = ReviewAnalyzer(BASE / os.getenv("MODEL_DIR", "model"),
                              model_version=os.getenv("MODEL_VERSION", "nb-v1"),
                              mymemory_email=os.getenv("MYMEMORY_EMAIL"))
    analyzer.analyze("ทดสอบระบบ วิวสวยมาก")          # อุ่นเครื่อง ให้คำขอแรกไม่ช้า
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        repo = SupabaseRepo(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    log.info("ready: %s | database: %s", analyzer.versions, "connected" if repo else "not configured")


@asynccontextmanager
async def lifespan(_app):
    startup()                                          # โหลดโมเดลครั้งเดียวตอนเปิดเซิร์ฟเวอร์
    yield


app = FastAPI(title="Songkhla Review Sentiment API", version="1.0", lifespan=lifespan)


def require_secret(secret):
    expected = os.getenv("WEBHOOK_SECRET", "")
    if not expected or not secret or not hmac.compare_digest(secret, expected):
        raise HTTPException(status_code=401, detail="invalid secret")


def require_repo():
    if repo is None:
        raise HTTPException(status_code=503, detail="database not configured")


def process_review(review_id, text):
    result = analyzer.analyze(text)
    try:
        repo.update(review_id, to_db_payload(result))
    except Exception as e:
        log.error("update failed for %s: %s", review_id, e)
        return "db_error"
    if result.status == "failed":
        log.warning("analysis failed for %s: %s", review_id, result.error)
    return result.status


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LEN)


@app.get("/health")
def health():
    return {"status": "ok", **(analyzer.versions if analyzer else {}), "database": repo is not None}


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest):
    result = analyzer.analyze(req.text)
    return {"status": result.status, "error": result.error, **result.fields}


@app.post("/v1/webhook/reviews", status_code=202)
def webhook(payload: dict, background: BackgroundTasks, x_webhook_secret: str = Header(None)):
    require_secret(x_webhook_secret)
    require_repo()
    record = payload.get("record") or payload          # รองรับรูปแบบของ Supabase Database Webhook
    if payload.get("type", "INSERT") != "INSERT":
        return {"skipped": "not an insert"}
    review_id = record.get(COLUMN_MAP["id_column"])
    text = record.get(COLUMN_MAP["text_column"])
    if not review_id:
        raise HTTPException(status_code=400, detail="missing review id")
    # ตอบกลับทันที แล้วประมวลผลเบื้องหลัง (Supabase webhook มีเวลารอจำกัด)
    background.add_task(process_review, review_id, (text or "")[:MAX_TEXT_LEN])
    return {"accepted": review_id}


@app.post("/v1/process-pending")
def process_pending(limit: int = 50, x_webhook_secret: str = Header(None)):
    require_secret(x_webhook_secret)
    require_repo()
    rows = repo.fetch_pending(min(max(limit, 1), 200))
    summary = {"done": 0, "failed": 0, "db_error": 0}
    for row in rows:
        status = process_review(row[COLUMN_MAP["id_column"]],
                                (row.get(COLUMN_MAP["text_column"]) or "")[:MAX_TEXT_LEN])
        summary[status] = summary.get(status, 0) + 1
    return {"processed": len(rows), **summary}
