"""ทดสอบปลายทางจริงกับฐานข้อมูลจำลองของเว็บเพื่อน (Postgres บนเครื่อง + SQL ติดตั้งจริง)
API -> (PostgREST จำลอง: GET/PATCH แปลงเป็น SQL ในฐานะ service_role) -> trigger คำนวณ sentiment สถานที่
ข้ามอัตโนมัติถ้าไม่ได้ตั้ง FRIEND_PG_DSN (ไม่จำเป็นต่อการ deploy)"""
import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from api import app as appmod

DSN = os.getenv("FRIEND_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="ตั้ง FRIEND_PG_DSN เพื่อทดสอบกับ Postgres จำลอง")
SECRET = "test-secret"
FRIEND_MAP = json.loads((appmod.Path(appmod.__file__).parent / "column_map.friend.json").read_text(encoding="utf-8"))


class PostgrestOnPostgres:
    """จำลอง Supabase REST เท่าที่ API ใช้ (select ... in.(pending,failed) / patch ... eq.id)"""

    def __init__(self, conn):
        self.conn = conn

    def _as_service(self, cur):
        cur.execute("set local role service_role")
        cur.execute("select set_config('request.jwt.claims', '{\"role\":\"service_role\"}', true)")

    def handler(self, request):
        p = request.url.params
        with self.conn.transaction(), self.conn.cursor() as cur:
            self._as_service(cur)
            if request.method == "GET":
                cols = p["select"].split(",")
                cur.execute(f"select {', '.join(cols)} from public.reviews "
                            "where analysis_status in ('pending','failed') limit %s", (int(p["limit"]),))
                return httpx.Response(200, json=[dict(zip(cols, r)) for r in cur.fetchall()])
            if request.method == "PATCH":
                body = json.loads(request.content)
                rid = int(p["id"].removeprefix("eq."))
                sets = ", ".join(f'"{k}" = %s' for k in body)
                cur.execute(f"update public.reviews set {sets} where id = %s "
                            "and analysis_status in ('pending','failed')", [*body.values(), rid])
                return httpx.Response(204)
        return httpx.Response(405)


@pytest.fixture
def client(monkeypatch):
    import psycopg
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setattr(appmod, "COLUMN_MAP", FRIEND_MAP)
    conn = psycopg.connect(DSN, autocommit=True)
    with TestClient(appmod.app) as c:
        appmod.repo = appmod.SupabaseRepo("https://fake.supabase.co", "key",
                                          transport=httpx.MockTransport(PostgrestOnPostgres(conn).handler))
        yield c, conn
    conn.close()


def submit_as_user(conn, attraction_id, stars, text, user="11111111-1111-1111-1111-111111111111"):
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("set local role authenticated")
        cur.execute("select set_config('request.jwt.claims', %s, true)",
                    (json.dumps({"role": "authenticated", "sub": user}),))
        cur.execute("select public.submit_tourist_review(%s, %s, %s, %s)", (attraction_id, user, stars, text))
        cur.execute("select id, text from public.reviews order by id desc limit 1")
        return cur.fetchone()


def test_new_review_flows_to_place_sentiment(client):
    c, conn = client
    before = conn.execute("select analyzed_reviews, avg_prob_negative from public.attractions where id = 2").fetchone()
    rid, text = submit_as_user(conn, 2, 1, "ห้องน้ำสกปรกมาก คนเยอะเกินไป ไม่ประทับใจเลย")
    payload = {"type": "INSERT", "table": "reviews", "record": {"id": rid, "text": text, "analysis_status": "pending"}}
    r = c.post("/v1/webhook/reviews", json=payload, headers={"x-webhook-secret": SECRET})
    assert r.status_code == 202
    row = conn.execute("select analysis_status, sentiment, analyzable, negative_text, model_version "
                       "from public.reviews where id = %s", (rid,)).fetchone()
    assert row[0] == "done" and row[1] == "negative" and row[2] == "yes", row
    assert row[3] and "ห้องน้ำสกปรก" in row[3]
    after = conn.execute("select analyzed_reviews, avg_prob_negative, place_sentiment from public.attractions "
                         "where id = 2").fetchone()
    assert after[0] == before[0] + 1 and after[2] == "negative"
    terms = {t for (t,) in conn.execute("select term from public.attraction_phrase_counts "
                                        "where attraction_id = 2 and polarity = 'negative'")}
    assert "ห้องน้ำสกปรก" in terms


def test_process_pending_picks_up_old_reviews(client):
    c, conn = client
    n = conn.execute("select count(*) from public.reviews where analysis_status <> 'done'").fetchone()[0]
    r = c.post("/v1/process-pending?limit=50", headers={"x-webhook-secret": SECRET}).json()
    assert r["processed"] == n
    assert conn.execute("select count(*) from public.reviews where analysis_status = 'pending'").fetchone()[0] == 0
