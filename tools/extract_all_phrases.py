"""สกัดวลีด้วย Typhoon LLM ให้รีวิวทั้งชุด (7,000+ รีวิว) พร้อม checkpoint -- หยุดเมื่อไหร่ก็ได้ รันใหม่ทำต่อจากเดิม

ติดตั้ง (ครั้งเดียว):  py -m pip install pythainlp==5.3.8 httpx pandas openpyxl
ตั้ง API key (PowerShell):  $env:TYPHOON_API_KEY="ใส่-key"

ใช้งาน:
    py tools/extract_all_phrases.py reviews.csv
    py tools/extract_all_phrases.py reviews.csv --workers 2      # เร็วขึ้นเกือบ 2 เท่า (ยังอยู่ในโควตา)
หยุดกลางทาง: กด Ctrl+C (หรือปิดหน้าต่าง/คอมดับ) -> รันคำสั่งเดิมอีกครั้ง จะทำต่อจากรีวิวที่ค้าง
รีวิวที่ล้ม (เช่น โควตาเต็ม) จะถูกลองใหม่อัตโนมัติทุกครั้งที่รันคำสั่งเดิม
ไฟล์ที่ได้ (โฟลเดอร์ phrases_<รุ่น>/):
    checkpoint.jsonl        ผลดิบทีละรีวิว (ห้ามลบระหว่างรัน -- ลบเมื่ออยากเริ่มใหม่ทั้งหมด)
    reviews_phrases.csv     1 แถวต่อรีวิว: วลีจาก LLM และจาก lexicon คู่กัน
    phrases_long.csv        1 แถวต่อวลี (reviewId, ลำดับ, วลี, ขั้ว, หัวข้อ) -- word cloud ใน Power BI
                            (ปิด Word breaking ของภาพ Word Cloud ไม่งั้นวลีไทยจะถูกตัดเป็นคำ)
                            aspect = หัวข้อที่ถูกประเมิน (ห้องน้ำ ราคา บรรยากาศ) ว่างได้ ใช้ทำ cloud/กราฟแยกหัวข้อ
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentiment_core import llm_extractor as lx                                   # noqa: E402
from sentiment_core.extractor import PhraseExtractor                             # noqa: E402
from sentiment_core.llm_extractor import LLMExtractionError, make_extractor      # noqa: E402

_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def has_letters(text):
    """มีตัวอักษรไหม (กฎเดียวกับ sentiment_core.language แต่ไม่ต้องลง langdetect)"""
    return bool(_LETTER.search(text or ""))


MAX_CONSECUTIVE_ERRORS = 10      # ล้มติดกันเท่านี้ = น่าจะโควตาหมด/เน็ตหลุด -> พักก่อน
PAUSE_SECONDS = 60
MAX_PAUSES = 5                   # พักครบเท่านี้แล้วยังล้ม -> หยุดโปรแกรม (ผลที่ได้แล้วยังอยู่ใน checkpoint)


def read_table(path):
    path = Path(path)
    return pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)


def load_checkpoint(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:      # บรรทัดสุดท้ายอาจขาดครึ่งถ้าคอมดับกลางทาง -> ข้าม
                continue
            done[row["reviewId"]] = row       # ถ้ามีซ้ำ ใช้ผลล่าสุด (เช่น หลัง --retry-errors)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="ไฟล์รีวิว .csv/.xlsx ที่มีคอลัมน์ reviewId, processed_text")
    ap.add_argument("--workers", type=int, default=1, help="จำนวนคำขอพร้อมกัน 1-3 (ค่าเริ่มต้น 1)")
    ap.add_argument("--limit", type=int, help="ทำแค่กี่รีวิว (ไว้ทดลอง)")
    ap.add_argument("--model", default=os.getenv("TYPHOON_MODEL"))
    ap.add_argument("--prompt", default="v9", choices=["v8.1", "v9"], help="รุ่น prompt (ค่าเริ่มต้น v9)")
    args = ap.parse_args()

    key = os.getenv("TYPHOON_API_KEY")
    if not key:
        sys.exit("ยังไม่ได้ตั้ง TYPHOON_API_KEY  (PowerShell: $env:TYPHOON_API_KEY=\"...\")")
    workers = max(1, min(args.workers, 3))

    llm = make_extractor(args.prompt, key, args.model)
    out_dir = Path(f"phrases_{llm.version}")
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "checkpoint.jsonl"
    print(f"ตัวสกัด: {llm.version} | วลีสูงสุด {lx.MAX_TOKENS} คำ | ผลลัพธ์: {out_dir}{os.sep}")

    df = read_table(args.input)
    df = df[df["processed_text"].notna()].drop_duplicates("reviewId").reset_index(drop=True)
    done = load_checkpoint(ckpt)

    def needs_work(r):
        d = done.get(r.reviewId)
        return d is None or d["status"].startswith("ERROR")      # ยังไม่ทำ หรือเคยล้ม -> ทำ
    todo = [r for r in df.itertuples() if needs_work(r)]
    if args.limit:
        todo = todo[:args.limit]
    n_err_before = sum(d["status"].startswith("ERROR") for d in done.values())
    print(f"รีวิวทั้งหมด {len(df)} | ทำแล้ว {len(done)} (ล้ม {n_err_before}) | รอบนี้จะทำ {len(todo)} | พร้อมกัน {workers}")

    lock = threading.Lock()
    state = {"n": 0, "ok": 0, "err": 0, "consec": 0, "pauses": 0, "t0": time.time()}

    def work(r):
        text = str(r.processed_text)
        if not has_letters(text):                 # อีโมจิ/สัญลักษณ์ล้วน: ไม่ต้องเรียก LLM
            return {"reviewId": r.reviewId, "kept": [], "dropped": [], "status": "skipped",
                    "seconds": 0, "version": llm.version}
        t = time.time()
        items = []
        try:
            items = llm.extract_items(text)          # เก็บคำตอบดิบไว้ด้วย: แก้กฎตรวจแล้วประมวลผลซ้ำได้ไม่ต้องเรียก API
            kept, dropped = llm.process(text, items)
            status = "ok"
        except LLMExtractionError as e:
            kept, dropped, status = [], [], f"ERROR: {e}"[:300]
        return {"reviewId": r.reviewId, "kept": kept, "dropped": dropped, "items": items, "status": status,
                "seconds": round(time.time() - t, 2), "version": llm.version}

    def record(row, f):
        with lock:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())                  # บันทึกลงดิสก์ทันที กันคอมดับ
            done[row["reviewId"]] = row
            state["n"] += 1
            if row["status"].startswith("ERROR"):
                state["err"] += 1
                state["consec"] += 1
            else:
                state["ok"] += 1
                state["consec"] = 0
            n = state["n"]
            if n % 25 == 0 or n == len(todo):
                el = time.time() - state["t0"]
                eta = el / n * (len(todo) - n)
                print(f"[{n}/{len(todo)}] สำเร็จ {state['ok']} ล้ม {state['err']} | "
                      f"{el / n:.2f} วิ/รีวิว | เหลือประมาณ {eta / 60:.0f} นาที", flush=True)

    stopped = None
    pool = ThreadPoolExecutor(workers)
    try:
        with ckpt.open("a", encoding="utf-8") as f:
            i = 0
            while i < len(todo):
                batch = todo[i:i + workers * 4]
                for fut in as_completed([pool.submit(work, r) for r in batch]):
                    record(fut.result(), f)
                i += len(batch)
                if state["consec"] >= MAX_CONSECUTIVE_ERRORS:
                    state["pauses"] += 1
                    if state["pauses"] > MAX_PAUSES:
                        stopped = "Typhoon ล้มติดกันหลายรอบ (โควตาหมดหรือเน็ตหลุด)"
                        break
                    last = next(d for d in reversed(list(done.values())) if d["status"].startswith("ERROR"))
                    print(f"⚠️ ล้มติดกัน {state['consec']} ครั้ง ({last['status'][:80]}) "
                          f"-> พัก {PAUSE_SECONDS} วิ (ครั้งที่ {state['pauses']}/{MAX_PAUSES})", flush=True)
                    time.sleep(PAUSE_SECONDS)
                    state["consec"] = 0
    except KeyboardInterrupt:
        stopped = "หยุดโดยผู้ใช้ (Ctrl+C)"
    finally:
        pool.shutdown(wait=True, cancel_futures=True)   # ยกเลิกงานที่ยังไม่เริ่ม ไม่เสียโควตาเปล่า

    # ---------------------------------------------------------------- เขียนไฟล์ผลลัพธ์จากทุกอย่างที่ทำแล้ว
    lex = PhraseExtractor()
    wide, long_rows = [], []
    for r in df.itertuples():
        d = done.get(r.reviewId)
        lexr = lex.analyze(str(r.processed_text)) if has_letters(str(r.processed_text)) else \
            {"sentiment_text": None, "positive_text": None, "negative_text": None, "n_sentiment_terms": 0}
        row = {"reviewId": r.reviewId, "attraction_id": getattr(r, "attraction_id", None),
               "processed_text": r.processed_text, "sentiment": getattr(r, "sentiment", None)}
        if d is None:
            row.update({"llm_status": "pending"})
        else:
            pos = [k["phrase"] for k in d["kept"] if k["polarity"] == "positive"]
            neg = [k["phrase"] for k in d["kept"] if k["polarity"] == "negative"]
            row.update({"llm_sentiment_text": " ".join(k["phrase"] for k in d["kept"]) or None,
                        "llm_positive_text": " ".join(pos) or None, "llm_negative_text": " ".join(neg) or None,
                        "llm_n_terms": len(d["kept"]),
                        "llm_dropped": " ".join(x["phrase"] for x in d["dropped"]) or None,
                        "llm_status": d["status"], "llm_seconds": d["seconds"], "llm_version": d.get("version")})
            for order, k in enumerate(d["kept"], start=1):
                long_rows.append({"reviewId": r.reviewId, "attraction_id": row["attraction_id"],
                                  "term_order": order, "term": k["phrase"], "polarity": k["polarity"],
                                  "aspect": k.get("aspect", "")})
        row.update({"lex_sentiment_text": lexr["sentiment_text"], "lex_positive_text": lexr["positive_text"],
                    "lex_negative_text": lexr["negative_text"], "lex_n_terms": lexr["n_sentiment_terms"]})
        wide.append(row)
    W = pd.DataFrame(wide)
    W.to_csv(out_dir / "reviews_phrases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(long_rows, columns=["reviewId", "attraction_id", "term_order", "term", "polarity", "aspect"]).to_csv(
        out_dir / "phrases_long.csv", index=False, encoding="utf-8-sig")

    st = W["llm_status"].fillna("pending")
    ok = W[st == "ok"]
    print(f"\nบันทึก: {out_dir / 'reviews_phrases.csv'} และ {out_dir / 'phrases_long.csv'}")
    print(f"สถานะ: สำเร็จ {(st == 'ok').sum()} | ข้าม (ไม่มีตัวอักษร) {(st == 'skipped').sum()} | "
          f"ล้ม {st.str.startswith('ERROR').sum()} | ยังไม่ได้ทำ {(st == 'pending').sum()}")
    if len(ok):
        print(f"รีวิวที่มีวลี: LLM {(ok['llm_n_terms'] > 0).mean():.1%} | lexicon {(ok['lex_n_terms'] > 0).mean():.1%}"
              f"  (เฉพาะรีวิวที่ LLM ทำสำเร็จ)")
        print(f"เวลาเฉลี่ย {ok['llm_seconds'].mean():.2f} วิ/รีวิว")
    left = int((st == "pending").sum() + st.str.startswith("ERROR").sum())
    if stopped:
        print(f"\n⏸ {stopped} -- รันคำสั่งเดิมอีกครั้งเพื่อทำต่อ (เหลือ {left} รีวิว)")
    elif left:
        print(f"\nยังเหลือ {left} รีวิว (ยังไม่ได้ทำหรือเคยล้ม) -- รันคำสั่งเดิมอีกครั้งเพื่อทำต่อ")
    else:
        print("\n✅ ครบทุกรีวิวแล้ว")


if __name__ == "__main__":
    main()
