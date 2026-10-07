"""เปรียบเทียบการสกัดวลี 2 วิธี บนรีวิวชุดเดียวกัน: พจนานุกรม (lexicon) vs Typhoon LLM (API)

ใช้ตัดสินใจก่อนเปิด PHRASE_METHOD=llm บน Render -- รันบนเครื่องตัวเอง (ต้องต่ออินเทอร์เน็ต)

ติดตั้ง (ครั้งเดียว ในโฟลเดอร์ของ repo):
    pip install pythainlp==5.3.8 httpx pandas openpyxl

ตั้ง API key (Windows cmd):            set TYPHOON_API_KEY=ใส่-key-ตรงนี้
            (PowerShell):              $env:TYPHOON_API_KEY="ใส่-key-ตรงนี้"

ตัวอย่าง:
    # 1) ลอง 30 รีวิวก่อน ดูเวลา/คุณภาพ
    python tools/compare_phrase_methods.py reviews.csv --sample 30
    # 2) รันเฉพาะรีวิวในชุด gold (ไฟล์ที่มีคอลัมน์ reviewId) เพื่อนำไปเทียบกับคำตอบที่ label มือ
    python tools/compare_phrase_methods.py reviews.csv --ids gold_phrase_annotation.xlsx

ไฟล์ input ต้องมีคอลัมน์ reviewId และ processed_text (เช่น reviews.csv จาก notebook 04)
ผลลัพธ์: phrase_method_comparison.xlsx
  - lex_*  = ผลจากพจนานุกรม (ตัวเดียวกับใน API)
  - llm_*  = ผลจาก Typhoon (ชื่อคอลัมน์เดียวกับสคริปต์ extract_phrases_with_typhoon.py เดิม
             จึงใช้กับ notebook เปรียบเทียบที่มีอยู่ได้ทันที)
  - llm_seconds = เวลาที่ใช้ต่อรีวิว
หยุดกลางทางได้ รันคำสั่งเดิมอีกครั้งจะทำต่อจากเดิม (checkpoint)
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentiment_core.extractor import PhraseExtractor                      # noqa: E402
from sentiment_core.llm_extractor import LLMExtractionError, LLMPhraseExtractor   # noqa: E402


def read_table(path):
    path = Path(path)
    return pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="ไฟล์รีวิว (.csv/.xlsx) ที่มี reviewId, processed_text")
    ap.add_argument("--ids", help="รันเฉพาะ reviewId ในไฟล์นี้ (เช่น ชุด gold)")
    ap.add_argument("--sample", type=int, help="สุ่มมาทดลองกี่รีวิว")
    ap.add_argument("--out", default="phrase_method_comparison.xlsx")
    ap.add_argument("--model", default=os.getenv("TYPHOON_MODEL"), help="ชื่อโมเดล Typhoon (ไม่ใส่ = ค่าเริ่มต้น)")
    args = ap.parse_args()

    key = os.getenv("TYPHOON_API_KEY")
    if not key:
        sys.exit("ยังไม่ได้ตั้ง TYPHOON_API_KEY (ดูวิธีตั้งที่หัวไฟล์นี้)")

    df = read_table(args.input)
    df = df[df["processed_text"].notna()].reset_index(drop=True)
    if args.ids:
        ids = set(read_table(args.ids)["reviewId"])
        df = df[df["reviewId"].isin(ids)].reset_index(drop=True)
        print(f"รันเฉพาะ {len(df)} รีวิวจาก {args.ids}")
    elif args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
        print(f"สุ่ม {len(df)} รีวิว")

    lex = PhraseExtractor()
    llm = LLMPhraseExtractor(key, args.model)
    from sentiment_core import llm_extractor as _lx
    print(f"ตัวสกัด: {_lx.VERSION} | วลีสูงสุด {_lx.MAX_TOKENS} คำ | ไฟล์ {_lx.__file__}")
    ckpt_path = Path(args.out).with_suffix(".checkpoint.jsonl")
    done = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["reviewId"]] = row
        print(f"ทำต่อจากเดิม: เสร็จแล้ว {len(done)} รีวิว")

    todo = df[~df["reviewId"].isin(done)]
    t_start = time.time()
    with ckpt_path.open("a", encoding="utf-8") as ck:
        for i, r in enumerate(todo.itertuples(), start=1):
            text = str(r.processed_text)
            t0 = time.time()
            try:
                kept, dropped = llm.extract(text)
                status = "ok"
            except LLMExtractionError as e:
                kept, dropped, status = [], [], f"ERROR: {e}"
            row = {"reviewId": r.reviewId, "kept": kept, "dropped": dropped, "status": status, "version": llm.version,
                   "seconds": round(time.time() - t0, 2)}
            done[r.reviewId] = row
            ck.write(json.dumps(row, ensure_ascii=False) + "\n")
            ck.flush()
            if i % 10 == 0 or i == len(todo):
                avg = (time.time() - t_start) / i
                print(f"[{i}/{len(todo)}] เฉลี่ย {avg:.1f} วิ/รีวิว | เหลือประมาณ {avg * (len(todo) - i) / 60:.0f} นาที")

    out_rows = []
    for r in df.itertuples():
        d = done[r.reviewId]
        lx = lex.analyze(str(r.processed_text))
        pos = [k["phrase"] for k in d["kept"] if k["polarity"] == "positive"]
        neg = [k["phrase"] for k in d["kept"] if k["polarity"] == "negative"]
        out_rows.append({
            "reviewId": r.reviewId, "processed_text": r.processed_text,
            "sentiment": getattr(r, "sentiment", None),
            "lex_sentiment_text": lx["sentiment_text"], "lex_positive_text": lx["positive_text"],
            "lex_negative_text": lx["negative_text"], "lex_n_terms": lx["n_sentiment_terms"],
            "llm_sentiment_text": " ".join(k["phrase"] for k in d["kept"]) or None,
            "llm_positive_text": " ".join(pos) or None, "llm_negative_text": " ".join(neg) or None,
            "llm_n_terms": len(d["kept"]),
            "llm_dropped": " ".join(x["phrase"] for x in d["dropped"]) or None,
            "llm_status": d["status"], "llm_seconds": d["seconds"], "llm_version": d.get("version"),
        })
    out = pd.DataFrame(out_rows)
    out.to_excel(args.out, index=False)

    ok = out[out["llm_status"] == "ok"]
    secs = sorted(ok["llm_seconds"]) or [0]
    p95 = secs[min(len(secs) - 1, int(len(secs) * 0.95))]
    print(f"\nบันทึก: {args.out}")
    print(f"{'':28}{'lexicon':>10}{'LLM':>10}")
    print(f"{'รีวิวที่มีวลี':28}{(out['lex_n_terms'] > 0).mean():>10.1%}{(out['llm_n_terms'] > 0).mean():>10.1%}")
    print(f"{'วลีเฉลี่ยต่อรีวิว':28}{out['lex_n_terms'].mean():>10.2f}{out['llm_n_terms'].mean():>10.2f}")
    print(f"LLM เวลา: เฉลี่ย {statistics.mean(secs):.2f} วิ | มัธยฐาน {statistics.median(secs):.2f} วิ | p95 {p95:.2f} วิ")
    print(f"LLM ล้มเหลว: {(out['llm_status'] != 'ok').sum()} รีวิว | รีวิวที่มีวลีถูกตัด (ไม่พบในข้อความ/ยาวเกิน): "
          f"{out['llm_dropped'].notna().sum()}")


if __name__ == "__main__":
    main()
