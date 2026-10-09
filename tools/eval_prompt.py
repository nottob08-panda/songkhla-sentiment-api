"""วัดผล prompt ของ LLM บนชุดทดสอบคงที่ (tools/prompt_testset.csv, 97 รีวิวจริง) -- ใช้ทุกครั้งที่แก้ prompt

ชุดทดสอบเลือกจากผลรันทั้งชุดของ v8.1 ให้ครอบคลุมจุดที่เคยพลาด:
  aspect_loss (หัวข้อหาย), exists_mi ("มี X"), long, elongation (มากกก), advice, negation, no_opinion,
  human_checked (รีวิวที่คนตรวจแล้ว), random
ผลของ v8.1 และ lexicon เก็บไว้ในไฟล์แล้ว จึงรันเฉพาะ prompt ใหม่ (~2-3 นาที)

ใช้งาน (PowerShell):
    $env:TYPHOON_API_KEY="..."
    py tools/eval_prompt.py                 # วัด v9
    py tools/eval_prompt.py --prompt v8.1   # รัน v8.1 ซ้ำ (ไว้ดูความแกว่งของ LLM)
    py tools/eval_prompt.py --reprocess     # ไม่เรียก API: ใช้คำตอบดิบที่เก็บไว้ ตรวจใหม่ด้วยกฎในโค้ดปัจจุบัน
    py tools/eval_prompt.py --testset tools/holdout_testset.csv --out-tag holdout
                                            # ชุด 30 รีวิวที่ไม่เคยใช้ปรับ prompt -> ให้คนตรวจด้วยตา (ชีต 'ตรวจด้วยตา')
หยุดกลางทางได้ รันใหม่ทำต่อจากเดิม ผลอยู่ในโฟลเดอร์ eval_<รุ่น>/  (สรุป + eval_<รุ่น>.xlsx ไว้อ่านทีละรีวิว)

ตัวชี้วัดทั้งหมดคำนวณอัตโนมัติ ไม่ต้องมีคนให้คะแนน:
  เก็บหัวข้อ   = เมื่อพจนานุกรมพบคู่ "หัวข้อ+คำประเมิน" ติดกันในรีวิว (เช่น บรรยากาศดี) LLM คงหัวข้อไว้กี่ %
  หัวข้อหาย    = วลีที่เหลือแต่คำประเมิน ทั้งที่ในรีวิวมีคำหัวข้ออยู่ติดหน้า (เช่น ได้ "ดี" จาก "บรรยากาศดี")
  วลีขยะ       = คำเชื่อมล้วน / ลงท้าย "มี" / ตัวอักษรซ้ำ 3 ตัวขึ้นไป
  ถูกปฏิเสธ    = วลีที่ตัวตรวจไม่ยอมรับ (ไม่อยู่ในรีวิว / ยาวเกิน 6 คำ)
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sentiment_core import llm_extractor as lx                          # noqa: E402
from sentiment_core.extractor import PhraseExtractor                    # noqa: E402
from sentiment_core.llm_extractor import LLMExtractionError, make_extractor  # noqa: E402

LEX = PhraseExtractor()
ASPECTS = sorted(LEX.ASP_T - LEX.GENERIC_ASPECTS, key=len, reverse=True)


def flat(s):
    return re.sub(r"\s+", "", str(s or "")).replace("ๆ", "")


def toks(s):
    return [] if pd.isna(s) or not str(s).strip() else str(s).split()


def aspect_pairs(text):
    """คู่ (หัวข้อ, คำประเมิน) ที่พจนานุกรมพบในรีวิว"""
    out = []
    for e in LEX.extract(str(text)):
        if e["aspect"] and e["aspect"] not in LEX.GENERIC_ASPECTS:
            out.append((e["aspect"], e["neg"] + LEX.COMPACT_NORMALIZE.get(e["core"], e["core"])))
    return out


def review_metrics(text, phrases):
    f = flat(text)
    kept = lost = 0
    for a, c in aspect_pairs(text):
        if any(a in p and c in p for p in phrases) or any(a in p and c.replace("ไม่", "") in p for p in phrases):
            kept += 1
        elif c in phrases:
            lost += 1
    bare = sum(1 for p in phrases if not any(p.startswith(a) for a in ASPECTS)
               and any((a + p) in f for a in ASPECTS))
    junk = sum(1 for p in phrases if p in lx.FUNCTION_WORDS or (p.endswith("มี") and not p.endswith("ไม่มี"))
               or re.search(r"(.)\1\1", p))
    return {"pair_kept": kept, "pair_lost": lost, "bare": bare, "junk": junk, "n": len(phrases)}


def summarize(df, col):
    rows = [review_metrics(t, toks(p)) for t, p in zip(df["processed_text"], df[col])]
    m = pd.DataFrame(rows)
    kept, lost = m.pair_kept.sum(), m.pair_lost.sum()
    return {
        "รีวิวที่มีวลี": f"{(m.n > 0).mean():.1%}",
        "วลี/รีวิว": f"{m.n.mean():.2f}",
        "เก็บหัวข้อ": f"{kept / (kept + lost):.1%}" if kept + lost else "-",
        "หัวข้อหาย (วลี)": int(m.bare.sum()),
        "วลีขยะ": int(m.junk.sum()),
    }, m


def check_sheet(T, ok, done):
    """หนึ่งแถวต่อวลี ให้คนกรอก: คะแนน 1 = ถูกและครบ, 0.5 = ใช่ความเห็นแต่ขาดหัวข้อ/เกิน/จัดรูปแบบแปลก, 0 = ไม่ใช่ความเห็น/ผิด
    ขั้ว: ถูก/ผิด  และท้ายแต่ละรีวิวมีแถว "ความเห็นที่หลุด" ให้พิมพ์ความเห็นที่ LLM ไม่ได้ดึงมา"""
    rows = []
    for n, r in enumerate(ok.itertuples(), 1):
        kept = done[r.reviewId]["kept"]
        first = True
        for k in kept or [None]:
            rows.append({"ลำดับ": n, "reviewId": r.reviewId, "รีวิว": r.processed_text if first else "",
                         "วลี": k["phrase"] if k else "(ไม่มีวลี)",
                         "ขั้ว": ("บวก" if k["polarity"] == "positive" else "ลบ") if k else "",
                         "ที่มา": k.get("how", "") if k else "", "คะแนน (1/0.5/0)": "", "ขั้วถูก? (ถูก/ผิด)": ""})
            first = False
        rows.append({"ลำดับ": n, "reviewId": r.reviewId, "รีวิว": "", "วลี": "ความเห็นที่หลุด ->", "ขั้ว": "",
                     "ที่มา": "", "คะแนน (1/0.5/0)": "", "ขั้วถูก? (ถูก/ผิด)": ""})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="v9", choices=sorted(lx.PROMPTS))
    ap.add_argument("--testset", default=str(ROOT / "tools" / "prompt_testset.csv"))
    ap.add_argument("--model", default=os.getenv("TYPHOON_MODEL"))
    ap.add_argument("--out-tag", default="", help="ต่อท้ายชื่อโฟลเดอร์ผลลัพธ์ (เช่น holdout) แยกจากชุดทดสอบหลัก")
    ap.add_argument("--reprocess", action="store_true", help="ไม่เรียก API ใช้คำตอบดิบใน checkpoint ตรวจใหม่")
    args = ap.parse_args()
    key = os.getenv("TYPHOON_API_KEY") or ("offline" if args.reprocess else None)
    if not key:
        sys.exit("ยังไม่ได้ตั้ง TYPHOON_API_KEY  (PowerShell: $env:TYPHOON_API_KEY=\"...\")")

    llm = make_extractor(args.prompt, key, args.model)
    out_dir = Path(f"eval_{llm.version}" + (f"_{args.out_tag}" if args.out_tag else ""))
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "checkpoint.jsonl"
    print(f"ตัวสกัด: {llm.version} | ชุดทดสอบ: {args.testset} | ผลลัพธ์: {out_dir}{os.sep}")

    T = pd.read_csv(args.testset)
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[r["reviewId"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
    if args.reprocess:
        n = 0
        for rid, row in done.items():
            if row.get("status") == "ok" and "items" in row:
                text = T.loc[T.reviewId == rid, "processed_text"]
                if len(text):
                    row["kept"], row["dropped"] = llm.process(str(text.iloc[0]), row["items"])
                    n += 1
        print(f"ตรวจใหม่จากคำตอบดิบ {n} รีวิว (ไม่เรียก API)")
    todo = [] if args.reprocess else [r for r in T.itertuples()
                                      if r.reviewId not in done or done[r.reviewId]["status"] != "ok"]
    print(f"{len(T)} รีวิว | ทำแล้ว {len(T) - len(todo)} | รอบนี้ {len(todo)}")
    try:
        with ckpt.open("a", encoding="utf-8") as fh:
            for i, r in enumerate(todo, 1):
                t0 = time.time()
                items = []
                try:
                    items = llm.extract_items(str(r.processed_text))
                    kept, dropped = llm.process(str(r.processed_text), items)
                    status = "ok"
                except LLMExtractionError as e:
                    kept, dropped, status = [], [], f"ERROR: {e}"[:200]
                row = {"reviewId": r.reviewId, "kept": kept, "dropped": dropped, "items": items, "status": status,
                       "seconds": round(time.time() - t0, 2)}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done[r.reviewId] = row
                if i % 10 == 0 or i == len(todo):
                    print(f"[{i}/{len(todo)}]", flush=True)
    except KeyboardInterrupt:
        print("หยุดกลางทาง -- รันคำสั่งเดิมเพื่อทำต่อ")

    ver = llm.version
    T[f"{ver}_text"] = [" ".join(k["phrase"] for k in done[i]["kept"]) if i in done else None for i in T.reviewId]
    T[f"{ver}_dropped"] = [" | ".join(f'{k["phrase"]} ({k["reason"]})' if k.get("reason") else k["phrase"]
                                      for k in done[i]["dropped"]) if i in done else None for i in T.reviewId]
    T[f"{ver}_salvaged"] = [" ".join(k["phrase"] for k in done[i]["kept"] if k.get("how") == "กู้คืน")
                            if i in done else None for i in T.reviewId]
    T[f"{ver}_llm_raw"] = [json.dumps(done[i].get("items", []), ensure_ascii=False) if i in done else None
                           for i in T.reviewId]
    T["status"] = [done[i]["status"] if i in done else "pending" for i in T.reviewId]
    T["seconds"] = [done[i]["seconds"] if i in done else None for i in T.reviewId]
    ok = T[T.status == "ok"]

    cols = {"lexicon": "lex_sentiment_text", "llm-typhoon-v8.1 (เดิม)": "v81_sentiment_text", ver: f"{ver}_text"}
    summary, per = {}, {}
    for name, col in cols.items():
        summary[name], per[name] = summarize(ok, col)
    S = pd.DataFrame(summary)
    n_kept = sum(len(done[i]["kept"]) for i in ok.reviewId)
    n_drop = sum(len(done[i]["dropped"]) for i in ok.reviewId)
    secs = ok.seconds.tolist() or [0]
    reasons = {}
    for i in ok.reviewId:
        for k in done[i]["dropped"]:
            reasons[k.get("reason", "-")] = reasons.get(k.get("reason", "-"), 0) + 1
    n_salv = sum(1 for i in ok.reviewId for k in done[i]["kept"] if k.get("how") == "กู้คืน")
    extra = {"ถูกปฏิเสธ": f"{n_drop}/{n_kept + n_drop} ข้อ ({n_drop / max(1, n_kept + n_drop):.1%})",
             "ถูกปฏิเสธ แยกเหตุผล": ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])) or "-",
             "กู้คืน (หัวข้อ+คำประเมิน)": f"{n_salv}/{n_kept} วลี",
             "รีวิวที่ LLM ตอบว่าง": int(sum(1 for i in ok.reviewId if not done[i].get("items", [1]))),
             "เวลา (เฉลี่ย / p95)": f"{statistics.mean(secs):.2f} / {sorted(secs)[int(len(secs) * .95) - 1]:.2f} วิ",
             "ล้มเหลว": int((T.status != "ok").sum())}

    pd.set_option("display.width", 200)
    print(f"\n=== สรุป ({len(ok)} รีวิว) ===")
    print(S.to_string())
    for k, v in extra.items():
        print(f"{ver} {k}: {v}")

    cat = ok.assign(**{f"{n}_lost": per[n].bare.values for n in cols})
    by_cat = cat.groupby("category")[[f"{n}_lost" for n in cols]].sum()
    by_cat.columns = [f"หัวข้อหาย: {n}" for n in cols]
    print("\nหัวข้อหายแยกตามกลุ่ม:\n" + by_cat.to_string())

    with pd.ExcelWriter(out_dir / f"eval_{ver}.xlsx") as xw:
        pd.concat([S, pd.DataFrame({ver: extra})]).to_excel(xw, sheet_name="สรุป")
        by_cat.to_excel(xw, sheet_name="แยกกลุ่ม")
        check_sheet(T, ok, done).to_excel(xw, sheet_name="ตรวจด้วยตา", index=False)
        T[["category", "reviewId", "processed_text", "lex_sentiment_text", "v81_sentiment_text",
           f"{ver}_text", f"{ver}_salvaged", f"{ver}_dropped", f"{ver}_llm_raw", "status", "seconds"]].to_excel(xw, sheet_name="รายรีวิว", index=False)
    print(f"\nบันทึก: {out_dir / f'eval_{ver}.xlsx'}  (ชีต 'รายรีวิว' วางผลแต่ละวิธีคู่กัน, "
          f"ชีต 'ตรวจด้วยตา' ให้คนกรอกคะแนนทีละวลี)")


if __name__ == "__main__":
    main()
