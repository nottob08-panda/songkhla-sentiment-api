"""เสนอคำความรู้สึกใหม่ให้คนพิจารณาเพิ่มในพจนานุกรม (resources/lexicon.json)

ทำไมไม่เพิ่มอัตโนมัติ: น้ำหนักคำจาก Naive Bayes ปนคำบอก "หัวข้อ" (เช่น ไหว้ ครอบครัว หลังคา)
มากกว่าคำบอกความรู้สึก ถ้าเพิ่มอัตโนมัติจะได้วลีผิด จึงใช้เป็นรายการเสนอให้คนคัดเท่านั้น

วิธีใช้
  1) Supabase > Table editor > view "reviews_without_phrases" > Export to CSV
  2) python tools/suggest_lexicon_words.py reviews_without_phrases.csv
  3) เปิด lexicon_candidates.csv เลือกคำที่เป็นคำความรู้สึกจริง
     เพิ่มใน POSITIVE_WORDS / NEGATIVE_WORDS ของ lexicon.json และเปลี่ยน "version" (เช่น lexicon-2)
  4) deploy ใหม่ แล้ววิเคราะห์รีวิวเก่าซ้ำได้โดยตั้ง analysis_status = 'pending'
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from pythainlp.corpus import thai_stopwords

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sentiment_core.extractor import PhraseExtractor  # noqa: E402
from sentiment_core.model import SentimentModel        # noqa: E402


def main(csv_path, out_path="lexicon_candidates.csv", min_reviews=3, min_score=1.0):
    df = pd.read_csv(csv_path)
    ex, model = PhraseExtractor(), SentimentModel(ROOT / "model")
    scores = model.word_polarity_scores()
    known = (ex.LEX | ex.ASP_T | ex.NEG_T | ex.PRE_T | ex.POST_T | ex.FILL_T | ex.ABUND_T
             | set(thai_stopwords()))

    counts, examples = Counter(), {}
    for text in df["processed_text"].fillna(""):
        for tok in set(ex.tokenize(text)):
            if re.fullmatch(r"[ก-๙]{2,}", tok) and tok not in known and tok in scores:
                counts[tok] += 1
                examples.setdefault(tok, text[:120])

    rows = [{"word": w, "suggested_polarity": "positive" if scores[w] > 0 else "negative",
             "nb_score": round(scores[w], 3), "n_reviews_without_phrases": n, "example": examples[w]}
            for w, n in counts.items() if n >= min_reviews and abs(scores[w]) >= min_score]
    out = pd.DataFrame(rows).sort_values(["n_reviews_without_phrases", "nb_score"],
                                         key=lambda s: s.abs(), ascending=False)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"เสนอ {len(out)} คำ จากรีวิวที่สกัดวลีไม่เจอ {len(df):,} รายการ -> {out_path}")
    print("ตรวจด้วยตาก่อนเพิ่มทุกครั้ง: คำหลายคำเป็นหัวข้อ ไม่ใช่ความรู้สึก")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python tools/suggest_lexicon_words.py reviews_without_phrases.csv")
    main(sys.argv[1])
