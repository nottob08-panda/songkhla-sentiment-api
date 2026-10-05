"""โหลดโมเดลและทำนาย sentiment

รองรับไฟล์โมเดล 2 แบบ
- โมเดลเปล่า + tfidf_vectorizer แยกไฟล์ (Naive Bayes ที่ใช้อยู่ตอนนี้)
- Pipeline ที่มี TF-IDF อยู่ในตัว (ถ้าวันหน้าเปลี่ยนไปใช้โมเดลจาก GridSearchCV)

ไฟล์ .joblib ที่บันทึกจาก Colab อ้างถึงฟังก์ชัน `__main__.thai_tokenizer`
ตอนโหลดจึงต้องผูกฟังก์ชันเข้ากับ __main__ ก่อน (ปัญหาเดียวกับที่เจอใน notebook)
"""
import sys
import __main__
from pathlib import Path

import joblib
import numpy as np

from .text import thai_tokenizer

CLASSES = ("positive", "neutral", "negative")
UNANALYZABLE_SPREAD = 0.001   # ความน่าจะเป็นทั้ง 3 คลาสห่างกันน้อยกว่านี้ = โมเดลไม่รู้จักคำในรีวิว


class SentimentModel:
    def __init__(self, model_dir, model_version="nb-v1"):
        model_dir = Path(model_dir)
        __main__.thai_tokenizer = thai_tokenizer            # สำหรับไฟล์ที่บันทึกจาก notebook
        sys.modules["__main__"].thai_tokenizer = thai_tokenizer

        self.model = joblib.load(model_dir / "sentiment_model.joblib")
        self.is_pipeline = hasattr(self.model, "named_steps")
        self.vectorizer = None
        if not self.is_pipeline:
            self.vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
            # ผูก tokenizer กับฟังก์ชันในแพ็กเกจ (ไม่ต้องพึ่ง __main__ ต่อจากนี้)
            if getattr(self.vectorizer, "tokenizer", None) is not None:
                self.vectorizer.tokenizer = thai_tokenizer
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("โมเดลต้องมี predict_proba เพื่อคำนวณค่าความน่าจะเป็น")
        self.classes = [str(c) for c in self.model.classes_]
        self.version = model_version

    def _features(self, texts):
        if self.is_pipeline:
            return [" ".join(thai_tokenizer(t)) for t in texts]
        return self.vectorizer.transform(texts)

    def predict(self, texts):
        """คืน list ของ dict ต่อข้อความ: sentiment, prob_*, confidence, analyzable"""
        texts = ["" if t is None else str(t) for t in texts]
        proba = self.model.predict_proba(self._features(texts))
        idx = {c: i for i, c in enumerate(self.classes)}
        out = []
        for row in proba:
            p = {c: float(row[idx[c]]) for c in CLASSES}
            out.append({
                "sentiment": self.classes[int(np.argmax(row))],
                "prob_positive": round(p["positive"], 4),
                "prob_neutral": round(p["neutral"], 4),
                "prob_negative": round(p["negative"], 4),
                "confidence": round(float(row.max()), 4),
                "analyzable": "no" if row.max() - row.min() < UNANALYZABLE_SPREAD else "yes",
            })
        return out

    # ---- น้ำหนักคำจาก Naive Bayes (ใช้สร้างคำความรู้สึกเสริม) ----
    def word_polarity_scores(self):
        """log P(คำ|positive) - log P(คำ|negative) ของทุกคำเดี่ยวใน vocabulary"""
        vec = self.model.named_steps["tfidf"] if self.is_pipeline else self.vectorizer
        clf = self.model.named_steps["clf"] if self.is_pipeline else self.model
        if not hasattr(clf, "feature_log_prob_"):
            return {}
        lp = clf.feature_log_prob_
        pos, neg = self.classes.index("positive"), self.classes.index("negative")
        return {term: float(lp[pos, j] - lp[neg, j])
                for term, j in vec.vocabulary_.items() if " " not in term}
