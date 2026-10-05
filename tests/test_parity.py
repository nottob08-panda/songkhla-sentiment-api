"""ตรวจว่าโค้ดในแพ็กเกจให้ผลเหมือนผลเดิมใน reviews.csv ทุกแถว

รัน:  REVIEWS_CSV=path/to/reviews.csv pytest tests/test_parity.py
ใช้ตรวจทุกครั้งที่แก้โค้ด หรือย้ายเซิร์ฟเวอร์/อัปเดตไลบรารี
(ถ้าตั้งใจแก้พจนานุกรม ผล sentiment_text จะเปลี่ยนตามปกติ)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentiment_core.extractor import PhraseExtractor
from sentiment_core.model import SentimentModel

ROOT = Path(__file__).resolve().parent.parent
CSV = os.getenv("REVIEWS_CSV")
pytestmark = pytest.mark.skipif(not CSV, reason="set REVIEWS_CSV to run the parity test")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV)


def test_model_parity(df):
    res = pd.DataFrame(SentimentModel(ROOT / "model").predict(df["processed_text"].fillna("")))
    assert (res["sentiment"].values == df["sentiment"].values).all()
    assert (res["analyzable"].values == df["analyzable"].values).all()
    for c in ["prob_positive", "prob_neutral", "prob_negative", "confidence"]:
        assert np.abs(res[c].values - df[c].values).max() < 1e-4, c


def test_phrase_parity(df):
    ex = PhraseExtractor()
    res = pd.DataFrame([ex.analyze(t) for t in df["processed_text"].fillna("")])
    for c in ["sentiment_text", "positive_text", "negative_text"]:
        assert (res[c].fillna("") == df[c].fillna("")).all(), c
    assert (res["n_sentiment_terms"] == df["n_sentiment_terms"]).all()
