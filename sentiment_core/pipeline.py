"""ขั้นตอนวิเคราะห์รีวิว 1 รายการ (ใช้ทั้งใน API และ script)

ข้อความรีวิว -> ตรวจภาษา -> (แปลเป็นไทย) -> processed_text
            -> โมเดล (TF-IDF + Naive Bayes) -> sentiment, prob_*, confidence, analyzable
            -> สกัดวลี -> sentiment_text, positive_text, negative_text, n_sentiment_terms

ผลลัพธ์ใช้ "ชื่อกลาง" (ไม่ผูกกับชื่อคอลัมน์ในฐานข้อมูล) การจับคู่กับคอลัมน์อยู่ใน api/column_map.json
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .extractor import PhraseExtractor
from .language import TranslationError, detect_language, has_letters, translate_to_thai
from .model import SentimentModel


@dataclass
class AnalysisResult:
    status: str                      # "done" | "failed"
    fields: dict = field(default_factory=dict)
    error: str | None = None


class ReviewAnalyzer:
    def __init__(self, model_dir, model_version="nb-v1", translate=True, mymemory_email=None,
                 translator=None, typhoon_api_key=None, typhoon_model=None):
        self.model = SentimentModel(model_dir, model_version)
        self.extractor = PhraseExtractor()
        self.translate = translate
        self.mymemory_email = mymemory_email
        self.typhoon_api_key = typhoon_api_key
        self.typhoon_model = typhoon_model
        self._translator = translator or translate_to_thai   # เปลี่ยนได้ (เช่น ใช้ตัวจำลองตอนทดสอบ)

    @property
    def versions(self):
        return {"model_version": self.model.version, "lexicon_version": self.extractor.lexicon_version}

    def prepare_text(self, text):
        """คืน (processed_text, originalLanguage, ผู้ให้บริการแปล)"""
        text = (text or "").strip()
        lang = detect_language(text)
        if lang in (None, "th") or not has_letters(text) or not self.translate:
            return text, lang, None
        translated, provider = self._translator(text, lang, mymemory_email=self.mymemory_email,
                                                typhoon_api_key=self.typhoon_api_key,
                                                typhoon_model=self.typhoon_model)
        return translated, lang, provider

    def analyze(self, text):
        try:
            processed, lang, provider = self.prepare_text(text)
        except TranslationError as e:
            return AnalysisResult("failed", {"originalLanguage": detect_language(text)},
                                  f"translation failed: {e}")

        fields = {"processed_text": processed, "originalLanguage": lang}
        if not has_letters(processed):
            # กฎเดียวกับขั้นทำความสะอาดข้อมูล: ข้อความที่ไม่มีตัวอักษรเลย (อีโมจิ/สัญลักษณ์ล้วน) วิเคราะห์ไม่ได้
            fields.update({"sentiment": None, "prob_positive": None, "prob_neutral": None,
                           "prob_negative": None, "confidence": None, "analyzable": "no",
                           "sentiment_text": None, "positive_text": None, "negative_text": None,
                           "n_sentiment_terms": 0})
        else:
            fields.update(self.model.predict([processed])[0])
            fields.update(self.extractor.analyze(processed))
        fields.update({"model_version": f"{self.model.version}/{self.extractor.lexicon_version}",
                       "analyzed_at": datetime.now(timezone.utc).isoformat(),
                       "translation_provider": provider})
        return AnalysisResult("done", fields)
