"""ขั้นตอนวิเคราะห์รีวิว 1 รายการ (ใช้ทั้งใน API และ script)

ข้อความรีวิว -> ตรวจภาษา -> (แปลเป็นไทย) -> processed_text
            -> โมเดล (TF-IDF + Naive Bayes) -> sentiment, prob_*, confidence, analyzable
            -> สกัดวลี -> sentiment_text, positive_text, negative_text, n_sentiment_terms
               phrase_method="lexicon" : พจนานุกรม (เร็ว ผลคงที่)
               phrase_method="llm"     : Typhoon LLM (ครอบคลุมกว่า) ถ้า Typhoon ใช้ไม่ได้ จะถอยไปใช้พจนานุกรมเอง

ผลลัพธ์ใช้ "ชื่อกลาง" (ไม่ผูกกับชื่อคอลัมน์ในฐานข้อมูล) การจับคู่กับคอลัมน์อยู่ใน api/column_map.json
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .extractor import PhraseExtractor
from .llm_extractor import LLMExtractionError, LLMPhraseExtractor
from .language import TranslationError, detect_language, has_letters, translate_to_thai
from .model import SentimentModel


@dataclass
class AnalysisResult:
    status: str                      # "done" | "failed" | "scored" (มี sentiment แล้ว รอวลี -- ใช้กับการบันทึก 2 จังหวะ)
    fields: dict = field(default_factory=dict)
    error: str | None = None


class ReviewAnalyzer:
    def __init__(self, model_dir, model_version="nb-v1", translate=True, mymemory_email=None,
                 translator=None, typhoon_api_key=None, typhoon_model=None, phrase_method="lexicon",
                 llm_extractor=None):
        self.model = SentimentModel(model_dir, model_version)
        self.extractor = PhraseExtractor()
        self.translate = translate
        self.mymemory_email = mymemory_email
        self.typhoon_api_key = typhoon_api_key
        self.typhoon_model = typhoon_model
        self._translator = translator or translate_to_thai   # เปลี่ยนได้ (เช่น ใช้ตัวจำลองตอนทดสอบ)
        if phrase_method not in ("lexicon", "llm"):
            raise ValueError(f"phrase_method must be 'lexicon' or 'llm', got {phrase_method!r}")
        if phrase_method == "llm" and not (llm_extractor or typhoon_api_key):
            raise ValueError("phrase_method='llm' needs TYPHOON_API_KEY")
        self.phrase_method = phrase_method
        self.llm = llm_extractor or (LLMPhraseExtractor(typhoon_api_key, typhoon_model)
                                     if phrase_method == "llm" else None)

    @property
    def versions(self):
        return {"model_version": self.model.version, "lexicon_version": self.extractor.lexicon_version,
                "phrase_method": self.phrase_method}

    def extract_phrases(self, text):
        """คืน (ผลวลี, รุ่นของวิธีที่ใช้จริง)"""
        if self.llm is not None:
            try:
                return self.llm.analyze(text), self.llm.version
            except LLMExtractionError:
                pass                                 # Typhoon ล่ม/โควตาเต็ม -> ใช้พจนานุกรมแทน ไม่ให้รีวิวค้าง
        return self.extractor.analyze(text), self.extractor.lexicon_version

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

    def analyze(self, text, with_phrases=True):
        """with_phrases=False: คืนสถานะ "scored" (มี sentiment แต่ยังไม่สกัดวลี) แล้วเรียก complete() ต่อ
        ใช้กับ LLM ที่ช้ากว่า เพื่อให้หน้าเว็บเห็นผลบวก/ลบได้ก่อน"""
        try:
            processed, lang, provider = self.prepare_text(text)
        except TranslationError as e:
            return AnalysisResult("failed", {"originalLanguage": detect_language(text)},
                                  f"translation failed: {e}")

        fields = {"processed_text": processed, "originalLanguage": lang, "translation_provider": provider}
        if not has_letters(processed):
            # กฎเดียวกับขั้นทำความสะอาดข้อมูล: ข้อความที่ไม่มีตัวอักษรเลย (อีโมจิ/สัญลักษณ์ล้วน) วิเคราะห์ไม่ได้
            fields.update({"sentiment": None, "prob_positive": None, "prob_neutral": None,
                           "prob_negative": None, "confidence": None, "analyzable": "no",
                           "sentiment_text": None, "positive_text": None, "negative_text": None,
                           "n_sentiment_terms": 0})
            return self._finish(fields, self.extractor.lexicon_version)
        fields.update(self.model.predict([processed])[0])
        result = AnalysisResult("scored", fields)
        return self.complete(result) if with_phrases else result

    def complete(self, result):
        """จังหวะที่ 2: สกัดวลีจาก processed_text แล้วปิดงานเป็น done"""
        phrases, phrase_version = self.extract_phrases(result.fields["processed_text"])
        return self._finish({**result.fields, **phrases}, phrase_version)

    def _finish(self, fields, phrase_version):
        fields.update({"model_version": f"{self.model.version}/{phrase_version}",
                       "analyzed_at": datetime.now(timezone.utc).isoformat()})
        return AnalysisResult("done", fields)
