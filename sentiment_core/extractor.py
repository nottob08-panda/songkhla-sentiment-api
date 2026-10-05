"""สกัดวลีแสดงความรู้สึก (ย้ายมาจาก notebook 03 โดยไม่เปลี่ยน logic)

พจนานุกรมอยู่ในไฟล์ resources/lexicon.json แก้/เพิ่มคำได้โดยไม่ต้องแก้โค้ด
เมื่อแก้พจนานุกรม ให้เปลี่ยน "version" ในไฟล์ด้วย เพื่อให้รู้ว่าวลีในฐานข้อมูลมาจากพจนานุกรมรุ่นไหน
"""
import json
import re
from pathlib import Path

from pythainlp.corpus.common import thai_words
from pythainlp.tokenize import word_tokenize
from pythainlp.util import dict_trie

RESOURCES = Path(__file__).parent / "resources"


class PhraseExtractor:
    def __init__(self, lexicon_path=RESOURCES / "lexicon.json"):
        lx = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
        self.lexicon_version = lx.get("version", "unknown")
        self.POS_SET, self.NEG_SET = set(lx["POSITIVE_WORDS"]), set(lx["NEGATIVE_WORDS"])
        self.LEX = self.POS_SET | self.NEG_SET
        self.NEGATIONS = lx["NEGATIONS"]
        self.NEG_T, self.PRE_T, self.POST_T = set(lx["NEGATIONS"]), set(lx["PRE_INTENSIFIERS"]), set(lx["POST_INTENSIFIERS"])
        self.FILL_T, self.ASP_T = set(lx["FILLERS"]), set(lx["ASPECTS"])
        self.SPLIT_SUFFIX = sorted(lx["POST_INTENSIFIERS"], key=len, reverse=True)
        self.ABUND_T, self.ABUND_ASP_T = set(lx["ABUNDANCE_WORDS"]), set(lx["ABUNDANCE_ASPECTS"])
        self.ASP_T |= self.ABUND_ASP_T
        self.ASPECTS_BY_LEN = sorted(self.ASP_T, key=len, reverse=True)
        self.NO_SPLIT = set(lx["NO_SPLIT"])
        self.GENERIC_ASPECTS = set(lx["GENERIC_ASPECTS"])
        self.COMPACT_NORMALIZE = lx["COMPACT_NORMALIZE"]
        self.TRIE = dict_trie((set(thai_words()) - set(lx["BAD_DICT_WORDS"])) | self.LEX | self.NEG_T
                              | self.PRE_T | self.POST_T | self.ASP_T | self.ABUND_T)


    # ------------------------------------------------------------------ ตัดคำ
    @staticmethod
    def clean_text(text):
        text = str(text).lower().replace('ๆ', '')
        text = re.sub(r'[^฀-๿a-z\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def split_compound(self, tok):
        if tok in self.LEX or tok in self.ASP_T or tok in self.NO_SPLIT:
            return [tok]
        for asp in self.ASPECTS_BY_LEN:
            rest = tok[len(asp):]
            if tok.startswith(asp) and rest and (rest in self.LEX or rest in self.ABUND_T
                                                 or self.split_compound(rest)[0] in self.LEX | self.NEG_T):
                return [asp] + self.split_compound(rest)
        for suf in self.SPLIT_SUFFIX:
            if tok.endswith(suf) and tok[:-len(suf)] in self.LEX:
                return [tok[:-len(suf)], suf]
        for neg in sorted(self.NEGATIONS, key=len, reverse=True):
            if tok.startswith(neg) and tok[len(neg):] in self.LEX:
                return [neg, tok[len(neg):]]
        return [tok]

    def tokenize(self, text):
        toks = word_tokenize(self.clean_text(text), custom_dict=self.TRIE, engine='newmm', keep_whitespace=False)
        out = []
        for t in toks:
            if t.strip():
                out.extend(self.split_compound(t))
        return out

    # ------------------------------------------------------------------ สกัดวลี
    def extract(self, text):
        """คืน list ของ dict: word, phrase, aspect, polarity, core, neg"""
        toks = self.tokenize(text)
        results = []
        for i, t in enumerate(toks):
            if t not in self.LEX and t not in self.ABUND_T:
                continue
            polarity = 'positive' if t in self.POS_SET or t in self.ABUND_T else 'negative'
            j, pre, neg, aspect = i - 1, '', '', ''
            if j >= 0 and toks[j] in self.PRE_T:
                pre = toks[j]; j -= 1
            if j >= 0 and toks[j] in self.NEG_T:
                neg = toks[j]; j -= 1
                if not pre and j >= 0 and toks[j] in self.PRE_T:
                    pre = toks[j]; j -= 1
            skipped = 0
            while j >= 0 and toks[j] in self.FILL_T and skipped < 3:
                j -= 1; skipped += 1
            if j >= 0 and toks[j] in self.ASP_T:
                aspect = toks[j]
            if t in self.ABUND_T and (aspect not in self.ABUND_ASP_T or neg):
                continue
            post, k = [], i + 1
            while k < len(toks) and toks[k] in self.POST_T and len(post) < 2:
                if not post or toks[k] != post[-1]:
                    post.append(toks[k])
                k += 1
            if neg:
                polarity = 'negative' if polarity == 'positive' else 'positive'
            neg_word = neg if neg in ('ไม่มี', 'ไร้') else ('ไม่' if neg else '')
            results.append({'word': neg_word + t, 'phrase': aspect + neg + pre + t + ''.join(post),
                            'aspect': aspect or None, 'polarity': polarity, 'core': t, 'neg': neg_word})
        return results


    # ------------------------------------------------------------------ ย่อเป็นข้อความสั้น
    def compact_term(self, r):
        core = self.COMPACT_NORMALIZE.get(r["core"], r["core"])
        aspect = r["aspect"] if r["aspect"] and r["aspect"] not in self.GENERIC_ASPECTS else ""
        return aspect + r["neg"] + core

    def compact_texts(self, extracted):
        all_terms, pos_terms, neg_terms = [], [], []
        for r in extracted:
            term = self.compact_term(r)
            if term not in all_terms:
                all_terms.append(term)
                (pos_terms if r["polarity"] == "positive" else neg_terms).append(term)
        return {"sentiment_text": " ".join(all_terms) or None,
                "positive_text": " ".join(pos_terms) or None,
                "negative_text": " ".join(neg_terms) or None,
                "n_sentiment_terms": len(all_terms)}

    def analyze(self, text):
        return self.compact_texts(self.extract(text))
