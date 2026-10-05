"""ตัวตัดคำที่ใช้ตอนเทรนโมเดล

ฟังก์ชันนี้ต้องให้ผลเหมือนตอนเทรนทุกตัวอักษร เพราะ TfidfVectorizer เก็บ "ชื่อ" ฟังก์ชันนี้ไว้ในไฟล์ .joblib
(ไม่ได้เก็บโค้ด) การแยกไว้ในโมดูลนี้ทำให้ทุกที่ (API, notebook, script) ใช้ตัวเดียวกัน
"""
from pythainlp.tokenize import word_tokenize


def thai_tokenizer(text):
    return word_tokenize(str(text), engine="newmm")
