# Sentiment API สำหรับเว็บไซต์รีวิวสถานที่ท่องเที่ยวสงขลา

วิเคราะห์รีวิวใหม่อัตโนมัติด้วยโมเดลเดียวกับในรายงาน (TF-IDF + Naive Bayes) แล้วบันทึกผลลง Supabase

## ภาพรวมการทำงาน

```
[เว็บ Vite] --1. INSERT รีวิว (status = pending)--> [Supabase: reviews]
     ^                                                   |
     |                                     2. Database Webhook (อัตโนมัติ)
     |                                                   v
     |                                  [API นี้ บน Render (ฟรี)]
     |                                    ตรวจภาษา -> แปลเป็นไทย -> TF-IDF + NB
     |                                    -> สกัดวลี -> เขียนผลกลับ (status = done)
     |                                                   |
     +----- 3. Realtime: แสดงผลทันทีที่แถวอัปเดต <---------+

[GitHub Actions ทุก 15 นาที] วิเคราะห์รีวิวที่ค้าง pending/failed
[cron-job.org ทุก 10 นาที]   ping /health กันเซิร์ฟเวอร์หลับ
```

| ส่วน | ไฟล์ |
|---|---|
| โค้ดหลัก (ตัดคำ, โมเดล, สกัดวลี, แปลภาษา) | `sentiment_core/` |
| พจนานุกรมสกัดวลี (แก้ได้โดยไม่แตะโค้ด) | `sentiment_core/resources/lexicon.json` |
| API | `api/app.py` |
| ตั้งค่าโฮสต์ Render | `render.yaml` |
| การจับคู่ผลลัพธ์กับคอลัมน์ในตาราง | `api/column_map.json` |
| ไฟล์โมเดล (จาก notebook 01) | `model/` |
| SQL สำหรับ Supabase | `supabase/01_setup_sentiment_pipeline.sql` |
| งานตามเวลา | `.github/workflows/process-pending.yml` |
| ตัวอย่างโค้ดฝั่งเว็บ | `web/reviewClient.js` |
| ทดสอบ | `tests/` |
| เครื่องมือเสนอคำใหม่ให้พจนานุกรม | `tools/suggest_lexicon_words.py` |

**ผลการทดสอบก่อนส่งมอบ**
- ผลซ้ำกับข้อมูลเดิมทั้ง 7,185 รีวิว **ตรงกันทุกแถว** (sentiment, ค่าความน่าจะเป็น, วลี)
- เวลาวิเคราะห์ต่อรีวิว (ไม่รวมแปลภาษา) ประมาณ 3 มิลลิวินาที
- หน่วยความจำคงที่ประมาณ 360 MB แม้รับ 2,300 คำขอต่อเนื่อง (Render ฟรีให้ 512 MB)
- SQL ทดสอบบน PostgreSQL 16 กับข้อมูลจริง: view สรุปรายสถานที่ตรงกับไฟล์ attractions เดิมทุกสถานที่

---

## ขั้นตอนติดตั้ง

### ขั้นที่ 1: ตั้งค่าฐานข้อมูล (Supabase)

1. Supabase Dashboard > **SQL Editor** > วางเนื้อหา `supabase/01_setup_sentiment_pipeline.sql` > **Run**
2. สิ่งที่ได้
   - คอลัมน์ใหม่ใน `reviews`: `analysis_status`, `analysis_error`, `model_version`, `analyzed_at`
     (รีวิวเดิมทั้งหมดถูกตั้งเป็น `done`)
   - `reviewId` สร้างอัตโนมัติต่อเนื่องจาก **R007202** (R007186–R007201 ถูกใช้แล้วกับรีวิวที่ตัดออกตอนทำความสะอาดข้อมูล)
   - `publishedAtDate` ใส่เวลาปัจจุบันให้อัตโนมัติ
   - รีวิวที่เพิ่มจากเว็บจะถูกล้างผลวิเคราะห์และตั้งเป็น `pending` เสมอ (กันการส่งผลปลอม)
   - เปิด Realtime ให้ตาราง `reviews`
   - view `place_sentiment_summary` = สรุปรายสถานที่ที่อัปเดตเองเมื่อมีรีวิวใหม่
   - view `reviews_without_phrases` = รีวิวที่สกัดวลีไม่เจอ (ใช้ปรับปรุงพจนานุกรม)

> SQL สมมติว่าตาราง `attractions` มีคอลัมน์ `attraction_id, attraction_name, category_id`
> ถ้าชื่อต่างจากนี้ให้แก้ในส่วนที่ 5 ของไฟล์ SQL

### ขั้นที่ 2: สร้าง API บน Render (ฟรี ไม่ต้องใช้บัตรเครดิต)

> **ทำไมไม่ใช้ Hugging Face Spaces:** ตั้งแต่กลางปี 2026 บัญชีฟรีใหม่ใช้ได้แค่ ZeroGPU
> ซึ่งออกแบบสำหรับงาน GPU ส่วน CPU Basic และ Docker ต้องสมัคร PRO

1. อัปโหลดโฟลเดอร์นี้ทั้งหมดขึ้น **GitHub repository** (ใช้ repo เดียวกับขั้นที่ 4 ได้)
2. สมัคร https://render.com ด้วยบัญชี GitHub
3. **New > Blueprint** > เลือก repo > Render อ่าน `render.yaml` แล้วตั้งค่าให้เอง
4. กรอกค่าความลับที่ระบบถาม

| ชื่อ | ค่า |
|---|---|
| `SUPABASE_URL` | Supabase > Project Settings > API > Project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase > Project Settings > API > `service_role` key (**ห้ามใส่ในโค้ดเว็บ**) |
| `WEBHOOK_SECRET` | ตั้งเองเป็นข้อความสุ่มยาวๆ เช่น จาก https://www.uuidgenerator.net |
| `TYPHOON_API_KEY` | API key ฟรีจาก Typhoon (ดูวิธีขอด้านล่าง) ใช้แปลรีวิวต่างภาษา **ควรใส่** |
| `MYMEMORY_EMAIL` | (ไม่บังคับ) อีเมล เพิ่มโควตาแปลภาษาสำรองเป็น 50,000 ตัวอักษร/วัน |

**วิธีขอ Typhoon API key (ฟรี):** เข้า https://playground.opentyphoon.ai > สมัคร/เข้าสู่ระบบ >
เมนู API Key > สร้าง key (ขึ้นต้นด้วย `sk-`) > คัดลอกมาใส่

> **ทำไมต้องใช้ Typhoon:** บริการแปลฟรีแบบไม่ใช้ key (Google, MyMemory) นับโควตาตาม IP
> เซิร์ฟเวอร์ฟรีอย่าง Render ใช้ IP ร่วมกับผู้ใช้คนอื่นจำนวนมาก จึงมักโดนบล็อกตั้งแต่คำขอแรก (error 429)
> ส่วน Typhoon นับโควตาตาม key ของเราเอง (5 คำขอ/วินาที, 200 คำขอ/นาที) จึงไม่ติดปัญหานี้
> Google และ MyMemory ยังถูกใช้เป็นตัวสำรองเมื่อ Typhoon ไม่ตอบ

5. รอ Build เสร็จ (ครั้งแรกประมาณ 3–5 นาที) แล้วเปิด
   ```
   https://<ชื่อ service>.onrender.com/health
   ```
   ควรได้ `{"status":"ok","model_version":"nb-v1","lexicon_version":"lexicon-1","database":true,"typhoon_translation":true}`
   และหน้า `/docs` ใช้ทดลองเรียก API ทุกตัวจากเบราว์เซอร์

ถ้า Build ฟ้องเรื่องเวอร์ชัน Python ให้แก้ `PYTHON_VERSION` ใน `render.yaml` เป็นเวอร์ชัน 3.12 ที่ Render รองรับ

### ขั้นที่ 3: ตั้ง Database Webhook (ให้ฐานข้อมูลเรียก API เองเมื่อมีรีวิวใหม่)

Supabase Dashboard > **Database > Webhooks > Create a new hook**

| ช่อง | ค่า |
|---|---|
| Table | `reviews` |
| Events | **Insert** เท่านั้น |
| Type | HTTP Request |
| Method | POST |
| URL | `https://<ชื่อ service>.onrender.com/v1/webhook/reviews` |
| HTTP Headers | `x-webhook-secret` = ค่าเดียวกับ `WEBHOOK_SECRET` |
| Timeout | 5000 ms |

API ตอบกลับทันที (202) แล้ววิเคราะห์เบื้องหลัง จึงไม่ติดเวลารอของ webhook

### ขั้นที่ 4: งานตามเวลา

**4.1 กันเซิร์ฟเวอร์หลับ (cron-job.org)** — Render ฟรีจะหลับเมื่อไม่มีคำขอ 15 นาที และใช้เวลาตื่นประมาณ 1 นาที

1. สมัคร https://cron-job.org (ฟรี)
2. Create cronjob > URL `https://<ชื่อ service>.onrender.com/health` > ทุก **10 นาที**

ชั่วโมงฟรี 750 ชั่วโมง/เดือนของ Render พอให้เปิด 1 service ได้ตลอดเดือน
(หมายเหตุ: Render ไม่ได้รับรองวิธี ping นี้อย่างเป็นทางการ ถ้าวันหนึ่งใช้ไม่ได้ ระบบยังทำงานต่อได้ แค่ผลจะขึ้นช้าลง)

**4.2 เก็บตกรีวิวที่ค้าง (GitHub Actions)**

1. อัปโหลดโฟลเดอร์ `.github/workflows/` ไว้ใน repo
2. Settings > Secrets and variables > Actions > เพิ่ม `SENTIMENT_API_URL` และ `WEBHOOK_SECRET`
3. แท็บ Actions > process-pending-reviews > **Run workflow** เพื่อทดสอบ

ทำงานทุก 15 นาที วิเคราะห์รีวิวที่ webhook พลาด (เช่น มาตอนเซิร์ฟเวอร์หลับ) หรือแปลภาษาไม่สำเร็จ

### ขั้นที่ 5: เชื่อมเว็บไซต์

ดู `web/reviewClient.js`
- `submitReview({attractionId, stars, text})` บันทึกรีวิวแล้วรอผล (ค่าเริ่มต้นรอไม่เกิน 8 วินาที)
- `getPlaceSummary(attractionId)` อ่านสรุปรายสถานที่จาก view

---

## ผลลัพธ์ที่บันทึกลงตาราง

| คอลัมน์ | ความหมาย |
|---|---|
| `processed_text` | ข้อความไทยที่ใช้วิเคราะห์ (ภาษาอื่นถูกแปลเป็นไทย) |
| `originalLanguage` | ภาษาต้นฉบับ เช่น `th`, `en`, `zh-Hans` |
| `sentiment`, `prob_*`, `confidence` | ผลโมเดล |
| `analyzable` | `no` = วิเคราะห์ไม่ได้ (ไม่รู้จักคำ หรือมีแต่อีโมจิ) |
| `sentiment_text`, `positive_text`, `negative_text`, `n_sentiment_terms` | วลีความรู้สึก |
| `analysis_status` | `pending` / `done` / `failed` |
| `analysis_error` | สาเหตุเมื่อ `failed` |
| `model_version` | เช่น `nb-v1/lexicon-1` |
| `analyzed_at` | เวลาที่วิเคราะห์ |

---

## การดูแลระบบในอนาคต

**เปลี่ยนชื่อคอลัมน์/ตาราง** — แก้ `api/column_map.json` (ไม่ต้องแก้โค้ด) และถ้าเป็นคอลัมน์ผลวิเคราะห์
ให้แก้ฟังก์ชัน `reviews_reset_analysis()` ในไฟล์ SQL ด้วย ถ้าไม่อยากบันทึกผลใด ให้ใส่ `null` ใน column_map

**สลับวิธีสกัดวลี (พจนานุกรม ↔ LLM)**
- Render → Environment → `PHRASE_METHOD` = `llm` และ `LLM_PROMPT` = `v9` (llm-typhoon-v9.3) แล้ว Save (Render deploy ใหม่เอง) ; ถ้า Typhoon ล่ม ระบบถอยไปใช้ lexicon เอง
- `llm` ใช้ Typhoon API สกัดวลี ครอบคลุมคำนอกพจนานุกรม ถ้า Typhoon ล่ม/โควตาเต็ม จะถอยไปใช้พจนานุกรมให้อัตโนมัติ
- ดูได้ว่าแถวไหนใช้วิธีอะไรจาก `model_version`: `nb-v1/llm-typhoon-v8.1` หรือ `nb-v1/lexicon-1`
- โหมด `llm` บันทึก 2 จังหวะ: sentiment ก่อน (สถานะยัง `pending`) แล้ววลี + `done` ตามมาอีก 1–2 วินาที
  หน้าเว็บใช้ `submitReview(..., { onProgress })` แสดงผลบวก/ลบก่อนได้ (ดู `web-app/src/main.js`)
- ก่อนเปิดใช้ ให้วัดผลด้วย `python tools/compare_phrase_methods.py reviews.csv --sample 30`
- สกัดวลีด้วย LLM ให้รีวิวทั้งชุด (มี checkpoint หยุด/ทำต่อได้): `python tools/extract_all_phrases.py reviews.csv --workers 2`

**เพิ่มคำในพจนานุกรมสกัดวลี**
1. Export view `reviews_without_phrases` เป็น CSV
2. `python tools/suggest_lexicon_words.py reviews_without_phrases.csv` ได้รายการคำที่น่าสนใจ
   (**ต้องคัดด้วยตา** เพราะหลายคำเป็นชื่อสถานที่/หัวข้อ ไม่ใช่ความรู้สึก)
3. เพิ่มคำใน `lexicon.json` และเปลี่ยน `"version"` เช่น `lexicon-2` แล้วอัปโหลดใหม่
4. (ถ้าต้องการ) วิเคราะห์รีวิวเก่าซ้ำ:
   `update reviews set analysis_status = 'pending' where model_version <> 'nb-v1/lexicon-2';`
   งานตามเวลาจะทยอยประมวลผลให้

**เปลี่ยนโมเดล (เทรนใหม่)** — แทนไฟล์ใน `model/`, ตั้ง Variable `MODEL_VERSION` เช่น `nb-v2`,
อัปโหลดใหม่ แล้วตั้ง `pending` กับแถวรุ่นเก่าเหมือนข้อ 4 ด้านบน
ถ้าโมเดลใหม่บันทึกด้วย scikit-learn เวอร์ชันอื่น ให้แก้ `requirements.txt` ให้ตรงกัน

**รันบนเครื่องตัวเอง**
```
pip install -r requirements.txt
uvicorn api.app:app --port 8000      # เปิด http://localhost:8000/docs
```

**ตรวจว่าผลไม่เพี้ยนหลังแก้ไข**
```
pip install -r requirements.txt pytest
REVIEWS_CSV=reviews.csv pytest tests
```
`test_parity.py` เทียบกับข้อมูลเดิมทุกแถว (ถ้าตั้งใจแก้พจนานุกรม ผลวลีจะเปลี่ยนตามปกติ)

---

**ถ้าวันหน้าย้ายไปโฮสต์อื่น** — ใช้คำสั่ง `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
กับโฮสต์ Python ทั่วไป หรือใช้ `Dockerfile` กับโฮสต์ที่รัน Docker ได้

## ข้อจำกัดที่ควรรู้

- **การแปลภาษา** ใช้ Typhoon เป็นหลัก (สำรองด้วย Google และ MyMemory) คำแปลจาก LLM อาจต่างจาก
  คำแปลของ Google Maps ที่ใช้ในข้อมูลเทรนเล็กน้อย จึงอาจกระทบความแม่นยำของรีวิวต่างภาษา
  ถ้าแปลไม่สำเร็จทุกตัว ระบบจะบันทึก `failed` แล้วลองใหม่อัตโนมัติ
  (การเรียก API จริงทดสอบในสภาพแวดล้อมที่พัฒนาไม่ได้ ทดสอบด้วยตัวจำลองแทน ควรทดสอบรีวิวภาษาอังกฤษหลัง deploy)
- **เวลาตอบสนอง** ปกติเห็นผลภายใน 1–3 วินาที (รีวิวต่างภาษาขึ้นกับความเร็วบริการแปล)
  ถ้าเซิร์ฟเวอร์หลับ webhook จะหมดเวลารอ รีวิวจะค้างเป็น pending แล้ว GitHub Actions
  วิเคราะห์ให้ภายใน 15 นาที รีวิวไม่หาย หน้าเว็บจะแสดง "กำลังวิเคราะห์" แทน
- **CPU ของ Render ฟรีมีแค่ 0.1 CPU** การวิเคราะห์ต่อรีวิวจึงช้ากว่าที่วัดบนเครื่องทดสอบ
  (คาดว่าหลักสิบมิลลิวินาที ยังต่ำกว่า 3 วินาทีมาก) ควรวัดซ้ำหลัง deploy
- **โมเดลไม่เรียนรู้เองจากรีวิวใหม่** ต้องเทรนใหม่แยกต่างหาก
- **การสกัดวลีจับได้เฉพาะคำในพจนานุกรม** ปรับปรุงได้ตามหัวข้อ "เพิ่มคำในพจนานุกรม"
