# หน้าเว็บรีวิวสถานที่ท่องเที่ยว (Vite)

หน้าเว็บนี้ใช้เลือกสถานที่ ดูสรุป sentiment ส่งรีวิว (ภาษาใดก็ได้) แล้วเห็นผลวิเคราะห์ทันที

เว็บติดต่อ Supabase ด้วย **anon key** เท่านั้น และไม่ได้เรียกโมเดลเอง
เมื่อเว็บบันทึกรีวิวแล้ว Database Webhook จะเรียก API บน Render ให้ API เขียนผลกลับลงแถวเดิม จากนั้นเว็บรับผลผ่าน Realtime

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `index.html` | โครงหน้าเว็บ |
| `src/main.js` | การทำงานของหน้า (โหลดสถานที่ ฟอร์ม แสดงผล) |
| `src/reviewClient.js` | ฟังก์ชันคุยกับ Supabase (`submitReview`, `getPlaceSummary`, `listReviews`) |
| `src/style.css` | หน้าตา |
| `.env.example` | ตัวอย่างค่าเชื่อมต่อ (คัดลอกเป็น `.env`) |

## รันบนเครื่อง

```bash
npm install
cp .env.example .env     # แล้วแก้ค่าใน .env
npm run dev              # เปิด http://localhost:5173
```

## ขึ้นเว็บจริง (Render Static Site ฟรี ไม่หลับ)

ตั้งค่าดังนี้: Root Directory = `web-app`, Build Command = `npm install && npm run build`, Publish Directory = `dist`
และ Environment = `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`

ใส่ `?place=A001` ต่อท้าย URL เพื่อเปิดหน้าที่เลือกสถานที่ไว้แล้ว
