// ตัวอย่างโค้ดฝั่งเว็บ (Vite + JavaScript)
// ติดตั้ง: npm install @supabase/supabase-js
// ตั้งค่าในไฟล์ .env ของโปรเจกต์เว็บ:
//   VITE_SUPABASE_URL=https://xxxx.supabase.co
//   VITE_SUPABASE_ANON_KEY=<anon public key>      <- ใช้ anon key เท่านั้น ห้ามใช้ service_role key ในเว็บ
//
// เว็บไม่ได้เรียกโมเดลเอง: แค่บันทึกรีวิว แล้วรอผลที่ API เขียนกลับมาในแถวเดียวกัน (Realtime)

import { createClient } from "@supabase/supabase-js";

export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY
);

const RESULT_COLUMNS =
  "reviewId, analysis_status, sentiment, prob_positive, prob_neutral, prob_negative, " +
  "confidence, analyzable, sentiment_text, positive_text, negative_text";

/**
 * ส่งรีวิวใหม่ แล้วรอผลวิเคราะห์
 * @param {{attractionId: string, stars: number, text: string}} review
 * @param {{timeoutMs?: number}} options  รอผลนานสุดกี่มิลลิวินาที (เกินแล้วคืนสถานะ pending)
 * @returns {Promise<object>} แถวรีวิวพร้อมผล หรือ { reviewId, analysis_status: "pending" }
 */
export async function submitReview({ attractionId, stars, text }, { timeoutMs = 8000 } = {}) {
  // 1) บันทึกรีวิว (reviewId, วันที่, สถานะ pending ฐานข้อมูลสร้างให้เอง)
  const { data, error } = await supabase
    .from("reviews")
    .insert({ attraction_id: attractionId, stars, text })
    .select("reviewId")
    .single();
  if (error) throw error;
  const reviewId = data.reviewId;

  // 2) รอผลแบบ Realtime
  return new Promise((resolve) => {
    let finished = false;
    const finish = (row) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      supabase.removeChannel(channel);
      resolve(row);
    };

    const timer = setTimeout(
      () => finish({ reviewId, analysis_status: "pending" }), // ยังไม่เสร็จ: ผลจะขึ้นเองภายหลัง
      timeoutMs
    );

    const channel = supabase
      .channel(`review-${reviewId}`)
      .on(
        "postgres_changes",
        { event: "UPDATE", schema: "public", table: "reviews", filter: `reviewId=eq.${reviewId}` },
        (payload) => {
          if (payload.new.analysis_status !== "pending") finish(payload.new);
        }
      )
      .subscribe(async (status) => {
        if (status !== "SUBSCRIBED") return;
        // กันกรณีวิเคราะห์เสร็จก่อนที่จะเริ่มฟัง: อ่านแถวซ้ำ 1 ครั้ง
        const { data: row } = await supabase
          .from("reviews").select(RESULT_COLUMNS).eq("reviewId", reviewId).single();
        if (row && row.analysis_status !== "pending") finish(row);
      });
  });
}

/** สรุป sentiment รายสถานที่ (คำนวณสดจาก view ในฐานข้อมูล) */
export async function getPlaceSummary(attractionId) {
  let query = supabase.from("place_sentiment_summary").select("*");
  if (attractionId) query = query.eq("attraction_id", attractionId);
  const { data, error } = await query;
  if (error) throw error;
  return data;
}

/* ตัวอย่างการใช้งานในหน้าเว็บ
const result = await submitReview({ attractionId: "A001", stars: 5, text: "วิวสวยมาก อาหารอร่อย" });
if (result.analysis_status === "done") {
  // result.sentiment = "positive" | "neutral" | "negative" (null ถ้าวิเคราะห์ไม่ได้ เช่น มีแต่อีโมจิ)
  // result.sentiment_text = "วิวสวย อาหารอร่อย"
} else if (result.analysis_status === "failed") {
  // เช่น บริการแปลภาษาล่ม: ระบบจะลองใหม่อัตโนมัติภายใน 15 นาที
} else {
  // pending: แสดง "กำลังวิเคราะห์" ผลจะปรากฏเมื่อโหลดหน้าใหม่
}
*/
