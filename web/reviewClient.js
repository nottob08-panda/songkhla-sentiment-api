// ติดต่อฐานข้อมูล Supabase จากหน้าเว็บ
// ค่าเชื่อมต่ออยู่ในไฟล์ .env (ดู .env.example) -- ใช้ anon key เท่านั้น ห้ามใช้ service_role key ในเว็บ
//
// เว็บไม่ได้เรียกโมเดลเอง: แค่บันทึกรีวิว แล้วรอผลที่ API เขียนกลับมาในแถวเดียวกัน (Realtime)

import { createClient } from "@supabase/supabase-js";

const URL = import.meta.env.VITE_SUPABASE_URL;
const KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const isConfigured = Boolean(URL && KEY && !URL.includes("xxxx"));
export const supabase = isConfigured ? createClient(URL, KEY) : null;

const RESULT_COLUMNS =
  "reviewId, attraction_id, stars, text, publishedAtDate, analysis_status, sentiment, " +
  "prob_positive, prob_neutral, prob_negative, confidence, analyzable, " +
  "sentiment_text, positive_text, negative_text, processed_text, originalLanguage";

/**
 * ส่งรีวิวใหม่ แล้วรอผลวิเคราะห์
 * API อาจบันทึกผล 2 จังหวะ (โหมด LLM): ได้ sentiment ก่อน (สถานะยังเป็น pending) แล้ววลีตามมา (สถานะ done)
 * @param {{attractionId: string, stars: number, text: string}} review
 * @param {{timeoutMs?: number, onProgress?: (row: object) => void}} options
 *   timeoutMs  รอผลนานสุดกี่มิลลิวินาที
 *   onProgress เรียกเมื่อได้ sentiment แล้วแต่วลียังไม่มา (ใช้แสดงผลบวก/ลบก่อน)
 * @returns {Promise<object>} แถวที่ analysis_status เป็น done/failed
 *   หรือแถวที่มีแค่ sentiment (ถ้าวลียังไม่มาภายในเวลา) หรือ { reviewId, analysis_status: "pending" }
 */
export async function submitReview({ attractionId, stars, text }, { timeoutMs = 10000, onProgress } = {}) {
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
    let partial = null;
    let channel = null;
    const finish = (row) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      clearInterval(poll);
      if (channel) supabase.removeChannel(channel);
      resolve(row);
    };
    const handle = (row) => {
      if (!row || finished) return;
      if (row.analysis_status !== "pending") return finish(row);
      if (row.sentiment && !partial) {       // จังหวะที่ 1: มี sentiment แล้ว วลียังไม่มา
        partial = row;
        if (onProgress) onProgress(row);
      }
    };
    const check = async () => {
      const { data: row } = await supabase
        .from("reviews").select(RESULT_COLUMNS).eq("reviewId", reviewId).maybeSingle();
      handle(row);
    };

    const timer = setTimeout(
      () => finish(partial || { reviewId, analysis_status: "pending" }), // ยังไม่เสร็จ: ผลจะขึ้นเองภายหลัง
      timeoutMs
    );
    // สำรอง: ถ้า Realtime ใช้ไม่ได้ ก็ยังอ่านซ้ำทุก 1.5 วินาที
    const poll = setInterval(check, 1500);

    channel = supabase
      .channel(`review-${reviewId}`)
      .on(
        "postgres_changes",
        { event: "UPDATE", schema: "public", table: "reviews", filter: `reviewId=eq.${reviewId}` },
        (payload) => handle(payload.new)
      )
      .subscribe((status) => {
        // กันกรณีวิเคราะห์เสร็จก่อนที่จะเริ่มฟัง: อ่านแถวซ้ำ 1 ครั้ง
        if (status === "SUBSCRIBED") check();
      });
  });
}

/** สรุป sentiment รายสถานที่ (คำนวณสดจาก view ในฐานข้อมูล) ไม่ระบุ id = ทุกสถานที่ */
export async function getPlaceSummary(attractionId) {
  let query = supabase.from("place_sentiment_summary").select("*");
  if (attractionId) query = query.eq("attraction_id", attractionId);
  const { data, error } = await query;
  if (error) throw error;
  return data;
}

/** รีวิวล่าสุดของสถานที่ */
export async function listReviews(attractionId, limit = 10) {
  const { data, error } = await supabase
    .from("reviews")
    .select(RESULT_COLUMNS)
    .eq("attraction_id", attractionId)
    .order("publishedAtDate", { ascending: false, nullsFirst: false })
    .limit(limit);
  if (error) throw error;
  return data;
}

/** หมวดหมู่สถานที่ (ถ้าอ่านไม่ได้ คืนค่าว่าง หน้าเว็บยังทำงานต่อได้) */
export async function listCategories() {
  const { data, error } = await supabase.from("category").select("category_id, category_name");
  return error ? [] : data;
}
