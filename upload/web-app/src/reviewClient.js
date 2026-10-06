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
 * @param {{attractionId: string, stars: number, text: string}} review
 * @param {{timeoutMs?: number}} options  รอผลนานสุดกี่มิลลิวินาที (เกินแล้วคืนสถานะ pending)
 * @returns {Promise<object>} แถวรีวิวพร้อมผล หรือ { reviewId, analysis_status: "pending" }
 */
export async function submitReview({ attractionId, stars, text }, { timeoutMs = 10000 } = {}) {
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
    let channel = null;
    const finish = (row) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      clearInterval(poll);
      if (channel) supabase.removeChannel(channel);
      resolve(row);
    };
    const check = async () => {
      const { data: row } = await supabase
        .from("reviews").select(RESULT_COLUMNS).eq("reviewId", reviewId).maybeSingle();
      if (row && row.analysis_status !== "pending") finish(row);
    };

    const timer = setTimeout(
      () => finish({ reviewId, analysis_status: "pending" }), // ยังไม่เสร็จ: ผลจะขึ้นเองภายหลัง
      timeoutMs
    );
    // สำรอง: ถ้า Realtime ใช้ไม่ได้ ก็ยังอ่านซ้ำทุก 2 วินาที
    const poll = setInterval(check, 2000);

    channel = supabase
      .channel(`review-${reviewId}`)
      .on(
        "postgres_changes",
        { event: "UPDATE", schema: "public", table: "reviews", filter: `reviewId=eq.${reviewId}` },
        (payload) => {
          if (payload.new.analysis_status !== "pending") finish(payload.new);
        }
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
