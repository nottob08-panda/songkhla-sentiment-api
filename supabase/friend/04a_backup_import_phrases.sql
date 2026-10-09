-- =====================================================================
-- ขั้น A: สำรองวลีเดิม (lexicon) ของ reviews_import ก่อนแทนด้วยผล LLM v9.3
-- วิธีใช้: Supabase ของเพื่อน > SQL Editor > วาง > Run  (รันซ้ำได้: ถ้ามีตารางสำรองแล้วจะไม่ทับ)
-- =====================================================================
create table if not exists public.backup_reviews_import_phrases_20261009 as
select "reviewId", sentiment_text, positive_text, negative_text, n_sentiment_terms
  from public.reviews_import;

-- ตารางสำรองไม่ต้องให้หน้าเว็บอ่าน
alter table public.backup_reviews_import_phrases_20261009 enable row level security;

select count(*) as backed_up_rows from public.backup_reviews_import_phrases_20261009;   -- ควรได้ 7185
