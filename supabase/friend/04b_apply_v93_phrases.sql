-- =====================================================================
-- ขั้น C: แทนวลีใน reviews_import ด้วยผลสกัด LLM v9.3 (หลังนำเข้า CSV เป็นตาราง phrases_v93_import แล้ว)
-- แก้เฉพาะ 4 คอลัมน์วลี: sentiment_text, positive_text, negative_text, n_sentiment_terms
-- ไม่แตะ sentiment / prob_* / ดาว / ข้อความรีวิว (โมเดลจำแนกยังเป็นตัวเดิม nb-v1)
-- ทั้งไฟล์อยู่ใน transaction: ถ้าตรวจไม่ผ่าน จะยกเลิกทั้งหมด ไม่มีอะไรเปลี่ยน
-- =====================================================================
begin;

do $$
declare
  n_stage   integer;
  n_match   integer;
  n_backup  integer;
begin
  select count(*) into n_backup from public.backup_reviews_import_phrases_20261009;
  if n_backup = 0 then
    raise exception 'ยังไม่ได้สำรอง: รัน 04a_backup_import_phrases.sql ก่อน';
  end if;

  select count(*) into n_stage from public.phrases_v93_import;
  select count(*) into n_match
    from public.phrases_v93_import p
    join public.reviews_import ri on ri."reviewId" = p."reviewId";

  raise notice 'แถวใน CSV: %, ตรงกับ reviews_import: %', n_stage, n_match;
  if n_stage <> 7185 or n_match <> 7185 then
    raise exception 'จำนวนแถวไม่ครบ (CSV %, ตรงกัน %) -- ตรวจการนำเข้า CSV อีกครั้ง', n_stage, n_match;
  end if;
end $$;

update public.reviews_import ri
   set sentiment_text    = nullif(trim(p.sentiment_text::text), ''),
       positive_text     = nullif(trim(p.positive_text::text), ''),
       negative_text     = nullif(trim(p.negative_text::text), ''),
       n_sentiment_terms = coalesce(nullif(trim(p.n_sentiment_terms::text), ''), '0')
  from public.phrases_v93_import p
 where ri."reviewId" = p."reviewId";

commit;

-- ตรวจผล: จำนวนรีวิวที่มีวลี (ก่อน = lexicon, หลัง = v9.3)
select 'before (lexicon)' as version,
       count(*) filter (where coalesce(sentiment_text, '') <> '') as reviews_with_phrases
  from public.backup_reviews_import_phrases_20261009
union all
select 'after (llm v9.3)',
       count(*) filter (where coalesce(sentiment_text, '') <> '')
  from public.reviews_import;

-- -------------------------------------------------------------------------------------
-- หลังตรวจแล้วพอใจ ลบตารางชั่วคราวได้ (ไม่บังคับ):   drop table public.phrases_v93_import;
--
-- ถ้าต้องการคืนวลีเดิม (lexicon):
--   update public.reviews_import ri
--      set sentiment_text = b.sentiment_text, positive_text = b.positive_text,
--          negative_text = b.negative_text, n_sentiment_terms = b.n_sentiment_terms
--     from public.backup_reviews_import_phrases_20261009 b
--    where ri."reviewId" = b."reviewId";
-- -------------------------------------------------------------------------------------
