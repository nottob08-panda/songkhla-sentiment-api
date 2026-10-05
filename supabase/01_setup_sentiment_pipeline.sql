-- =====================================================================
-- ตั้งค่าฐานข้อมูลสำหรับวิเคราะห์ sentiment รีวิวใหม่อัตโนมัติ
-- วิธีใช้: Supabase Dashboard > SQL Editor > วางทั้งไฟล์ > Run
-- รันซ้ำได้ ไม่ทำให้ข้อมูลเดิมเสียหาย (idempotent)
--
-- สมมติฐานเรื่องโครงสร้าง (แก้ได้ถ้าชื่อไม่ตรง):
--   reviews     : ตามโครงสร้างปัจจุบัน ("reviewId", attraction_id, stars, text, ...)
--   attractions : attraction_id, attraction_name, category_id
--   category    : category_id, category_name
-- =====================================================================
begin;

-- ---------------------------------------------------------------------
-- 1) คอลัมน์ควบคุมการวิเคราะห์
--    analysis_status : pending = รอวิเคราะห์ | done = เสร็จ | failed = ไม่สำเร็จ (ระบบจะลองใหม่)
--    analysis_error  : สาเหตุที่ไม่สำเร็จ (เช่น บริการแปลภาษาล่ม)
--    model_version   : รุ่นโมเดล/พจนานุกรมที่ใช้ เช่น nb-v1/lexicon-1
--    analyzed_at     : เวลาที่วิเคราะห์
-- ---------------------------------------------------------------------
alter table public.reviews
  add column if not exists analysis_status text,
  add column if not exists analysis_error  text,
  add column if not exists model_version   text,
  add column if not exists analyzed_at     timestamptz;

-- ข้อมูลเดิม 7,185 รายการวิเคราะห์ไว้แล้วจาก notebook
update public.reviews
   set analysis_status = 'done',
       model_version   = coalesce(model_version, 'nb-v1/lexicon-1')
 where analysis_status is null;

alter table public.reviews
  alter column analysis_status set default 'pending',
  alter column analysis_status set not null;

do $$
begin
  if not exists (select 1 from pg_constraint where conname = 'reviews_analysis_status_check') then
    alter table public.reviews add constraint reviews_analysis_status_check
      check (analysis_status in ('pending', 'done', 'failed'));
  end if;
end $$;

-- ช่วยให้ค้นรีวิวที่ค้างเร็วขึ้น (เก็บเฉพาะแถวที่ยังไม่ done จึงเล็กมาก)
create index if not exists reviews_pending_idx
  on public.reviews (analysis_status) where analysis_status <> 'done';

-- ---------------------------------------------------------------------
-- 2) reviewId อัตโนมัติ ต่อจากรหัสล่าสุด (R007202, R007203, ...)
-- ---------------------------------------------------------------------
create sequence if not exists public.review_id_seq;

-- R007186-R007201 ถูกใช้แล้วกับรีวิวที่ตัดออกตอนทำความสะอาดข้อมูล (ดู sheet reviewId_mapping)
-- จึงเริ่มอย่างน้อยที่ 7202 เพื่อไม่ให้รหัสซ้ำกับข้อมูลชุดเดิม
-- ถ้ามีการรันซ้ำภายหลัง จะต่อจากรหัสล่าสุดในตารางเสมอ
select setval('public.review_id_seq',
              greatest(coalesce((select max(substring("reviewId" from 2)::bigint)
                                   from public.reviews where "reviewId" ~ '^R[0-9]{6,}$'), 0) + 1,
                       7202),
              false);    -- false = ค่าถัดไปที่ได้คือค่านี้พอดี

create or replace function public.next_review_id()
returns text language sql volatile as $$
  select 'R' || case when n < 1000000 then lpad(n::text, 6, '0') else n::text end
    from (select nextval('public.review_id_seq') as n) s;
$$;

alter table public.reviews
  alter column "reviewId" set default public.next_review_id(),
  alter column "publishedAtDate" set default now();

-- ---------------------------------------------------------------------
-- 3) กันผู้ใช้เว็บส่งผลวิเคราะห์ปลอมมาเอง
--    รีวิวที่เพิ่มจากเว็บ (role anon/authenticated) จะถูกล้างผลวิเคราะห์และตั้งเป็น pending เสมอ
--    ผลวิเคราะห์จะถูกเขียนโดย API (service_role) เท่านั้น
--    ส่วนการเพิ่มข้อมูลจาก SQL Editor หรือ service_role ไม่ถูกแตะ (ใช้นำเข้าข้อมูลเดิมได้)
--    ถ้าในอนาคตเพิ่มคอลัมน์ผลวิเคราะห์ ให้เพิ่มบรรทัดในฟังก์ชันนี้ด้วย
-- ---------------------------------------------------------------------
create or replace function public.reviews_reset_analysis()
returns trigger language plpgsql as $$
declare
  caller text := coalesce(nullif(current_setting('request.jwt.claims', true), '')::json ->> 'role', '');
begin
  if caller in ('anon', 'authenticated') then
    new.analysis_status    := 'pending';
    new.analysis_error     := null;
    new.model_version      := null;
    new.analyzed_at        := null;
    new.processed_text     := null;
    new."originalLanguage" := null;
    new.sentiment          := null;
    new.prob_positive      := null;
    new.prob_neutral       := null;
    new.prob_negative      := null;
    new.confidence         := null;
    new.analyzable         := null;
    new.sentiment_text     := null;
    new.positive_text      := null;
    new.negative_text      := null;
    new.n_sentiment_terms  := null;
  end if;
  return new;
end $$;

drop trigger if exists reviews_reset_analysis on public.reviews;
create trigger reviews_reset_analysis
  before insert on public.reviews
  for each row execute function public.reviews_reset_analysis();

-- ---------------------------------------------------------------------
-- 4) เปิด Realtime ให้หน้าเว็บรับผลทันทีเมื่อวิเคราะห์เสร็จ
-- ---------------------------------------------------------------------
do $$
begin
  if exists (select 1 from pg_publication where pubname = 'supabase_realtime')
     and not exists (select 1 from pg_publication_tables
                      where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = 'reviews') then
    alter publication supabase_realtime add table public.reviews;
  end if;
end $$;

-- ---------------------------------------------------------------------
-- 5) สรุป sentiment รายสถานที่แบบอัปเดตอัตโนมัติ (คำนวณใหม่ทุกครั้งที่ query)
--    วิธีเดียวกับ notebook 04: เฉลี่ยความน่าจะเป็นของรีวิวที่ analyzable = 'yes'
--    แล้วเลือกคลาสที่ค่าเฉลี่ยสูงสุด / ดาวเฉลี่ยใช้รีวิวทุกรายการ
-- ---------------------------------------------------------------------
create or replace view public.place_sentiment_summary
with (security_invoker = true) as
with agg as (
  select a.attraction_id,
         a.attraction_name,
         a.category_id,
         count(r."reviewId")                                         as total_reviews,
         avg(r.stars)                                                as avg_stars,
         count(*) filter (where r.analysis_status = 'done' and r.analyzable = 'yes') as analyzed_reviews,
         avg(r.prob_positive) filter (where r.analysis_status = 'done' and r.analyzable = 'yes') as p_pos,
         avg(r.prob_neutral)  filter (where r.analysis_status = 'done' and r.analyzable = 'yes') as p_neu,
         avg(r.prob_negative) filter (where r.analysis_status = 'done' and r.analyzable = 'yes') as p_neg,
         count(*) filter (where r.analysis_status <> 'done')         as pending_reviews
    from public.attractions a
    left join public.reviews r on r.attraction_id = a.attraction_id
   group by a.attraction_id, a.attraction_name, a.category_id
)
select attraction_id,
       attraction_name,
       category_id,
       total_reviews,
       round(avg_stars::numeric, 2) as avg_stars,
       analyzed_reviews,
       round(p_pos::numeric, 4)     as avg_prob_positive,
       round(p_neu::numeric, 4)     as avg_prob_neutral,
       round(p_neg::numeric, 4)     as avg_prob_negative,
       case when p_pos is null                     then null
            when p_pos >= p_neu and p_pos >= p_neg then 'positive'
            when p_neu >= p_neg                    then 'neutral'
            else 'negative' end     as place_sentiment,
       pending_reviews
  from agg;

-- ---------------------------------------------------------------------
-- 6) รีวิวที่สกัดวลีไม่เจอ: ใช้หาคำใหม่ไปเพิ่มในพจนานุกรม (lexicon.json)
-- ---------------------------------------------------------------------
create or replace view public.reviews_without_phrases
with (security_invoker = true) as
select "reviewId", attraction_id, "publishedAtDate", sentiment, processed_text, model_version
  from public.reviews
 where analysis_status = 'done' and analyzable = 'yes' and coalesce(n_sentiment_terms, 0) = 0;

commit;

-- =====================================================================
-- (แนะนำ) สิทธิ์ของผู้ใช้เว็บ: อนุญาตให้เพิ่มรีวิวได้เฉพาะคอลัมน์ที่จำเป็น
-- ใช้เมื่อเปิด Row Level Security แล้ว ตรวจนโยบายเดิมก่อนรัน เพราะอาจกระทบการอ่านข้อมูลของเว็บ
--
-- revoke insert on public.reviews from anon;
-- grant insert (attraction_id, stars, text) on public.reviews to anon;
-- create policy "anyone can add a review" on public.reviews for insert to anon with check (true);
-- create policy "anyone can read reviews"  on public.reviews for select to anon using (true);
-- =====================================================================
