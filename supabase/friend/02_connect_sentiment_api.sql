-- =====================================================================
-- เชื่อมฐานข้อมูลเว็บเพื่อนกับ Sentiment API (songkhla-sentiment-api บน Render)
-- วิธีใช้: Supabase ของเพื่อน > SQL Editor > วางทั้งไฟล์ > Run
-- รันซ้ำได้ (idempotent) / ไม่ลบข้อมูล / ไม่แก้ฟังก์ชัน submit_tourist_review และหน้าเว็บส่งรีวิวเดิม
--
-- สิ่งที่ไฟล์นี้ทำ
--   1) เพิ่มคอลัมน์ผลวิเคราะห์ในตาราง reviews (รีวิวใหม่จะเป็น pending จน API เขียนผลกลับ)
--   2) กันผู้ใช้เว็บปลอมผลวิเคราะห์ (ผู้ใช้ INSERT/UPDATE ตาราง reviews ได้ตาม policy เดิม)
--   3) คำนวณ avg_prob_* / place_sentiment ของสถานที่ใหม่อัตโนมัติ เมื่อรีวิววิเคราะห์เสร็จหรือถูกลบ
--      วิธีเดียวกับค่าเดิม: เฉลี่ยความน่าจะเป็นของรีวิวที่ analyzable = yes (reviews_import + reviews) แล้วเลือกคลาสที่สูงสุด
--   4) view วลีรายสถานที่ (ใช้ทำ word cloud) รวมรีวิวเก่าและรีวิวใหม่
--   5) คำนวณ sentiment ทุกสถานที่ใหม่ 1 ครั้ง (ค่าควรเท่าเดิม)
-- =====================================================================
begin;

-- ---------------------------------------------------------------------
-- 1) คอลัมน์ผลวิเคราะห์ (ชื่อเดียวกับ reviews_import เพื่อให้หน้าเว็บใช้โค้ดแสดงผลเดียวกันได้)
-- ---------------------------------------------------------------------
alter table public.reviews
  add column if not exists analysis_status    text,
  add column if not exists analysis_error     text,
  add column if not exists model_version      text,
  add column if not exists analyzed_at        timestamptz,
  add column if not exists processed_text     text,
  add column if not exists "originalLanguage" text,
  add column if not exists sentiment          text,
  add column if not exists prob_positive      numeric,
  add column if not exists prob_neutral       numeric,
  add column if not exists prob_negative      numeric,
  add column if not exists confidence         numeric,
  add column if not exists analyzable         text,
  add column if not exists sentiment_text     text,
  add column if not exists positive_text      text,
  add column if not exists negative_text      text,
  add column if not exists n_sentiment_terms  integer;

-- รีวิวที่มีอยู่แล้ว (ยังไม่เคยวิเคราะห์) -> pending ให้ API วิเคราะห์ย้อนหลัง
update public.reviews set analysis_status = 'pending' where analysis_status is null;

alter table public.reviews
  alter column analysis_status set default 'pending',
  alter column analysis_status set not null;

do $$
begin
  if not exists (select 1 from pg_constraint where conname = 'reviews_analysis_status_check') then
    alter table public.reviews add constraint reviews_analysis_status_check
      check (analysis_status in ('pending', 'done', 'failed'));
  end if;
  if not exists (select 1 from pg_constraint where conname = 'reviews_sentiment_check') then
    alter table public.reviews add constraint reviews_sentiment_check
      check (sentiment is null or sentiment in ('positive', 'neutral', 'negative'));
  end if;
end $$;

-- ค้นรีวิวที่ค้างเร็วขึ้น (index เล็ก เก็บเฉพาะแถวที่ยังไม่ done)
create index if not exists reviews_pending_idx
  on public.reviews (analysis_status) where analysis_status <> 'done';
create index if not exists reviews_attraction_idx on public.reviews (attraction_id);
create index if not exists reviews_import_attraction_idx on public.reviews_import (attraction_id);

-- ---------------------------------------------------------------------
-- 2) กันการปลอมผลวิเคราะห์
--    policy เดิมให้ผู้ใช้ INSERT และ UPDATE รีวิวของตัวเองได้โดยตรง
--    - INSERT จากผู้ใช้ (รวมผ่าน submit_tourist_review) : ล้างผลวิเคราะห์ ตั้งเป็น pending เสมอ
--    - UPDATE จากผู้ใช้ : คงผลวิเคราะห์เดิมไว้ (แก้เองไม่ได้) ถ้าแก้ข้อความ -> กลับเป็น pending ให้วิเคราะห์ใหม่
--    - API (service_role) และ SQL Editor (postgres) ไม่ถูกแตะ
--    ตรวจ role จาก JWT ของคำขอ (ใช้ได้แม้เรียกผ่านฟังก์ชัน SECURITY DEFINER)
-- ---------------------------------------------------------------------
create or replace function public.reviews_guard_analysis()
returns trigger language plpgsql set search_path = public as $$
declare
  caller text := coalesce(nullif(current_setting('request.jwt.claims', true), '')::json ->> 'role', '');
  reset boolean;
begin
  if caller not in ('anon', 'authenticated') then
    return new;
  end if;
  reset := tg_op = 'INSERT' or new.text is distinct from old.text;
  if reset then
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
  else
    new.analysis_status    := old.analysis_status;
    new.analysis_error     := old.analysis_error;
    new.model_version      := old.model_version;
    new.analyzed_at        := old.analyzed_at;
    new.processed_text     := old.processed_text;
    new."originalLanguage" := old."originalLanguage";
    new.sentiment          := old.sentiment;
    new.prob_positive      := old.prob_positive;
    new.prob_neutral       := old.prob_neutral;
    new.prob_negative      := old.prob_negative;
    new.confidence         := old.confidence;
    new.analyzable         := old.analyzable;
    new.sentiment_text     := old.sentiment_text;
    new.positive_text      := old.positive_text;
    new.negative_text      := old.negative_text;
    new.n_sentiment_terms  := old.n_sentiment_terms;
  end if;
  return new;
end $$;

drop trigger if exists reviews_guard_analysis on public.reviews;
create trigger reviews_guard_analysis
  before insert or update on public.reviews
  for each row execute function public.reviews_guard_analysis();

-- ---------------------------------------------------------------------
-- 3) คำนวณ sentiment ของสถานที่ใหม่
--    reviews_import เชื่อมกับสถานที่ด้วย attractions.attraction_code (ทั้ง 46 สถานที่มีรหัส)
--    reviews เชื่อมด้วย attractions.id และนับเฉพาะ analysis_status = done
--    ไม่แตะ review_score / review_count (submit_tourist_review ดูแลอยู่แล้ว)
-- ---------------------------------------------------------------------
create or replace function public.refresh_attraction_sentiment(p_attraction_id bigint)
returns void language plpgsql security definer set search_path = public as $$
declare
  v_code text;
  v_name text;
  v_n    integer;
  v_pos  numeric;
  v_neu  numeric;
  v_neg  numeric;
  v_label text;
begin
  select attraction_code, name into v_code, v_name from public.attractions where id = p_attraction_id;
  if not found then
    return;
  end if;

  with probs as (
    select nullif(trim(prob_positive), '')::numeric as p_pos,
           nullif(trim(prob_neutral), '')::numeric  as p_neu,
           nullif(trim(prob_negative), '')::numeric as p_neg
      from public.reviews_import
     where v_code is not null and attraction_id = v_code
       and lower(trim(coalesce(analyzable, ''))) in ('yes', 'true', '1')
    union all
    select prob_positive, prob_neutral, prob_negative
      from public.reviews
     where attraction_id = p_attraction_id and analysis_status = 'done' and analyzable = 'yes'
  )
  select count(*), avg(p_pos), avg(p_neu), avg(p_neg)
    into v_n, v_pos, v_neu, v_neg
    from probs
   where p_pos is not null and p_neu is not null and p_neg is not null;

  v_label := case when v_n = 0 then null
                  when v_pos >= v_neu and v_pos >= v_neg then 'positive'
                  when v_neg >= v_neu then 'negative'
                  else 'neutral' end;

  update public.attractions
     set avg_prob_positive     = v_pos,
         avg_prob_neutral      = v_neu,
         avg_prob_negative     = v_neg,
         place_sentiment       = v_label,
         analyzed_reviews      = v_n,
         analyzed_review_count = v_n
   where id = p_attraction_id;

  if v_n > 0 then
    insert into public.attraction_sentiment_summary
           (attraction_id, attraction_name, analyzed_review_count, avg_positive, avg_neutral, avg_negative,
            sentiment, calculated_at)
    values (p_attraction_id, v_name, v_n, v_pos, v_neu, v_neg, v_label, now())
    on conflict (attraction_id) do update
       set attraction_name       = excluded.attraction_name,
           analyzed_review_count = excluded.analyzed_review_count,
           avg_positive          = excluded.avg_positive,
           avg_neutral           = excluded.avg_neutral,
           avg_negative          = excluded.avg_negative,
           sentiment             = excluded.sentiment,
           calculated_at         = excluded.calculated_at;
  end if;
end $$;

-- ให้เรียกได้เฉพาะฝั่งเซิร์ฟเวอร์ (trigger / SQL Editor) ไม่ให้ผู้ใช้เว็บเรียกตรง
revoke all on function public.refresh_attraction_sentiment(bigint) from public, anon, authenticated;

create or replace function public.reviews_refresh_attraction()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  if tg_op = 'DELETE' then
    if old.analysis_status = 'done' then
      perform public.refresh_attraction_sentiment(old.attraction_id);
    end if;
    return old;
  end if;
  -- วิเคราะห์เสร็จ / ผลเปลี่ยน / ย้ายสถานที่ / ถูกตั้งกลับเป็น pending หลังแก้ข้อความ
  if (new.analysis_status = 'done' or old.analysis_status = 'done')
     and (new.analysis_status is distinct from old.analysis_status
          or new.prob_positive is distinct from old.prob_positive
          or new.analyzable is distinct from old.analyzable
          or new.attraction_id is distinct from old.attraction_id) then
    perform public.refresh_attraction_sentiment(new.attraction_id);
    if new.attraction_id is distinct from old.attraction_id then
      perform public.refresh_attraction_sentiment(old.attraction_id);
    end if;
  end if;
  return new;
end $$;

drop trigger if exists reviews_refresh_attraction on public.reviews;
create trigger reviews_refresh_attraction
  after update or delete on public.reviews
  for each row execute function public.reviews_refresh_attraction();

-- ---------------------------------------------------------------------
-- 4) วลีรายสถานที่สำหรับ word cloud (รีวิวเก่า + รีวิวใหม่ที่วิเคราะห์แล้ว)
--    วลีแต่ละคำในคอลัมน์ positive_text / negative_text คั่นด้วยช่องว่าง (วลีเองไม่มีช่องว่าง)
--    ใช้: select term, polarity, mentions from attraction_phrase_counts where attraction_id = 5 order by mentions desc limit 40
-- ---------------------------------------------------------------------
create or replace view public.attraction_phrase_counts
with (security_invoker = true) as
with src as (
  select a.id as attraction_id, ri.positive_text, ri.negative_text
    from public.reviews_import ri
    join public.attractions a on a.attraction_code = ri.attraction_id
  union all
  select r.attraction_id, r.positive_text, r.negative_text
    from public.reviews r
   where r.analysis_status = 'done'
),
terms as (
  select attraction_id, t as term, 'positive'::text as polarity
    from src, regexp_split_to_table(coalesce(positive_text, ''), '\s+') t
  union all
  select attraction_id, t, 'negative'
    from src, regexp_split_to_table(coalesce(negative_text, ''), '\s+') t
)
select attraction_id, term, polarity, count(*)::int as mentions
  from terms
 where term <> ''
 group by attraction_id, term, polarity;

grant select on public.attraction_phrase_counts to anon, authenticated;

-- ---------------------------------------------------------------------
-- 5) คำนวณ sentiment ของทุกสถานที่ 1 ครั้ง (ค่าควรเท่าเดิม ถ้าไฟล์ 01_preview แสดง max_diff ~ 0)
-- ---------------------------------------------------------------------
select public.refresh_attraction_sentiment(id) from public.attractions;

commit;

-- ตรวจผล: ควรเห็นคอลัมน์ใหม่ และรีวิวเดิมเป็น pending
select id, attraction_id, stars, left(text, 40) as text, analysis_status from public.reviews order by id;
