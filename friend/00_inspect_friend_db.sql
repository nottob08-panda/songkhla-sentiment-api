-- =====================================================================
-- ตรวจสภาพฐานข้อมูลของเว็บเพื่อน (อ่านอย่างเดียว ไม่แก้อะไรในฐานข้อมูล)
-- วิธีใช้: Supabase ของเพื่อน > SQL Editor > New query > วางทั้งไฟล์ > Run
--         แล้วกด Export (มุมขวาบนของตารางผลลัพธ์) > Download CSV แล้วส่งไฟล์ CSV มา
-- ผลลัพธ์มี 3 คอลัมน์: section | name | detail
-- =====================================================================
with
-- 1) ฟังก์ชันทั้งหมดใน schema public (เช่น submit_tourist_review) พร้อมโค้ดเต็ม
fn as (
  select '1_function'::text as section,
         p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')' as name,
         'security_definer=' || p.prosecdef || E'\n' || pg_get_functiondef(p.oid) as detail
    from pg_proc p
   where p.pronamespace = 'public'::regnamespace and p.prokind = 'f'
),
-- 2) trigger บนตารางใน public (รวม Database Webhook ที่ตั้งจากหน้า Dashboard)
trg as (
  select '2_trigger', c.relname || '.' || t.tgname, pg_get_triggerdef(t.oid)
    from pg_trigger t join pg_class c on c.oid = t.tgrelid
   where c.relnamespace = 'public'::regnamespace and not t.tgisinternal
),
-- 3) สิทธิ์อ่าน/เขียน (RLS policy)
pol as (
  select '3_policy', tablename || '.' || policyname,
         'cmd=' || cmd || ' roles=' || array_to_string(roles, ',') ||
         ' using=' || coalesce(qual, '-') || ' check=' || coalesce(with_check, '-')
    from pg_policies where schemaname = 'public'
),
-- 4) ตารางไหนเปิด RLS อยู่บ้าง
rls as (
  select '4_rls', c.relname, 'rls_enabled=' || c.relrowsecurity
    from pg_class c where c.relnamespace = 'public'::regnamespace and c.relkind = 'r'
),
-- 5) คอลัมน์ ค่าเริ่มต้น และ not null ของตารางที่เกี่ยวข้อง
col as (
  select '5_column', table_name || '.' || column_name,
         data_type || ' default=' || coalesce(column_default, '-') || ' nullable=' || is_nullable
    from information_schema.columns
   where table_schema = 'public'
     and table_name in ('reviews', 'reviews_import', 'attractions', 'attraction_id_mapping',
                        'attraction_sentiment_summary')
),
-- 6) constraint (primary key / foreign key / unique / check)
con as (
  select '6_constraint', c.conrelid::regclass::text || '.' || c.conname, pg_get_constraintdef(c.oid)
    from pg_constraint c where c.connamespace = 'public'::regnamespace
),
-- 7) view ใน public
vw as (
  select '7_view', viewname, definition from pg_views where schemaname = 'public'
),
-- 8) จำนวนแถว และตัวอย่างรหัสที่ใช้เชื่อมรีวิวเก่ากับสถานที่
cnt as (
  select '8_count', 'reviews', (select count(*) from public.reviews)::text
  union all select '8_count', 'reviews_import', (select count(*) from public.reviews_import)::text
  union all select '8_count', 'reviews_import_distinct_reviewId',
                   (select count(distinct "reviewId") from public.reviews_import)::text
  union all select '8_count', 'attractions', (select count(*) from public.attractions)::text
  union all select '8_count', 'attractions_with_code',
                   (select count(*) from public.attractions where coalesce(attraction_code, '') <> '')::text
  union all select '8_count', 'attraction_id_mapping', (select count(*) from public.attraction_id_mapping)::text
  union all select '8_count', 'mapping_status',
                   (select string_agg(match_status || '=' || n, ', ')
                      from (select match_status, count(*) n from public.attraction_id_mapping group by 1) s)
  union all select '8_count', 'reviews_import_attraction_ids_not_mapped',
                   (select string_agg(distinct ri.attraction_id, ', ')
                      from public.reviews_import ri
                     where not exists (select 1 from public.attractions a where a.attraction_code = ri.attraction_id)
                       and not exists (select 1 from public.attraction_id_mapping m where m.csv_id = ri.attraction_id))
  union all select '8_count', 'sample_attractions',
                   (select string_agg(id || ':' || coalesce(attraction_code, '-') || ':' || name, ' | ' order by id)
                      from (select * from public.attractions order by id limit 8) s)
),
-- 9) extension ที่ติดตั้ง (pg_net ใช้ส่ง webhook)
ext as (
  select '9_extension', extname, extversion from pg_extension
),
-- 10) Realtime เปิดให้ตารางไหนบ้าง
rt as (
  select '10_realtime', tablename, pubname from pg_publication_tables where pubname = 'supabase_realtime'
)
select * from fn
union all select * from trg
union all select * from pol
union all select * from rls
union all select * from col
union all select * from con
union all select * from vw
union all select * from cnt
union all select * from ext
union all select * from rt
order by 1, 2;
