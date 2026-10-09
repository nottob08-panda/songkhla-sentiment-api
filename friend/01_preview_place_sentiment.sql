-- =====================================================================
-- ตรวจก่อนติดตั้ง (อ่านอย่างเดียว ไม่แก้อะไร): ค่า avg_prob_* / place_sentiment ใน attractions
-- ถูกคำนวณจาก reviews_import ด้วยวิธีไหน -> ให้ระบบอัตโนมัติคำนวณแบบเดียวกัน ตัวเลขบนเว็บจะไม่กระโดด
-- วิธีใช้: Supabase ของเพื่อน > SQL Editor > วางทั้งไฟล์ > Run > Export > Download CSV
-- =====================================================================
with ri as (
  select attraction_id as code,
         lower(trim(coalesce(analyzable, ''))) as analyzable,
         nullif(trim(prob_positive), '')::numeric as p_pos,
         nullif(trim(prob_neutral), '')::numeric  as p_neu,
         nullif(trim(prob_negative), '')::numeric as p_neg
    from public.reviews_import
),
calc as (
  select a.id, a.attraction_code, a.name,
         a.avg_prob_positive, a.avg_prob_neutral, a.avg_prob_negative, a.place_sentiment,
         a.analyzed_reviews, a.analyzed_review_count, a.total_reviews,
         -- วิธี A: เฉลี่ยเฉพาะรีวิวที่ analyzable = yes (วิธีเดียวกับ notebook)
         avg(ri.p_pos) filter (where ri.analyzable in ('yes', 'true', '1')) as a_pos,
         avg(ri.p_neu) filter (where ri.analyzable in ('yes', 'true', '1')) as a_neu,
         avg(ri.p_neg) filter (where ri.analyzable in ('yes', 'true', '1')) as a_neg,
         count(*)      filter (where ri.analyzable in ('yes', 'true', '1')) as a_n,
         -- วิธี B: เฉลี่ยทุกรีวิวที่มีค่าความน่าจะเป็น
         avg(ri.p_pos) as b_pos, avg(ri.p_neu) as b_neu, avg(ri.p_neg) as b_neg,
         count(ri.p_pos) as b_n,
         count(ri.code)  as n_all
    from public.attractions a
    left join ri on ri.code = a.attraction_code
   group by a.id
),
cmp as (
  select *,
         greatest(abs(coalesce(avg_prob_positive, 0) - coalesce(a_pos, 0)),
                  abs(coalesce(avg_prob_neutral, 0)  - coalesce(a_neu, 0)),
                  abs(coalesce(avg_prob_negative, 0) - coalesce(a_neg, 0))) as diff_a,
         greatest(abs(coalesce(avg_prob_positive, 0) - coalesce(b_pos, 0)),
                  abs(coalesce(avg_prob_neutral, 0)  - coalesce(b_neu, 0)),
                  abs(coalesce(avg_prob_negative, 0) - coalesce(b_neg, 0))) as diff_b,
         case when a_pos is null then null
              when a_pos >= a_neu and a_pos >= a_neg then 'positive'
              when a_neg >= a_neu then 'negative' else 'neutral' end as a_label
    from calc
)
select 'summary' as section, 'max_diff_method_A (analyzable=yes)' as name, round(max(diff_a), 6)::text as detail from cmp
union all select 'summary', 'max_diff_method_B (all rows)', round(max(diff_b), 6)::text from cmp
union all select 'summary', 'label_mismatch_method_A',
                 count(*) filter (where a_label is distinct from lower(place_sentiment))::text from cmp
union all select 'summary', 'analyzed_reviews_equals_count_A',
                 count(*) filter (where analyzed_reviews = a_n)::text || '/' || count(*) from cmp
union all select 'summary', 'analyzed_review_count_equals_count_A',
                 count(*) filter (where analyzed_review_count = a_n)::text || '/' || count(*) from cmp
union all select 'summary', 'total_reviews_equals_import_rows',
                 count(*) filter (where total_reviews = n_all)::text || '/' || count(*) from cmp
union all select 'values', 'reviews_import.analyzable',
                 (select string_agg(coalesce(analyzable, 'NULL') || '=' || n, ', ')
                    from (select analyzable, count(*) n from public.reviews_import group by 1) s)
union all select 'values', 'reviews_import.sentiment',
                 (select string_agg(coalesce(sentiment, 'NULL') || '=' || n, ', ')
                    from (select sentiment, count(*) n from public.reviews_import group by 1) s)
union all select 'values', 'attractions.place_sentiment',
                 (select string_agg(coalesce(place_sentiment, 'NULL') || '=' || n, ', ')
                    from (select place_sentiment, count(*) n from public.attractions group by 1) s)
union all select 'values', 'sample_prob_text',
                 (select string_agg(prob_positive, ' | ') from (select prob_positive from public.reviews_import limit 5) s)
union all select 'values', 'summary_table_rows',
                 (select count(*)::text from public.attraction_sentiment_summary)
union all select 'values', 'summary_table_id_is_identity',
                 (select is_identity || ' default=' || coalesce(column_default, '-')
                    from information_schema.columns
                   where table_schema = 'public' and table_name = 'attraction_sentiment_summary' and column_name = 'id')
union all select 'values', 'reviews_id_is_identity',
                 (select is_identity || ' default=' || coalesce(column_default, '-')
                    from information_schema.columns
                   where table_schema = 'public' and table_name = 'reviews' and column_name = 'id')
union all
select 'place', id || ' ' || attraction_code || ' ' || name,
       'stored=' || coalesce(round(avg_prob_positive, 4)::text, '-') || '/' || coalesce(round(avg_prob_neutral, 4)::text, '-')
       || '/' || coalesce(round(avg_prob_negative, 4)::text, '-') || ' ' || coalesce(place_sentiment, '-')
       || ' n=' || coalesce(analyzed_reviews::text, '-')
       || ' | A=' || coalesce(round(a_pos, 4)::text, '-') || '/' || coalesce(round(a_neu, 4)::text, '-')
       || '/' || coalesce(round(a_neg, 4)::text, '-') || ' ' || coalesce(a_label, '-') || ' n=' || a_n
       || ' | B_n=' || b_n || ' all=' || n_all
  from cmp
order by 1 desc, 2;
