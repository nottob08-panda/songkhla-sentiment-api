import "./style.css";
import {
  isConfigured, submitReview, getPlaceSummary, listReviews, listCategories,
} from "./reviewClient.js";

const $ = (id) => document.getElementById(id);
const LABEL = { positive: "เชิงบวก", neutral: "กลาง ๆ", negative: "เชิงลบ" };
const EMOJI = { positive: "😊", neutral: "😐", negative: "😞" };

let places = [];
let current = null; // attraction_id ที่เลือกอยู่

// สร้าง element โดยใส่ข้อความผ่าน textContent เสมอ (กันโค้ดแปลกปลอมจากข้อความรีวิว)
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function showFatal(message) {
  const box = $("fatal");
  box.textContent = message;
  box.hidden = false;
}

const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
const starText = (n) => "★".repeat(Math.round(n)) + "☆".repeat(5 - Math.round(n));

function sentimentBadge(sentiment) {
  if (!sentiment) return el("span", "badge none", "วิเคราะห์ไม่ได้");
  return el("span", `badge ${sentiment}`, `${EMOJI[sentiment]} ${LABEL[sentiment]}`);
}

function probBar(pos, neu, neg) {
  const bar = el("div", "bar");
  [["positive", pos], ["neutral", neu], ["negative", neg]].forEach(([k, v]) => {
    const seg = el("span", `seg ${k}`);
    seg.style.width = pct(v);
    seg.title = `${LABEL[k]} ${pct(v)}`;
    bar.append(seg);
  });
  return bar;
}

function phrases(row) {
  const box = el("div", "phrases");
  if (row.positive_text) box.append(el("p", "ph pos", `👍 ${row.positive_text}`));
  if (row.negative_text) box.append(el("p", "ph neg", `👎 ${row.negative_text}`));
  return box;
}

// ---------------------------------------------------------------- สรุปสถานที่
function renderSummary(p) {
  const box = $("summary");
  box.replaceChildren();
  if (!p) return;

  const head = el("div", "sum-head");
  head.append(el("h2", null, p.attraction_name));
  if (p.place_sentiment) head.append(sentimentBadge(p.place_sentiment));
  box.append(head);

  const stats = el("div", "stats");
  const stat = (value, label) => {
    const s = el("div", "stat");
    s.append(el("strong", null, value), el("span", "muted", label));
    return s;
  };
  stats.append(
    stat(p.avg_stars != null ? `${Number(p.avg_stars).toFixed(2)} ★` : "–", "ดาวเฉลี่ย"),
    stat(p.total_reviews ?? 0, "รีวิวทั้งหมด"),
    stat(p.analyzed_reviews ?? 0, "วิเคราะห์ได้"),
  );
  box.append(stats);

  if (p.avg_prob_positive != null) {
    box.append(el("p", "label", "ความน่าจะเป็นเฉลี่ยของแต่ละอารมณ์"));
    box.append(probBar(p.avg_prob_positive, p.avg_prob_neutral, p.avg_prob_negative));
    const legend = el("div", "legend");
    legend.append(
      el("span", "lg positive", `บวก ${pct(p.avg_prob_positive)}`),
      el("span", "lg neutral", `กลาง ${pct(p.avg_prob_neutral)}`),
      el("span", "lg negative", `ลบ ${pct(p.avg_prob_negative)}`),
    );
    box.append(legend);
  }
  if (p.pending_reviews > 0) {
    box.append(el("p", "muted small", `มี ${p.pending_reviews} รีวิวที่กำลังรอวิเคราะห์`));
  }
}

// ---------------------------------------------------------------- รายการรีวิว
function reviewItem(r) {
  const li = el("li", "review");
  const top = el("div", "rv-top");
  top.append(el("span", "rv-stars", starText(r.stars || 0)));
  if (r.analysis_status === "done") top.append(sentimentBadge(r.sentiment));
  else top.append(el("span", "badge none", r.analysis_status === "failed" ? "รอวิเคราะห์ใหม่" : "กำลังวิเคราะห์"));
  if (r.publishedAtDate) {
    top.append(el("span", "muted small", new Date(r.publishedAtDate).toLocaleDateString("th-TH")));
  }
  li.append(top, el("p", "rv-text", r.text), phrases(r));
  return li;
}

async function refreshPlace() {
  if (!current) return;
  const [summary, reviews] = await Promise.all([getPlaceSummary(current), listReviews(current, 10)]);
  renderSummary(summary[0]);
  const list = $("reviews");
  list.replaceChildren(...reviews.map(reviewItem));
  if (!reviews.length) list.append(el("li", "muted", "ยังไม่มีรีวิว เป็นคนแรกที่รีวิวเลย"));
}

// ---------------------------------------------------------------- ส่งรีวิว
function renderResult(r, t) {
  // t = { sentiment: วินาทีที่ได้ผลบวก/ลบ, phrases: วินาทีที่ได้วลี (ถ้ามา) }
  const box = $("result");
  box.replaceChildren();
  box.hidden = false;
  box.className = "result";

  if (r.sentiment) {
    const done = r.analysis_status === "done";
    const timing = `ผลบวก/ลบ ${t.sentiment ?? t.phrases} วิ` + (done && t.sentiment ? ` · วลี ${t.phrases} วิ` : "");
    const head = el("div", "res-head");
    head.append(sentimentBadge(r.sentiment), el("span", "muted small", `มั่นใจ ${pct(r.confidence)} · ${timing}`));
    box.append(head, probBar(r.prob_positive, r.prob_neutral, r.prob_negative));
    if (r.originalLanguage && r.originalLanguage !== "th" && r.processed_text) {
      box.append(el("p", "muted small translated", `แปลเป็นไทย (${r.originalLanguage}): ${r.processed_text}`));
    }
    if (!done) {                                   // จังหวะที่ 1: วลียังไม่มา
      const wait = el("p", "muted small waiting");
      wait.append(el("span", "spinner"), el("span", null, "กำลังสกัดวลี…"));
      box.append(wait);
      return;
    }
    box.append(phrases(r));
    if (!r.positive_text && !r.negative_text) {
      box.append(el("p", "muted small", "ไม่พบวลีแสดงความรู้สึกในรีวิวนี้"));
    }
  } else if (r.analysis_status === "done") {
    box.append(el("p", null, "บันทึกรีวิวแล้ว ข้อความนี้วิเคราะห์อารมณ์ไม่ได้ (เช่น มีแต่อีโมจิ)"));
  } else if (r.analysis_status === "failed") {
    box.append(el("p", null, "บันทึกรีวิวแล้ว แต่ระบบแปลภาษาขัดข้องชั่วคราว จะวิเคราะห์ใหม่ให้อัตโนมัติ"));
  } else {
    box.append(el("p", null, "บันทึกรีวิวแล้ว ระบบกำลังวิเคราะห์ (เซิร์ฟเวอร์อาจกำลังตื่น) ผลจะแสดงในรายการรีวิวภายหลัง"));
  }
}

function setupForm() {
  const form = $("review-form");
  const text = $("text");
  const button = $("submit");

  text.addEventListener("input", () => {
    $("count").textContent = `${text.value.length} / 5000`;
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const value = text.value.trim();
    if (!value || !current) return;

    button.disabled = true;
    button.textContent = "กำลังวิเคราะห์…";
    const box = $("result");
    box.hidden = false;
    box.className = "result loading";
    box.replaceChildren(el("span", "spinner"), el("span", null, "กำลังบันทึกและวิเคราะห์รีวิว"));

    const started = performance.now();
    try {
      const stars = Number(form.querySelector("input[name=stars]:checked").value);
      const secs = () => ((performance.now() - started) / 1000).toFixed(1);
      const t = {};
      const row = await submitReview({ attractionId: current, stars, text: value }, {
        onProgress: (partial) => { t.sentiment = secs(); renderResult(partial, t); },
      });
      t.phrases = secs();
      renderResult(row, t);
      form.reset();
      $("count").textContent = "0 / 5000";
      await refreshPlace();
    } catch (err) {
      box.className = "result error";
      box.replaceChildren(el("p", null, `ส่งรีวิวไม่สำเร็จ: ${err.message}`));
    } finally {
      button.disabled = false;
      button.textContent = "ส่งรีวิว";
    }
  });
}

// ---------------------------------------------------------------- เริ่มต้น
async function init() {
  if (!isConfigured) {
    showFatal("ยังไม่ได้ตั้งค่า VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY (ดูไฟล์ .env.example)");
    return;
  }
  setupForm();

  try {
    const [summary, categories] = await Promise.all([getPlaceSummary(), listCategories()]);
    places = summary.sort((a, b) => a.attraction_name.localeCompare(b.attraction_name, "th"));
    if (!places.length) {
      // ไม่มี error แต่ได้ 0 แถว = ส่วนใหญ่เป็นเพราะ Row Level Security ซ่อนข้อมูล (ยังไม่ได้รัน 02_web_permissions.sql)
      $("place").replaceChildren(el("option", null, "ไม่พบสถานที่"));
      $("submit").disabled = true;
      showFatal("ไม่พบข้อมูลสถานที่: ตรวจว่ารัน supabase/02_web_permissions.sql แล้ว " +
        "และ VITE_SUPABASE_URL ชี้ไปที่โปรเจกต์ Supabase ที่มีข้อมูล");
      return;
    }
    const catName = Object.fromEntries(categories.map((c) => [c.category_id, c.category_name]));

    // จัดกลุ่มตามหมวดหมู่ใน dropdown
    const select = $("place");
    select.replaceChildren();
    const groups = new Map();
    for (const p of places) {
      const key = catName[p.category_id] || "อื่น ๆ";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(p);
    }
    [...groups.keys()].sort((a, b) => a.localeCompare(b, "th")).forEach((key) => {
      const og = document.createElement("optgroup");
      og.label = key;
      groups.get(key).forEach((p) => {
        const opt = el("option", null, p.attraction_name);
        opt.value = p.attraction_id;
        og.append(opt);
      });
      select.append(og);
    });

    // เปิดหน้าด้วย ?place=A001 เพื่อเลือกสถานที่ไว้ล่วงหน้าได้
    const fromUrl = new URLSearchParams(location.search).get("place");
    current = places.some((p) => p.attraction_id === fromUrl) ? fromUrl : places[0]?.attraction_id;
    select.value = current;
    select.disabled = false;
    $("submit").disabled = !current;

    select.addEventListener("change", () => {
      current = select.value;
      history.replaceState(null, "", `?place=${current}`);
      $("result").hidden = true;
      refreshPlace().catch((err) => showFatal(`โหลดข้อมูลไม่สำเร็จ: ${err.message}`));
    });

    await refreshPlace();
  } catch (err) {
    showFatal(`โหลดข้อมูลไม่สำเร็จ: ${err.message}`);
  }
}

init();
