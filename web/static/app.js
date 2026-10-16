/* AnemiaLens Web — logika halaman (tanpa framework) */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const fmt = (v, d = 1) => Number(v).toLocaleString("id-ID", { minimumFractionDigits: d, maximumFractionDigits: d });
  const API = location.protocol === "file:" ? null : "";
  const HIST_KEY = "anemialens_history_v1";

  const state = {
    info: null, ready: false,
    file: null, upload: null, previewOk: false,
    poly: null, mode: "auto", group: { thr: 12, name: "Wanita dewasa" },
    last: null,
  };

  /* ---------------------------------------------------------------- gauge */
  function gauge(p, ts = 0.43, tp = 0.61, opts = {}) {
    const cx = 120, cy = 118, r = 96, w = opts.w || 18;
    const pt = (v, rr = r) => {
      const a = Math.PI * (1 - Math.min(1, Math.max(0, v)));
      return [cx + rr * Math.cos(a), cy - rr * Math.sin(a)];
    };
    const arc = (a, b, col) => {
      const [x0, y0] = pt(a), [x1, y1] = pt(b);
      return `<path d="M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}" stroke="${col}" stroke-width="${w}" fill="none" stroke-linecap="butt"/>`;
    };
    const gap = 0.008;
    const [nx, ny] = pt(p, r - 26);
    const label = opts.label !== false;
    return `<svg viewBox="0 0 240 ${label ? 150 : 128}" role="img" aria-label="Skor risiko ${Math.round(p * 100)} dari 100">
      <path d="M${cx - r} ${cy} A${r} ${r} 0 0 1 ${cx + r} ${cy}" stroke="#F2ECE8" stroke-width="${w + 8}" fill="none" stroke-linecap="round"/>
      ${arc(0, ts - gap, "#2BA38C")}${arc(ts + gap, tp - gap, "#F2A93B")}${arc(tp + gap, 1, "#E0475E")}
      <line x1="${cx}" y1="${cy}" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}" stroke="#1F1A2E" stroke-width="5" stroke-linecap="round" class="needle"/>
      <circle cx="${cx}" cy="${cy}" r="10" fill="#1F1A2E"/><circle cx="${cx}" cy="${cy}" r="4" fill="#fff"/>
      ${label ? `<text x="${cx}" y="146" text-anchor="middle" font-family="Plus Jakarta Sans,sans-serif" font-size="13" font-weight="700" fill="#6E6679">skor risiko ${Math.round(p * 100)}/100</text>` : ""}
    </svg>`;
  }

  /* ---------------------------------------------------------------- status server */
  async function loadInfo() {
    const pill = $("#status-pill"), txt = $("span", pill);
    if (API === null) return offline("Buka lewat server");
    try {
      const r = await fetch("/api/info", { cache: "no-store" });
      const d = await r.json();
      state.info = d;
      fillStats(d);
      if (d.ready) {
        state.ready = true;
        pill.className = "pill status ok"; txt.textContent = "Model AI siap";
        $("#offline").hidden = true;
      } else if (d.error) {
        pill.className = "pill status err"; txt.textContent = "Model gagal dimuat";
        const b = $("#offline"); b.hidden = false; b.innerHTML = `<b>Model belum bisa dipakai.</b> ${esc(d.error)}`;
      } else {
        pill.className = "pill status wait"; txt.textContent = "Memuat model…";
        setTimeout(loadInfo, 2000);
      }
    } catch (e) {
      offline("Server offline");
      setTimeout(loadInfo, 5000);
    }
    updateAnalyzeBtn();
  }
  function offline(msg) {
    const pill = $("#status-pill");
    pill.className = "pill status err"; $("span", pill).textContent = msg;
    $("#offline").hidden = false;
  }
  function fillStats(d) {
    const set = (k, v) => $$(`[data-k="${k}"]`).forEach(e => (e.textContent = v));
    if (d.auc) { set("auc", fmt(d.auc, 2)); set("auc2", fmt(d.auc, 3)); }
    if (d.n_patients) set("pat", d.n_patients);
    if (d.n_sources) set("src", d.n_sources);
    if (d.hb_mae) { set("mae", "±" + fmt(d.hb_mae, 1)); set("mae2", fmt(d.hb_mae, 2) + " g/dL"); }
    if (d.sensitivity) set("sens", Math.round(d.sensitivity * 100) + "%");
    if (d.specificity) set("spec", Math.round(d.specificity * 100) + "%");
    if (d.seg_dice) set("segtxt", d.seg_dice_hp
      ? `Memotong konjungtiva, tahan foto HP & mata pucat (Dice ${fmt(d.seg_dice, 2)} foto asli · ${fmt(d.seg_dice_hp, 2)} simulasi HP)`
      : `Memotong tepat area konjungtiva (Dice ${fmt(d.seg_dice, 2)})`);
    $("#hero-gauge").innerHTML = gauge(0.55, d.thr_sensitive || 0.43, d.thr_specific || 0.61, { w: 16, label: false });
  }

  /* ---------------------------------------------------------------- langkah */
  function setStep(n) {
    [1, 2, 3].forEach(i => ($("#step" + i).hidden = i !== n));
    $$("#stepper li").forEach(li => {
      const s = +li.dataset.s;
      li.classList.toggle("on", s === n);
      li.classList.toggle("done", s < n);
    });
  }
  function scrollToApp() {
    const top = $("#app").getBoundingClientRect().top + scrollY - 80;
    if (Math.abs(scrollY - top) > 40) scrollTo({ top, behavior: "smooth" });
  }

  // langkah 1: kelompok
  $$("#group-chips .chip").forEach(c => c.addEventListener("click", () => {
    $$("#group-chips .chip").forEach(x => x.classList.remove("on"));
    c.classList.add("on");
    state.group = { thr: parseFloat(c.dataset.thr), name: c.dataset.g };
  }));
  $("#go2").addEventListener("click", () => { setStep(2); showPicker(); scrollToApp(); });

  /* ---------------------------------------------------------------- langkah 2: foto */
  function showPicker() {
    $("#pick").hidden = false; $("#editor").hidden = true; $("#loading").hidden = true;
  }
  ["#in-cam", "#in-file"].forEach(s => $(s).addEventListener("change", e => {
    const f = e.target.files && e.target.files[0];
    e.target.value = "";
    if (f) onFile(f);
  }));
  const pick = $("#pick");
  ["dragenter", "dragover"].forEach(ev => pick.addEventListener(ev, e => { e.preventDefault(); pick.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => pick.addEventListener(ev, e => { e.preventDefault(); pick.classList.remove("drag"); }));
  pick.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) onFile(f); });

  function onFile(f) {
    state.file = f; state.upload = f; state.poly = null; drawLasso(null);
    const img = $("#preview");
    if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
    const url = URL.createObjectURL(f);
    img.dataset.url = url;
    img.onload = () => {
      state.previewOk = true;
      $("#stage").hidden = false; $("#noprev").hidden = true;
      prepareUpload(img, f);
      updateMode();
    };
    img.onerror = () => {   // mis. HEIC di Chrome: tidak bisa dipratinjau, tetap dikirim apa adanya
      state.previewOk = false;
      $("#stage").hidden = true; $("#noprev").hidden = false;
      setMode("auto");
    };
    img.src = url;
    $("#pick").hidden = true; $("#editor").hidden = false;
    $("#opt-card").checked = false;
    state.mode = "auto";
    setStep(2);
  }

  // Perkecil foto besar di HP sebelum dikirim (lebih cepat; orientasi EXIF sudah diterapkan browser)
  function prepareUpload(img, f) {
    const W = img.naturalWidth, H = img.naturalHeight, M = 2000;
    if (Math.max(W, H) <= M && f.size < 4e6) return;
    const s = Math.min(1, M / Math.max(W, H));
    const c = document.createElement("canvas");
    c.width = Math.round(W * s); c.height = Math.round(H * s);
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    c.toBlob(b => { if (b && state.file === f) state.upload = b; }, "image/jpeg", 0.93);
  }

  // alat lingkari (lasso): pengguna menandai konjungtiva sendiri -> area itu yang dinilai.
  // Paling andal untuk konjungtiva pucat (anemia) yang kadang tidak dikenali AI.
  (() => {
    const stage = $("#stage"), img = $("#preview");
    let pts = null;
    const rel = e => {
      const r = img.getBoundingClientRect();
      return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
    };
    stage.addEventListener("pointerdown", e => {
      if (!state.previewOk || state.mode !== "lasso") return;
      e.preventDefault();
      stage.setPointerCapture(e.pointerId);
      pts = [rel(e)];
      state.poly = null; drawLasso(pts, false);
    });
    stage.addEventListener("pointermove", e => {
      if (!pts) return;
      const p = rel(e), q = pts[pts.length - 1];
      if (Math.hypot(p[0] - q[0], p[1] - q[1]) > 0.004) { pts.push(p); drawLasso(pts, false); }
    });
    const end = () => {
      if (!pts) return;
      const ok = pts.length >= 6 && Math.abs(polyArea(pts)) > 0.0008;
      state.poly = ok ? simplify(pts, 160) : null;
      pts = null;
      drawLasso(state.poly, true); updateMode();
    };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);
  })();
  function polyArea(p) { let a = 0; for (let i = 0; i < p.length; i++) { const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length]; a += x1 * y2 - x2 * y1; } return a / 2; }
  function simplify(p, max) { if (p.length <= max) return p; const s = p.length / max; return Array.from({ length: max }, (_, i) => p[Math.floor(i * s)]); }
  function drawLasso(p, closed) {
    $("#lasso-path").setAttribute("d", p && p.length ? "M" + p.map(q => q[0].toFixed(4) + " " + q[1].toFixed(4)).join("L") + (closed ? "Z" : "") : "");
    $("#lasso").classList.toggle("closed", !!closed);
  }
  $("#lasso-reset").addEventListener("click", () => { state.poly = null; drawLasso(null); updateMode(); });
  $$(".seg-toggle button").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
  function setMode(m) {
    state.mode = m;
    if (m === "auto") { state.poly = null; drawLasso(null); }
    updateMode();
  }
  function updateMode() {
    $$(".seg-toggle button").forEach(b => b.classList.toggle("on", b.dataset.mode === state.mode));
    $("#stage").classList.toggle("drawing", state.mode === "lasso");
    $("#lasso-reset").hidden = !state.poly;
    const hint = $("#mode-hint");
    if (state.mode === "auto") hint.innerHTML = "AI mencari bagian dalam kelopak bawah sendiri. Jika hasilnya meleset, pilih <b>Lingkari sendiri</b>.";
    else if (!state.poly) hint.innerHTML = "<b>Lingkari dengan jari/mouse</b> bagian dalam kelopak bawah: jaringan basah di balik bulu mata (merah muda atau pucat), <b>bukan kulit</b> di bawahnya.";
    else hint.innerHTML = "Lingkaran siap. Area ini yang akan dinilai AI. Tekan <b>Analisis foto</b>, atau lingkari ulang bila kurang pas.";
    const b = $("#analyze");
    if (b) b.disabled = !state.ready || (state.mode === "lasso" && !state.poly);
  }
  $("#change").addEventListener("click", showPicker);

  function updateAnalyzeBtn() {
    const b = $("#analyze");
    if (!b) return;
    b.disabled = !state.ready || (state.mode === "lasso" && !state.poly);
    b.title = state.ready ? "" : "Server AI belum siap";
  }

  /* ---------------------------------------------------------------- analisis */
  $("#analyze").addEventListener("click", () => analyze(false));

  async function analyze(force) {
    if (!state.upload) return;
    setStep(2);
    $("#pick").hidden = true; $("#editor").hidden = true; $("#loading").hidden = false;
    scrollToApp();
    const steps = $$("#load-steps li");
    steps.forEach(li => (li.className = ""));
    let i = 0;
    steps[0].className = "on";
    const timer = setInterval(() => {
      if (i < steps.length - 1) { steps[i].className = "done"; steps[++i].className = "on"; }
    }, 420);
    const fd = new FormData();
    fd.append("file", state.upload, state.file.name || "foto.jpg");
    fd.append("use_card", $("#opt-card").checked);
    fd.append("force", !!force);
    if (state.poly) fd.append("polygon", JSON.stringify(state.poly));
    const t0 = performance.now();
    let data, err;
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      data = await r.json();
      if (!r.ok) err = data.detail || "Terjadi kesalahan di server.";
    } catch (e) {
      err = "Tidak bisa terhubung ke server AI. Pastikan server masih berjalan.";
    }
    const wait = Math.max(0, 1700 - (performance.now() - t0));   // biar animasi tahapan terbaca
    await new Promise(r => setTimeout(r, wait));
    clearInterval(timer);
    steps.forEach(li => (li.className = "done"));
    await new Promise(r => setTimeout(r, 250));
    if (data) data.forced = !!force;
    state.last = data;
    renderResult(err ? null : data, err);
    setStep(3);
    scrollToApp();
    if (!err && data.status === "ok") saveHistory(data);
  }

  /* ---------------------------------------------------------------- hasil */
  const TIER = {
    rendah: { title: "Tidak tampak tanda anemia", sub: "Bagian dalam kelopak mata tidak tampak pucat. Tetap cek Hb bila sering lemas, pusing, atau sesak.", icon: '<path d="M5 12.5l4.5 4.5L19 7.5"/>' },
    sedang: { title: "Ada kemungkinan anemia", sub: "Ada tanda pucat ringan pada konjungtiva. Disarankan cek hemoglobin untuk memastikan.", icon: '<path d="M12 7v6M12 17h.01"/>' },
    tinggi: { title: "Risiko anemia tinggi", sub: "Konjungtiva tampak pucat. Segera periksa hemoglobin di puskesmas atau klinik dalam beberapa hari ini.", icon: '<path d="M12 6v8M12 18h.01"/>' },
    ulang: { title: "Foto perlu diulang", icon: '<path d="M4 4v6h6M20 20v-6h-6"/><path d="M20 10a8 8 0 00-14.3-4.3L4 10M4 14a8 8 0 0014.3 4.3L20 14"/>' },
  };
  const NEXT = {
    rendah: ["Tidak perlu tindakan khusus. Pertahankan makan bergizi seimbang (lauk hewani, sayur hijau, kacang-kacangan).",
      "Bila tetap sering lemas, pusing, pucat, atau sesak, cek hemoglobin di puskesmas.",
      "Ulangi skrining berkala, misalnya tiap bulan untuk ibu hamil dan remaja putri."],
    sedang: ["Lakukan cek hemoglobin (tes darah) di puskesmas, klinik, atau laboratorium untuk memastikan.",
      "Perbanyak makanan sumber zat besi: daging merah, hati, ikan, telur, sayur hijau.",
      "Ibu hamil dan remaja putri: tanyakan tablet tambah darah (TTD) kepada tenaga kesehatan."],
    tinggi: ["Segera cek hemoglobin di puskesmas atau klinik dalam beberapa hari ini.",
      "Bawa hasil ini (simpan sebagai PDF) agar petugas tahu alasan pemeriksaan.",
      "Cari pertolongan segera bila ada sesak napas, jantung berdebar, atau hampir pingsan."],
  };
  const MODEL_NAME = { color: "Warna konjungtiva (regresi logistik)", color_rf: "Warna konjungtiva (Random Forest)", cnn: "Deep learning (EfficientNet-B0)", dino: "Vision transformer (DINOv2)" };
  const ROI_TXT = {
    pred: "Garis hijau: konjungtiva yang ditemukan otomatis oleh U-Net.",
    manual: "Garis hijau: area yang Anda lingkari sendiri.",
    full: "Seluruh foto dinilai (konjungtiva tidak dipotong).",
    none: "Konjungtiva tidak ditemukan pada foto ini.",
  };

  function renderResult(d, err) {
    const el = $("#result");
    const now = new Date();
    const printHead = `<div class="print-head"><b>AnemiaLens · Hasil skrining anemia</b><span>${now.toLocaleString("id-ID")} · ${esc(state.group.name)}</span></div>`;
    if (err || !d) {
      el.className = "t-ulang";
      el.innerHTML = `${head("ulang", "Analisis gagal", esc(err || "Terjadi kesalahan."))}
        <div class="res-actions"><button class="btn btn-rose" data-act="retry">Coba lagi</button><button class="btn btn-ghost" data-act="new">Ganti foto</button></div>`;
      return bindActs();
    }
    if (d.status !== "ok") {
      el.className = "t-ulang";
      el.innerHTML = `${head("ulang", TIER.ulang.title, "Supaya hasilnya bisa dipercaya, AI menolak foto yang tidak memenuhi syarat. Perbaiki hal berikut lalu foto ulang:")}
        <ul class="advice">${(d.advice || []).map(a => `<li>${esc(a)}</li>`).join("")}</ul>
        ${d.overlay ? `<div class="res-grid"><div class="panel"><h4>Foto yang dikirim</h4><div class="sub">${esc(ROI_TXT[d.roi_mode] || "")}</div><img class="overlay-img" src="${d.overlay}" alt="Foto yang dianalisis"></div>
          <div class="panel"><h4>Tips foto yang baik</h4><ul class="sub" style="padding-left:18px;margin:8px 0 0;display:grid;gap:6px">
            <li>Mata memenuhi sebagian besar layar.</li><li>Kelopak bawah ditarik sampai bagian dalam merah muda terlihat lebar.</li>
            <li>Cahaya putih dari depan, tanpa flash dan tanpa filter kamera.</li><li>Atau pilih <b>Lingkari sendiri</b> dan lingkari bagian dalam kelopak bawah.</li></ul></div></div>` : ""}
        <div class="res-actions">
          <button class="btn btn-rose" data-act="new">Ambil foto lagi</button>
          <button class="btn btn-ghost" data-act="lasso">Lingkari sendiri</button>
          <button class="link" data-act="force" style="margin-left:auto">Tetap analisis (kurang dapat dipercaya)</button>
        </div>`;
      return bindActs();
    }

    const tier = d.risk_tier, ts = d.thr_sensitive, tp = d.thr_specific;
    el.className = "t-" + tier;
    const hb = d.hb_estimate, mae = d.hb_mae || 1.4, thr = state.group.thr;
    const lo = 6, hi = 18, pos = v => ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * 100;
    const hbVerdict = hb == null ? "" : (thr >= hb - mae && thr <= hb + mae)
      ? "Kisaran perkiraan <b>mencakup batas</b>, jadi belum bisa dipastikan di atas atau di bawah batas."
      : hb < thr ? "Perkiraan <b>di bawah</b> batas." : "Perkiraan di atas batas.";
    const warns = [...(d.quality_warnings || [])];
    if (d.forced) warns.unshift("Foto ini sebenarnya tidak lolos pemeriksaan kualitas dan dianalisis paksa. Jangan jadikan acuan.");
    if (d.card_used) warns.push("Kartu putih terdeteksi dan dipakai untuk mengoreksi warna.");
    const models = Object.entries(d.per_model || {}).filter(([k]) => (state.info?.models || Object.keys(d.per_model)).includes(k));

    el.innerHTML = `${printHead}
      ${head(tier, TIER[tier].title, esc(TIER[tier].sub || d.message))}
      ${warns.length ? `<div class="warn">${warns.map(esc).join("<br>")}</div>` : ""}
      <div class="res-grid">
        <div class="panel">
          <h4>Skor risiko anemia</h4>
          <div class="sub">Gabungan ${models.length} model AI. Ambang dipilih agar ≥ 85% kasus anemia tertangkap.</div>
          <div class="gauge-wrap">${gauge(d.probability, ts, tp)}<div class="gauge-legend"><span>Rendah</span><span>Sedang</span><span>Tinggi</span></div></div>
        </div>
        <div class="panel">
          <h4>Perkiraan hemoglobin</h4>
          <div class="sub">Perkiraan kasar dari foto, bukan hasil lab.</div>
          ${hb != null ? `
          <div class="hb-big"><b>${fmt(hb, 1)}</b><span>g/dL</span></div>
          <div class="sub">Kisaran wajar ${fmt(hb - mae, 1)}–${fmt(hb + mae, 1)} g/dL (± selisih rata-rata model)</div>
          <div class="hb-scale">
            <div class="hb-track"></div>
            <div class="hb-band" style="left:${pos(hb - mae)}%;width:${pos(hb + mae) - pos(hb - mae)}%"></div>
            <div class="hb-thr" style="left:${pos(thr)}%"><span>batas ${fmt(thr, thr % 1 ? 1 : 0)}</span></div>
            <div class="hb-dot" style="left:${pos(hb)}%"></div>
            <div class="hb-ticks"><span>6</span><span>9</span><span>12</span><span>15</span><span>18</span></div>
          </div>
          <div class="sub" style="margin-top:6px">Batas anemia WHO untuk <b>${esc(state.group.name)}</b>: ${fmt(thr, thr % 1 ? 1 : 0)} g/dL. ${hbVerdict}</div>` : `<p class="sub">Tidak tersedia.</p>`}
        </div>
        <div class="panel">
          <h4>Area yang dianalisis</h4>
          <div class="sub">${esc(ROI_TXT[d.roi_mode] || "")}${d.card_used ? " Kotak biru: kartu putih." : ""}</div>
          ${d.overlay ? `<img class="overlay-img" src="${d.overlay}" alt="Foto dengan area konjungtiva ditandai">` : ""}
          ${d.roi_mode === "pred" ? `<button class="link fixline" data-act="lasso">Garis hijau meleset? Lingkari sendiri lalu analisis ulang</button>` : ""}
        </div>
        <div class="panel">
          <h4>Transparansi: suara tiap model</h4>
          <div class="sub">Skor akhir = rata-rata model di bawah.</div>
          <div class="models">${models.map(([k, v]) => `<div class="mrow"><span>${esc(MODEL_NAME[k] || k)}</span><b>${Math.round(v * 100)}</b><div class="bar"><i style="width:${Math.max(2, v * 100)}%"></i></div></div>`).join("")}</div>
        </div>
      </div>
      <div class="panel next"><h4>Langkah selanjutnya</h4><ol>${NEXT[tier].map(t => `<li>${esc(t)}</li>`).join("")}</ol></div>
      <div class="disclaimer"><svg viewBox="0 0 24 24" class="ic"><circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/></svg><span>${esc(d.disclaimer || "")}</span></div>
      <div class="res-actions">
        <button class="btn btn-ink" data-act="print"><svg viewBox="0 0 24 24" class="ic"><path d="M7 9V3h10v6M7 18H4v-7h16v7h-3M7 14h10v7H7z"/></svg>Simpan hasil (PDF)</button>
        <button class="btn btn-rose" data-act="new">Cek foto lain</button>
        <button class="btn btn-ghost" data-act="lasso">Lingkari ulang</button>
      </div>
      <div class="meta">Diproses ${fmt((d.latency_ms || 0) / 1000, 2)} detik · ${new Date().toLocaleString("id-ID")}</div>`;
    bindActs();
  }
  function head(tier, title, text) {
    return `<div class="res-head"><span class="badge"><svg viewBox="0 0 24 24" class="ic">${TIER[tier].icon}</svg></span><div><h3>${title}</h3><p>${text}</p></div></div>`;
  }
  function bindActs() {
    $$("#result [data-act]").forEach(b => b.addEventListener("click", () => {
      const a = b.dataset.act;
      if (a === "print") window.print();
      if (a === "new") { setStep(2); showPicker(); scrollToApp(); }
      if (a === "retry") analyze(false);
      if (a === "lasso") {
        setStep(2); $("#pick").hidden = true; $("#loading").hidden = true; $("#editor").hidden = false;
        if (state.previewOk) { state.poly = null; drawLasso(null); setMode("lasso"); }
        scrollToApp();
      }
      if (a === "force") analyze(true);
    }));
  }

  /* ---------------------------------------------------------------- riwayat (hanya di perangkat ini) */
  function readHist() { try { return JSON.parse(localStorage.getItem(HIST_KEY) || "[]"); } catch { return []; } }
  function writeHist(h) { try { localStorage.setItem(HIST_KEY, JSON.stringify(h)); } catch { /* penyimpanan penuh/diblokir */ } }
  function saveHistory(d) {
    const done = thumb => {
      const h = readHist();
      h.unshift({ t: Date.now(), tier: d.risk_tier, p: d.probability, hb: d.hb_estimate, g: state.group.name, thumb });
      writeHist(h.slice(0, 12));
      renderHist();
    };
    if (!d.overlay) return done(null);
    const im = new Image();
    im.onload = () => {
      const c = document.createElement("canvas"), s = 160 / Math.max(im.width, im.height);
      c.width = im.width * s; c.height = im.height * s;
      c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
      done(c.toDataURL("image/jpeg", 0.7));
    };
    im.onerror = () => done(null);
    im.src = d.overlay;
  }
  const TIER_LBL = { rendah: "Risiko rendah", sedang: "Risiko sedang", tinggi: "Risiko tinggi" };
  function renderHist() {
    const h = readHist();
    $("#riwayat-sec").hidden = !h.length; $("#nav-riwayat").hidden = !h.length;
    $("#hist").innerHTML = h.map(x => `<div class="hcard">${x.thumb ? `<img src="${x.thumb}" alt="">` : "<img alt=''>"}<div>
      <b>${x.hb != null ? "Hb ≈ " + fmt(x.hb, 1) + " g/dL" : "Skor " + Math.round(x.p * 100)}</b>
      <small>${new Date(x.t).toLocaleDateString("id-ID", { day: "numeric", month: "short", year: "numeric" })} · ${esc(x.g || "")}</small><br>
      <span class="tag ${x.tier}">${TIER_LBL[x.tier] || x.tier}</span></div></div>`).join("");
  }
  $("#hist-clear").addEventListener("click", () => { writeHist([]); renderHist(); });

  function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  /* ---------------------------------------------------------------- mulai */
  $("#hero-gauge").innerHTML = gauge(0.55, 0.43, 0.61, { w: 16, label: false });
  renderHist();
  setStep(1);
  updateAnalyzeBtn();
  loadInfo();
})();
