/**
 * learning.js -- The "Learning" dashboard for the Hide & Seek 2.0 viewer.
 *
 * A self-contained ES module (NO build step, NO chart library) that fetches
 * ./learning.json and renders a friendly, beginner-readable story of how the
 * two teams (Hiders / Seekers) co-evolved through self-play training:
 *
 *   (a) a one/two-sentence intro + a muted "this data is illustrative" note,
 *   (b) two team SKILL CARDS (big ELO, win-rate %, animated progress bar,
 *       the team's tactic) with numbers that COUNT UP and bars that fill,
 *   (c) an ARMS-RACE line chart (hider vs seeker win-rate % over steps) drawn
 *       on a devicePixelRatio-aware Canvas 2D, with milestone markers,
 *   (d) an ELO line chart (hider vs seeker ELO over steps), same style,
 *   (e) a MILESTONE TABLE -- the key "see the progress" piece.
 *
 * Public API:
 *   export async function renderLearning(rootEl, opts)
 *     - rootEl : the container element to build the dashboard into
 *     - opts   : {
 *         theme:        "dark" | "light",         // current theme name
 *         getThemeColors: () => ({...}),          // live theme color lookup
 *         reducedMotion: boolean,                 // honor prefers-reduced-motion
 *         url:          string                    // override learning.json URL
 *       }
 *     Lazily fetches + builds ONCE; subsequent calls re-theme / redraw in place.
 *     Returns a small handle: { redraw(), setTheme(name), destroy() }.
 *
 * The charts pull their colors from the live theme (via getThemeColors) so a
 * dark/light toggle re-skins them instantly; they also redraw on container
 * resize (ResizeObserver) and when setTheme() is called.
 */

// ---------------------------------------------------------------------------
// Number / step formatting helpers (compact, friendly).
// ---------------------------------------------------------------------------

/** 50_000_000 -> "50M", 1_500_000 -> "1.5M", 1551 -> "1551", 0 -> "0". */
function fmtSteps(n) {
  const v = Math.abs(n);
  if (v >= 1e9) return trimZero(n / 1e9) + "B";
  if (v >= 1e6) return trimZero(n / 1e6) + "M";
  if (v >= 1e3) return trimZero(n / 1e3) + "K";
  return String(Math.round(n));
}

/** Trim a trailing ".0" so 50.0 -> "50" but 1.5 stays "1.5". */
function trimZero(x) {
  const r = Math.round(x * 10) / 10;
  return Number.isInteger(r) ? String(r) : r.toFixed(1);
}

/** 0.62 -> "62%". */
function fmtPct(frac) {
  return Math.round(frac * 100) + "%";
}

/** 1234567 -> "1,234,567" (used for the headline total-timesteps line). */
function fmtComma(n) {
  return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Map the short emoji "codes" in learning.json (e.g. "fort", "run", "decoy")
 * to actual emoji glyphs. The JSON ships codes (not glyphs) so it stays ASCII;
 * we resolve them here for display. Unknown codes fall back to a sparkle.
 */
const EMOJI = {
  run: "\u{1F3C3}",    // runner
  hide: "\u{1F648}",   // see-no-evil monkey
  fort: "\u{1F9F1}",   // brick
  coop: "\u{1F91D}",   // handshake
  ramp: "\u{1FA9C}",   // ladder
  lock: "\u{1F512}",   // lock
  decoy: "\u{1FA84}",  // magic wand
  door: "\u{1F6AA}",   // door
};
function emojiFor(code) {
  return EMOJI[code] || "✨"; // sparkles fallback
}

// ---------------------------------------------------------------------------
// Easing + tiny animation runner (rAF based, cancellable).
// ---------------------------------------------------------------------------

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * Run an animation from 0..1 over `dur` ms, calling onUpdate(progress) each
 * frame and onDone() at the end. Returns a cancel function. If reduced motion
 * is requested, it jumps straight to 1 (no animation).
 */
function animateValue(dur, onUpdate, reduced, onDone) {
  if (reduced || dur <= 0) {
    onUpdate(1);
    if (onDone) onDone();
    return () => {};
  }
  let raf = 0;
  let cancelled = false;
  const start = performance.now();
  const tick = (now) => {
    if (cancelled) return;
    const t = Math.min(1, (now - start) / dur);
    onUpdate(easeOutCubic(t));
    if (t < 1) {
      raf = requestAnimationFrame(tick);
    } else if (onDone) {
      onDone();
    }
  };
  raf = requestAnimationFrame(tick);
  return () => {
    cancelled = true;
    if (raf) cancelAnimationFrame(raf);
  };
}

// ---------------------------------------------------------------------------
// Small DOM helper.
// ---------------------------------------------------------------------------

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

/**
 * Build (or re-theme) the Learning dashboard inside rootEl.
 * @param {HTMLElement} rootEl
 * @param {object} opts
 * @returns {Promise<{redraw:Function,setTheme:Function,destroy:Function}>}
 */
export async function renderLearning(rootEl, opts = {}) {
  // Reuse an already-built dashboard if one exists on this root: just re-theme.
  if (rootEl.__hns2Learning) {
    const inst = rootEl.__hns2Learning;
    if (opts.theme) inst.setTheme(opts.theme);
    if (opts.getThemeColors) inst._setColors(opts.getThemeColors);
    inst.redraw();
    inst.replayIntro();
    return inst;
  }

  const url = opts.url || "./learning.json";
  let getColors = opts.getThemeColors || defaultColors;
  let reduced = !!opts.reducedMotion;

  // --- fetch the data ----------------------------------------------------
  let data;
  try {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    data = await res.json();
  } catch (err) {
    rootEl.innerHTML = "";
    const card = el("div", "learn-error",
      "<b>Could not load learning data.</b><br/>" +
      "Expected <code>learning.json</code> next to the page. (" +
      (err && err.message ? err.message : "fetch failed") + ")");
    rootEl.appendChild(card);
    return { redraw() {}, setTheme() {}, destroy() {}, replayIntro() {} };
  }

  const meta = data.meta || {};
  const series = data.series || {};
  const milestones = Array.isArray(data.milestones) ? data.milestones.slice() : [];
  const teams = data.teams || {};
  milestones.sort((a, b) => (a.step || 0) - (b.step || 0));

  // --- scaffold the DOM --------------------------------------------------
  rootEl.innerHTML = "";
  const wrap = el("div", "learn-wrap");
  rootEl.appendChild(wrap);

  // (a) intro
  const intro = el("div", "learn-intro");
  intro.appendChild(el("h2", "learn-h",
    meta.title ? escapeHtml(meta.title) : "How the agents learned"));
  intro.appendChild(el("p", "learn-lead",
    "These agents learn by playing hide-and-seek against past versions of " +
    "themselves. As one team finds a new trick, the other adapts — an " +
    "arms race. Here’s how they progressed."));
  const noteBits = [];
  if (meta.total_timesteps) {
    noteBits.push("Trained for ~" + fmtComma(meta.total_timesteps) + " " +
      (meta.unit || "steps") + ".");
  }
  if (meta.note) noteBits.push(escapeHtml(meta.note));
  if (noteBits.length) {
    intro.appendChild(el("p", "learn-note", noteBits.join(" ")));
  }
  wrap.appendChild(intro);

  // (b) team skill cards
  const cards = el("div", "learn-cards");
  const hiderCard = buildTeamCard("hider", "Hiders", teams.hider || {});
  const seekerCard = buildTeamCard("seeker", "Seekers", teams.seeker || {});
  cards.appendChild(hiderCard.root);
  cards.appendChild(seekerCard.root);
  wrap.appendChild(cards);

  // (c) arms-race win-rate chart
  const wrSection = buildChartSection(
    "The arms race",
    "Win rate over training. Watch the lines cross as each team overtakes the other.",
    "learn-chart-winrate"
  );
  wrap.appendChild(wrSection.root);

  // (d) ELO chart
  const eloSection = buildChartSection(
    "Skill over time (ELO)",
    "Both teams keep getting stronger; the gap swings as new tactics emerge.",
    "learn-chart-elo"
  );
  wrap.appendChild(eloSection.root);

  // (e) milestone table
  const tableSection = el("div", "learn-section");
  tableSection.appendChild(el("h3", "learn-h3", "Milestones — what they learned"));
  tableSection.appendChild(el("p", "learn-sub",
    "Each new behaviour emerged on its own from the competition — nobody " +
    "scripted these."));
  tableSection.appendChild(buildMilestoneTable(milestones));
  wrap.appendChild(tableSection);

  // --- chart objects (canvas + draw fns) ---------------------------------
  const winrateChart = makeLineChart(wrSection.canvas, {
    kind: "winrate",
    xs: series.t || [],
    lines: [
      { key: "hider", ys: series.hider_winrate || [] },
      { key: "seeker", ys: series.seeker_winrate || [] },
    ],
    yMin: 0,
    yMax: 1,
    yTicks: [0, 0.25, 0.5, 0.75, 1],
    yFmt: (v) => fmtPct(v),
    milestones,
    legend: [["hider", "Hiders"], ["seeker", "Seekers"]],
  });

  const eloChart = makeLineChart(eloSection.canvas, {
    kind: "elo",
    xs: series.t || [],
    lines: [
      { key: "hider", ys: series.hider_elo || [] },
      { key: "seeker", ys: series.seeker_elo || [] },
    ],
    // pad the ELO range a little for headroom
    yMin: niceFloor(minOf(series.hider_elo, series.seeker_elo) - 20),
    yMax: niceCeil(maxOf(series.hider_elo, series.seeker_elo) + 20),
    yTicks: null, // auto
    yFmt: (v) => String(Math.round(v)),
    milestones,
    legend: [["hider", "Hiders"], ["seeker", "Seekers"]],
  });

  // --- the instance handle ----------------------------------------------
  let theme = opts.theme || "dark";
  let cancelers = [];

  function cancelAll() {
    cancelers.forEach((c) => { try { c(); } catch (_) {} });
    cancelers = [];
  }

  function applyColors() {
    const c = getColors();
    // recolor the static team-card chrome that CSS can't reach via vars
    hiderCard.applyColor(c.hider);
    seekerCard.applyColor(c.seeker);
    winrateChart.setColors(c);
    eloChart.setColors(c);
  }

  function redraw() {
    applyColors();
    winrateChart.resize();
    eloChart.resize();
    winrateChart.draw(winrateChart.progress);
    eloChart.draw(eloChart.progress);
  }

  function replayIntro() {
    cancelAll();
    // count-up the card numbers + fill the bars
    cancelers.push(hiderCard.animateIn(reduced));
    cancelers.push(seekerCard.animateIn(reduced));
    // draw the chart lines in left-to-right
    cancelers.push(winrateChart.animateIn(reduced));
    cancelers.push(eloChart.animateIn(reduced));
  }

  const inst = {
    redraw,
    replayIntro,
    setTheme(name) { theme = name; applyColors(); redraw(); },
    _setColors(fn) { getColors = fn || getColors; },
    destroy() {
      cancelAll();
      winrateChart.destroy();
      eloChart.destroy();
      if (ro) ro.disconnect();
      delete rootEl.__hns2Learning;
      rootEl.innerHTML = "";
    },
  };

  // redraw on container resize (debounced via rAF inside ResizeObserver)
  let ro = null;
  if (typeof ResizeObserver !== "undefined") {
    let pending = false;
    ro = new ResizeObserver(() => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        redraw();
      });
    });
    ro.observe(wrap);
  }

  rootEl.__hns2Learning = inst;

  // initial paint + entrance animation
  applyColors();
  winrateChart.resize();
  eloChart.resize();
  replayIntro();

  return inst;
}

// ---------------------------------------------------------------------------
// Team skill card.
// ---------------------------------------------------------------------------

function buildTeamCard(teamKey, label, info) {
  const root = el("div", "team-card team-" + teamKey);
  const elo = Math.round(info.elo || 0);
  const winrate = typeof info.winrate === "number" ? info.winrate : 0;
  const tactic = info.tactic || "";

  root.innerHTML = `
    <div class="tc-head">
      <span class="tc-dot"></span>
      <span class="tc-name">${escapeHtml(label)}</span>
      <span class="tc-elo-lbl">ELO</span>
    </div>
    <div class="tc-elo"><span class="tc-elo-num">0</span></div>
    <div class="tc-bar"><i class="tc-bar-fill"></i><span class="tc-bar-pct">0%</span></div>
    <div class="tc-meta">
      <span class="tc-wr">Win rate <b class="tc-wr-num">0%</b></span>
    </div>
    <div class="tc-tactic">${tactic ? "“" + escapeHtml(tactic) + "”" : ""}</div>
  `;

  const eloNum = root.querySelector(".tc-elo-num");
  const barFill = root.querySelector(".tc-bar-fill");
  const barPct = root.querySelector(".tc-bar-pct");
  const wrNum = root.querySelector(".tc-wr-num");

  function applyColor(hex) {
    const css = "#" + (hex >>> 0).toString(16).padStart(6, "0");
    root.style.setProperty("--team", css);
  }

  function animateIn(reduced) {
    // Count ELO up and win-rate up together; fill the bar to the win rate.
    return animateValue(1100, (p) => {
      eloNum.textContent = String(Math.round(elo * p));
      const wr = winrate * p;
      wrNum.textContent = fmtPct(wr);
      const pctW = Math.round(wr * 100);
      barFill.style.width = pctW + "%";
      barPct.textContent = pctW + "%";
    }, reduced);
  }

  return { root, applyColor, animateIn };
}

// ---------------------------------------------------------------------------
// Chart section scaffold (title + sub + canvas).
// ---------------------------------------------------------------------------

function buildChartSection(title, sub, canvasCls) {
  const root = el("div", "learn-section");
  root.appendChild(el("h3", "learn-h3", escapeHtml(title)));
  root.appendChild(el("p", "learn-sub", escapeHtml(sub)));
  const holder = el("div", "chart-holder");
  const canvas = document.createElement("canvas");
  canvas.className = canvasCls;
  holder.appendChild(canvas);
  root.appendChild(holder);
  return { root, canvas };
}

// ---------------------------------------------------------------------------
// Milestone table.
// ---------------------------------------------------------------------------

function buildMilestoneTable(milestones) {
  const table = el("table", "learn-table");
  table.innerHTML = `
    <thead>
      <tr>
        <th class="col-num">#</th>
        <th class="col-when">Emerged at</th>
        <th class="col-team">Team</th>
        <th class="col-beh">Behaviour</th>
        <th class="col-desc">What they learned</th>
      </tr>
    </thead>
  `;
  const tbody = el("tbody");
  milestones.forEach((m, i) => {
    const tr = el("tr");
    const teamKey = m.team === "seeker" ? "seeker" : "hider";
    const teamLabel = teamKey === "seeker" ? "Seekers" : "Hiders";
    tr.innerHTML = `
      <td class="col-num">${i + 1}</td>
      <td class="col-when">${fmtSteps(m.step || 0)}</td>
      <td class="col-team"><span class="team-chip chip-${teamKey}">${teamLabel}</span></td>
      <td class="col-beh"><span class="beh-emoji">${emojiFor(m.emoji)}</span> ${escapeHtml(m.title || "")}</td>
      <td class="col-desc">${escapeHtml(m.desc || "")}</td>
    `;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

// ---------------------------------------------------------------------------
// Canvas 2D line chart (devicePixelRatio-aware, themed, animated draw-in).
// ---------------------------------------------------------------------------

function makeLineChart(canvas, cfg) {
  const ctx = canvas.getContext("2d");
  let colors = defaultColors();
  let dpr = 1;
  let W = 0, H = 0;       // CSS pixel size
  let cancelAnim = null;

  const chart = {
    progress: 1,
    setColors(c) { colors = c; },
    resize,
    draw,
    animateIn,
    destroy() { if (cancelAnim) cancelAnim(); },
  };

  function resize() {
    const holder = canvas.parentElement;
    const cssW = Math.max(120, (holder ? holder.clientWidth : 600));
    // Aspect: keep charts a comfortable height, clamp for very wide/narrow.
    const cssH = Math.max(180, Math.min(300, Math.round(cssW * 0.42)));
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = cssW; H = cssH;
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
  }

  // Plot-area padding (CSS px).
  const PAD = { l: 46, r: 16, t: 16, b: 34 };

  function plotRect() {
    return {
      x: PAD.l,
      y: PAD.t,
      w: Math.max(10, W - PAD.l - PAD.r),
      h: Math.max(10, H - PAD.t - PAD.b),
    };
  }

  const xs = cfg.xs;
  const xMin = xs.length ? xs[0] : 0;
  const xMax = xs.length ? xs[xs.length - 1] : 1;

  function xPix(v, r) {
    const t = xMax > xMin ? (v - xMin) / (xMax - xMin) : 0;
    return r.x + t * r.w;
  }
  function yPix(v, r) {
    const t = cfg.yMax > cfg.yMin ? (v - cfg.yMin) / (cfg.yMax - cfg.yMin) : 0;
    return r.y + (1 - t) * r.h;
  }

  function lineColor(key) {
    const hex = colors[key] != null ? colors[key] : 0x888888;
    return hexCss(hex);
  }

  /** Build the X axis tick values: compact, evenly spaced across the range. */
  function xTicks() {
    // Aim for ~5 ticks at round step boundaries (0, 50M, 100M, ...).
    const span = xMax - xMin;
    if (span <= 0) return [xMin];
    const target = 4;
    const raw = span / target;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    let step;
    if (norm < 1.5) step = 1 * mag;
    else if (norm < 3) step = 2.5 * mag;
    else if (norm < 7) step = 5 * mag;
    else step = 10 * mag;
    const ticks = [];
    let v = Math.ceil(xMin / step) * step;
    for (; v <= xMax + 1e-6; v += step) ticks.push(v);
    if (ticks[0] > xMin + 1e-6) ticks.unshift(xMin);
    return ticks;
  }

  function yTickVals() {
    if (cfg.yTicks) return cfg.yTicks;
    // auto: 4 evenly spaced ticks
    const out = [];
    const n = 4;
    for (let i = 0; i <= n; i++) out.push(cfg.yMin + (i / n) * (cfg.yMax - cfg.yMin));
    return out;
  }

  /**
   * Draw the whole chart. `progress` in 0..1 reveals the lines left-to-right
   * (each line is clipped to progress * range along X) and fades them up.
   */
  function draw(progress) {
    chart.progress = progress;
    if (!W || !H) resize();
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const r = plotRect();
    const grid = hexCss(colors.chartGrid != null ? colors.chartGrid : 0x232c3d);
    const axis = hexCss(colors.chartAxis != null ? colors.chartAxis : 0x8a98ad);
    const text = hexCss(colors.chartText != null ? colors.chartText : 0x8a98ad);

    // --- gridlines + y labels ---
    ctx.lineWidth = 1;
    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    const yticks = yTickVals();
    for (const yv of yticks) {
      const py = Math.round(yPix(yv, r)) + 0.5;
      ctx.strokeStyle = grid;
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.moveTo(r.x, py);
      ctx.lineTo(r.x + r.w, py);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = text;
      ctx.fillText(cfg.yFmt(yv), r.x - 8, py);
    }

    // --- x labels ---
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const xv of xTicks()) {
      const px = xPix(xv, r);
      ctx.fillStyle = text;
      ctx.fillText(fmtStepLabel(xv), px, r.y + r.h + 8);
    }

    // --- axes baseline ---
    ctx.strokeStyle = axis;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(r.x, r.y + r.h + 0.5);
    ctx.lineTo(r.x + r.w, r.y + r.h + 0.5);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // --- milestone vertical markers ---
    if (cfg.milestones && cfg.milestones.length) {
      ctx.textAlign = "center";
      for (const m of cfg.milestones) {
        const step = m.step || 0;
        if (step < xMin || step > xMax) continue;
        // only reveal markers the line has "reached"
        const reachedX = xMin + (xMax - xMin) * progress;
        if (step > reachedX + 1e-6) continue;
        const px = Math.round(xPix(step, r)) + 0.5;
        const teamKey = m.team === "seeker" ? "seeker" : "hider";
        ctx.strokeStyle = lineColor(teamKey);
        ctx.globalAlpha = 0.28;
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(px, r.y);
        ctx.lineTo(px, r.y + r.h);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        // emoji glyph at the top of the marker
        ctx.font = "13px Inter, system-ui, sans-serif";
        ctx.fillText(emojiFor(m.emoji), px, r.y - 2);
      }
    }

    // --- the data lines ---
    const reveal = progress;
    for (const line of cfg.lines) {
      const ys = line.ys;
      if (!ys || ys.length < 2) continue;
      ctx.lineWidth = 2.4;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = lineColor(line.key);
      ctx.globalAlpha = 0.25 + 0.75 * reveal; // fade up while drawing
      ctx.beginPath();
      const maxX = xMin + (xMax - xMin) * reveal;
      let started = false;
      for (let i = 0; i < xs.length; i++) {
        const xv = xs[i];
        if (xv > maxX) {
          // interpolate the final partial segment to the reveal edge
          if (i > 0) {
            const x0 = xs[i - 1], x1 = xs[i];
            const t = (maxX - x0) / (x1 - x0);
            const yv = ys[i - 1] + (ys[i] - ys[i - 1]) * t;
            ctx.lineTo(xPix(maxX, r), yPix(yv, r));
          }
          break;
        }
        const px = xPix(xv, r);
        const py = yPix(ys[i], r);
        if (!started) { ctx.moveTo(px, py); started = true; }
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      // endpoint dot at the revealed tip
      const tipX = xPix(Math.min(maxX, xMax), r);
      const tipY = endpointY(line.ys, maxX, r);
      if (tipY != null) {
        ctx.fillStyle = lineColor(line.key);
        ctx.beginPath();
        ctx.arc(tipX, tipY, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // --- legend ---
    drawLegend(r, text);

    ctx.restore();
  }

  function endpointY(ys, maxX, r) {
    if (!ys || !ys.length) return null;
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] >= maxX) {
        if (i === 0) return yPix(ys[0], r);
        const x0 = xs[i - 1], x1 = xs[i];
        const t = x1 > x0 ? (maxX - x0) / (x1 - x0) : 0;
        const yv = ys[i - 1] + (ys[i] - ys[i - 1]) * t;
        return yPix(yv, r);
      }
    }
    return yPix(ys[ys.length - 1], r);
  }

  function drawLegend(r, text) {
    const items = cfg.legend || [];
    if (!items.length) return;
    ctx.font = "600 11px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    let lx = r.x + 6;
    const ly = r.y + 10;
    for (const [key, lbl] of items) {
      ctx.fillStyle = lineColor(key);
      ctx.beginPath();
      ctx.arc(lx + 4, ly, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = text;
      ctx.fillText(lbl, lx + 13, ly + 0.5);
      lx += 13 + ctx.measureText(lbl).width + 18;
    }
  }

  function animateIn(reduced) {
    if (cancelAnim) cancelAnim();
    cancelAnim = animateValue(1200, (p) => draw(p), reduced);
    return cancelAnim;
  }

  return chart;
}

// ---------------------------------------------------------------------------
// Misc helpers.
// ---------------------------------------------------------------------------

/** Compact axis label: keep it tidy (0, 50M, 100M, ...). */
function fmtStepLabel(v) {
  if (v === 0) return "0";
  return fmtSteps(v);
}

function hexCss(hex) {
  if (typeof hex === "string") return hex;
  return "#" + (hex >>> 0).toString(16).padStart(6, "0");
}

function minOf() {
  let m = Infinity;
  for (const arr of arguments) {
    if (!arr) continue;
    for (const v of arr) if (v < m) m = v;
  }
  return m === Infinity ? 0 : m;
}
function maxOf() {
  let m = -Infinity;
  for (const arr of arguments) {
    if (!arr) continue;
    for (const v of arr) if (v > m) m = v;
  }
  return m === -Infinity ? 1 : m;
}
function niceFloor(v) { return Math.floor(v / 10) * 10; }
function niceCeil(v) { return Math.ceil(v / 10) * 10; }

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Fallback theme colors if the host doesn't supply getThemeColors. These mirror
 * the DARK palette so the dashboard is legible even standalone.
 */
function defaultColors() {
  return {
    hider: 0x43b6ff,
    seeker: 0xff6b6b,
    chartGrid: 0x232c3d,
    chartAxis: 0x8a98ad,
    chartText: 0x8a98ad,
  };
}
