/*
 * livetrain.js -- REAL in-browser self-play reinforcement learning.
 *
 * Two tabular Q-learning agents (a seeker and a hider) learn hide-and-seek on a
 * grid with walls and line-of-sight, live, by playing each other. Nothing is
 * scripted: click "Train" and watch them go from random to skilled while the
 * learning curve climbs. This is a direct JS port of learn/train.py (which is
 * proven to converge), so the SAME learning happens in the browser.
 *
 * Structure:
 *   - The RL core (GridEnv, Trainer) uses NO DOM -> it can be unit-tested in Node.
 *   - mountTrain(rootEl) builds the live UI (charts + a mini-game) -- browser only.
 */

export const N = 9;
export const MAX_STEPS = 40;
export const SEE_RADIUS = 3;
const NA = 5;
const ACTIONS = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]; // stay,N,S,E,W
const WALL_CELLS = [
  [4, 1], [4, 2], [4, 3], [2, 5], [3, 5], [4, 5],
  [6, 4], [6, 5], [6, 6], [1, 7], [2, 7],
];
const WALL = new Uint8Array(N * N);
for (const [x, y] of WALL_CELLS) WALL[y * N + x] = 1;

function free(x, y) { return x >= 0 && x < N && y >= 0 && y < N && !WALL[y * N + x]; }
const FREE = [];
for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (free(x, y)) FREE.push([x, y]);

/** Deterministic RNG (mulberry32) so Node tests are reproducible. */
export function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const randint = (rng, n) => (rng() * n) | 0;

function losClear(x0, y0, x1, y1) {
  let dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0);
  let sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx - dy;
  let x = x0, y = y0;
  while (x !== x1 || y !== y1) {
    const e2 = 2 * err;
    if (e2 > -dy) { err -= dy; x += sx; }
    if (e2 < dx) { err += dx; y += sy; }
    if (x === x1 && y === y1) break;
    if (WALL[y * N + x]) return false;
  }
  return true;
}

export class GridEnv {
  constructor(rng) { this.rng = rng; this.sx = this.sy = this.hx = this.hy = 0; this.t = 0; }
  reset() {
    do {
      [this.sx, this.sy] = FREE[randint(this.rng, FREE.length)];
      [this.hx, this.hy] = FREE[randint(this.rng, FREE.length)];
    } while (Math.max(Math.abs(this.sx - this.hx), Math.abs(this.sy - this.hy)) < 3);
    this.t = 0;
    return this.state();
  }
  state() { return [this.sy * N + this.sx, this.hy * N + this.hx]; }
  canSee() {
    if (Math.max(Math.abs(this.sx - this.hx), Math.abs(this.sy - this.hy)) > SEE_RADIUS) return false;
    return losClear(this.sx, this.sy, this.hx, this.hy);
  }
  step(as, ah) {
    let [dxs, dys] = ACTIONS[as];
    if (free(this.sx + dxs, this.sy + dys)) { this.sx += dxs; this.sy += dys; }
    let [dxh, dyh] = ACTIONS[ah];
    if (free(this.hx + dxh, this.hy + dyh)) { this.hx += dxh; this.hy += dyh; }
    this.t++;
    const see = this.canSee();
    const caught = see && (Math.abs(this.sx - this.hx) + Math.abs(this.sy - this.hy) <= 1);
    const r_s = caught ? 3.0 : (see ? 1.0 : -0.05);
    const done = caught || this.t >= MAX_STEPS;
    return { see, caught, r_s, done };
  }
}

const NS = N * N;
const qi = (cs, ch, a) => (cs * NS + ch) * NA + a;
function greedy(Q, cs, ch) {
  let b = 0, bv = Q[qi(cs, ch, 0)];
  for (let a = 1; a < NA; a++) { const v = Q[qi(cs, ch, a)]; if (v > bv) { bv = v; b = a; } }
  return b;
}
function maxQ(Q, cs, ch) {
  let m = Q[qi(cs, ch, 0)];
  for (let a = 1; a < NA; a++) { const v = Q[qi(cs, ch, a)]; if (v > m) m = v; }
  return m;
}

export class Trainer {
  constructor(seed = 0, total = 40000) { this.total = total; this.reset(seed); }
  reset(seed = 0) {
    this.Qs = new Float32Array(NS * NS * NA);
    this.Qh = new Float32Array(NS * NS * NA);
    this.episode = 0;
    this.curve = [{ e: 0, s: 0.11, h: 0.90 }];
    this.rng = makeRng((seed | 0) || 1);
    this.env = new GridEnv(this.rng);
    this.alpha = 0.25; this.gamma = 0.95;
  }
  trainOne(eps) {
    const env = this.env, Qs = this.Qs, Qh = this.Qh, a = this.alpha, g = this.gamma;
    env.reset();
    let [cs, ch] = env.state();
    for (let t = 0; t < MAX_STEPS; t++) {
      const as = this.rng() < eps ? randint(this.rng, NA) : greedy(Qs, cs, ch);
      const ah = this.rng() < eps ? randint(this.rng, NA) : greedy(Qh, cs, ch);
      const { r_s, done } = env.step(as, ah);
      const [cs2, ch2] = env.state();
      const tS = r_s + (done ? 0 : g * maxQ(Qs, cs2, ch2));
      const tH = -r_s + (done ? 0 : g * maxQ(Qh, cs2, ch2));
      const iS = qi(cs, ch, as), iH = qi(cs, ch, ah);
      Qs[iS] += a * (tS - Qs[iS]);
      Qh[iH] += a * (tH - Qh[iH]);
      cs = cs2; ch = ch2;
      if (done) break;
    }
    this.episode++;
  }
  /** Train n episodes with an annealed epsilon. */
  trainMany(n) {
    for (let i = 0; i < n && this.episode < this.total; i++) {
      const eps = Math.max(0.05, 0.35 * (1 - this.episode / this.total));
      this.trainOne(eps);
    }
  }
  /** Skill vs a RANDOM opponent: a clean learning signal. */
  evalSkill(episodes = 150) {
    const rng = makeRng(123), env = new GridEnv(rng);
    let seen = 0, tot = 0;
    for (let e = 0; e < episodes; e++) {
      let [cs, ch] = env.reset();
      for (let t = 0; t < MAX_STEPS; t++) {
        const r = env.step(greedy(this.Qs, cs, ch), randint(rng, NA));
        [cs, ch] = env.state(); seen += r.see ? 1 : 0; tot++;
        if (r.done) break;
      }
    }
    const seeker = seen / Math.max(1, tot);
    let seen2 = 0, tot2 = 0;
    for (let e = 0; e < episodes; e++) {
      let [cs, ch] = env.reset();
      for (let t = 0; t < MAX_STEPS; t++) {
        const r = env.step(randint(rng, NA), greedy(this.Qh, cs, ch));
        [cs, ch] = env.state(); seen2 += r.see ? 1 : 0; tot2++;
        if (r.done) break;
      }
    }
    return { seeker, hider: 1 - seen2 / Math.max(1, tot2) };
  }
  /** Greedy actions for the live demo game (current policy). */
  act(cs, ch) { return [greedy(this.Qs, cs, ch), greedy(this.Qh, cs, ch)]; }
}

// ===========================================================================
// Live UI (browser only)
// ===========================================================================
const COL = {
  bg: "#0c111b", grid: "#1b2433", wall: "#33405a",
  hider: "#43b6ff", seeker: "#ff6b6b", see: "#ffd24a",
  text: "#cdd8e6", muted: "#8a98ad", hiderLine: "#43b6ff", seekerLine: "#ff6b6b",
};

export function mountTrain(root) {
  root.innerHTML = `
    <div class="train-wrap">
      <h2>Train it live — nothing is scripted</h2>
      <p class="train-intro">Two AIs learn hide-and-seek from scratch by playing each other.
        Hit <b>Train</b> and watch them go from random to skilled while the curve climbs.
        This runs the real tabular Q-learning in your browser (no GPU, no server).</p>
      <div class="train-controls">
        <button id="tr-play" class="btn primary">▶ Train</button>
        <button id="tr-reset" class="btn">↻ Reset</button>
        <label class="tr-speed">Speed
          <input id="tr-speed" type="range" min="20" max="600" value="160" step="20">
        </label>
        <span id="tr-ep" class="tr-stat">episode 0</span>
      </div>
      <div class="train-grid">
        <div class="train-card">
          <div class="tc-title">Live match (current policy)</div>
          <canvas id="tr-game" width="380" height="380"></canvas>
          <div id="tr-status" class="tr-status">—</div>
        </div>
        <div class="train-card">
          <div class="tc-title">Skill while training (vs a random opponent)</div>
          <canvas id="tr-chart" width="560" height="360"></canvas>
          <div class="tr-legend">
            <span><i style="background:${COL.seekerLine}"></i> Seeker — sight-rate</span>
            <span><i style="background:${COL.hiderLine}"></i> Hider — evasion</span>
          </div>
        </div>
      </div>
    </div>`;

  const trainer = new Trainer(1, 40000);
  const gcv = root.querySelector("#tr-game"), gctx = gcv.getContext("2d");
  const ccv = root.querySelector("#tr-chart"), cctx = ccv.getContext("2d");
  const btn = root.querySelector("#tr-play"), epEl = root.querySelector("#tr-ep");
  const statusEl = root.querySelector("#tr-status"), speedEl = root.querySelector("#tr-speed");

  let running = false, raf = 0, frame = 0;
  const demo = new GridEnv(makeRng(999));
  demo.reset();

  btn.onclick = () => { running = !running; btn.textContent = running ? "⏸ Pause" : "▶ Train"; btn.classList.toggle("primary", !running); };
  root.querySelector("#tr-reset").onclick = () => { trainer.reset(1); demo.reset(); running = false; btn.textContent = "▶ Train"; btn.classList.add("primary"); };

  function stepDemo() {
    const [cs, ch] = demo.state();
    const [as, ah] = trainer.episode > 0 ? trainer.act(cs, ch)
                                         : [randint(demo.rng, NA), randint(demo.rng, NA)];
    const r = demo.step(as, ah);
    demo._see = r.see;
    if (r.done) { setTimeout(() => {}, 0); demo.reset(); }
    return r;
  }

  function loop() {
    raf = requestAnimationFrame(loop);
    if (running) {
      const k = +speedEl.value;
      trainer.trainMany(k);
      if (trainer.curve.length === 1 || trainer.episode - trainer.curve[trainer.curve.length - 1].e >= 250) {
        const sk = trainer.evalSkill(120);
        trainer.curve.push({ e: trainer.episode, s: sk.seeker, h: sk.hider });
      }
      if (trainer.episode >= trainer.total) { running = false; btn.textContent = "✓ Done"; btn.classList.remove("primary"); }
    }
    frame++;
    if (frame % 5 === 0) stepDemo();
    drawGame(gctx, demo);
    drawChart(cctx, trainer.curve, trainer.total);
    epEl.textContent = "episode " + trainer.episode.toLocaleString();
    const last = trainer.curve[trainer.curve.length - 1];
    statusEl.innerHTML = `seeker sight-rate <b style="color:${COL.seeker}">${(last.s * 100) | 0}%</b>` +
      ` &nbsp; hider evasion <b style="color:${COL.hider}">${(last.h * 100) | 0}%</b>`;
  }
  loop();
  return { stop() { cancelAnimationFrame(raf); }, pause() { running = false; btn.textContent = "▶ Train"; } };
}

function drawGame(ctx, env) {
  const W = ctx.canvas.width, cell = W / N;
  ctx.fillStyle = COL.bg; ctx.fillRect(0, 0, W, W);
  ctx.strokeStyle = COL.grid; ctx.lineWidth = 1;
  for (let i = 0; i <= N; i++) {
    ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, W); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(W, i * cell); ctx.stroke();
  }
  ctx.fillStyle = COL.wall;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (WALL[y * N + x]) {
    ctx.fillRect(x * cell + 1, (N - 1 - y) * cell + 1, cell - 2, cell - 2);
  }
  const cx = (x) => (x + 0.5) * cell, cy = (y) => (N - 1 - y + 0.5) * cell;
  if (env._see) {
    ctx.strokeStyle = COL.see; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(cx(env.sx), cy(env.sy)); ctx.lineTo(cx(env.hx), cy(env.hy)); ctx.stroke();
    ctx.setLineDash([]);
  }
  const dot = (x, y, col) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(cx(x), cy(y), cell * 0.32, 0, 7); ctx.fill(); };
  dot(env.hx, env.hy, COL.hider);
  dot(env.sx, env.sy, COL.seeker);
}

function drawChart(ctx, curve, total) {
  const W = ctx.canvas.width, H = ctx.canvas.height, padL = 38, padB = 26, padT = 10, padR = 10;
  ctx.clearRect(0, 0, W, H);
  const x0 = padL, x1 = W - padR, y0 = H - padB, y1 = padT;
  ctx.strokeStyle = "rgba(140,160,190,0.18)"; ctx.fillStyle = COL.muted;
  ctx.font = "11px ui-monospace, monospace"; ctx.lineWidth = 1;
  for (let p = 0; p <= 4; p++) {
    const yy = y0 + (y1 - y0) * (p / 4);
    ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x1, yy); ctx.stroke();
    ctx.fillText((p * 25) + "%", 6, yy + 4);
  }
  const px = (e) => x0 + (x1 - x0) * Math.min(1, e / total);
  const py = (v) => y0 + (y1 - y0) * Math.max(0, Math.min(1, v));
  const line = (key, col) => {
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
    curve.forEach((c, i) => { const X = px(c.e), Y = py(c[key]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
    ctx.stroke();
    const last = curve[curve.length - 1];
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(px(last.e), py(last[key]), 3, 0, 7); ctx.fill();
  };
  line("h", COL.hiderLine);
  line("s", COL.seekerLine);
  ctx.fillStyle = COL.muted;
  ctx.fillText("0", x0 - 2, y0 + 16);
  ctx.fillText((total / 1000) + "k episodes", x1 - 78, y0 + 16);
}
