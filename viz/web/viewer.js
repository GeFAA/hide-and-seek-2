/**
 * viewer.js -- The bright, clean OpenAI "Emergent Tool Use" hide-and-seek 3D
 * viewer for Hide & Seek 2.0.
 *
 * A single cohesive ES module that:
 *   1. Boots a polished light studio Three.js scene (ACES tone mapping, soft
 *      shadows, gentle fog, a white tiled arena, a chunky white wall, a dense
 *      "foam city" backdrop of light cubes, hemisphere + warm key light).
 *   2. Loads an hns2-traj trajectory (default ./trajectories/demo_trajectory.json,
 *      plus a file picker and drag-and-drop), validating it via trajectory.js.
 *   3. Builds one reusable mesh per active entity (created once, transforms
 *      updated per frame) and animates them through the episode with smooth
 *      linear interpolation between integer frames.
 *   4. Renders the 2.0 mechanics visually: heading arrows, held-object link
 *      lines, "spotted" highlights, decoy ring pulses, fog discs, vision cones,
 *      trails, elevation (z), and a god-view that reveals true decoys.
 *   5. Wires a full glass HUD: top status bar, bottom transport (play/scrub/
 *      speed/loop), a collapsible left legend + toggles panel, and a
 *      hover/click inspector.
 *
 * Tech: Three.js r160 via importmap (no build step). Procedural geometry and
 * colors only -- no external assets beyond the Three.js CDN.
 *
 * The code is organized into clearly commented sections:
 *   [A] Imports & guards            [F] Entity mesh factory & registry
 *   [B] Constants & palette         [G] Per-frame update / animation loop
 *   [C] Renderer / scene / camera   [H] HUD construction & wiring
 *   [D] Lights / ground / arena     [I] Loading (fetch / file / drop)
 *   [E] Fog / overlays              [J] Boot
 */

// ============================================================================
// [A] Imports & boot guards
// ============================================================================

// Scene colors route through the theme system (PAL) below, not trajectory.js's
// static COLORS -- but we keep importing Trajectory (the validated parser).
import { Trajectory } from "./trajectory.js";

// The Learning dashboard (tables + canvas charts). Local ES module, no CDN, so
// a static import is safe and keeps the Learning tab instant on first open.
import { renderLearning } from "./learning.js";

// We import Three.js dynamically-ish: a static import would throw at module
// parse time if the CDN is down, and the browser would swallow it silently.
// Importing it here at top-level is fine for r160; the index.html watchdog +
// the try/catch in boot() surface any failure with a friendly message.
let THREE, OrbitControls, RoundedBoxGeometry = null;
try {
  THREE = await import("three");
  ({ OrbitControls } = await import("three/addons/controls/OrbitControls.js"));
  // RoundedBoxGeometry gives the boxes their soft, slightly-chamfered edges
  // (the OpenAI hide-and-seek look). It's an addon; if it fails to load we
  // fall back to a plain BoxGeometry so the viewer still works.
  try {
    ({ RoundedBoxGeometry } = await import("three/addons/geometries/RoundedBoxGeometry.js"));
  } catch (_) {
    RoundedBoxGeometry = null;
  }
} catch (err) {
  showBootError(
    "Could not load the Three.js engine from the CDN (unpkg.com). " +
    "Check your network connection, or serve this folder locally."
  );
  throw err;
}

/** Reveal the full-screen friendly fallback with a specific message. */
function showBootError(message) {
  const be = document.getElementById("boot-error");
  const msg = document.getElementById("boot-error-msg");
  if (msg && message) msg.textContent = message;
  if (be) be.hidden = false;
}

// ============================================================================
// [B] Theme system (dark default + light) -- scene colors AND DOM CSS vars
// ============================================================================
//
// Everything visual is driven from ONE active palette object, `PAL`. Each theme
// supplies the full set of colors for BOTH the Three.js scene (background
// gradient, fog, ground, grid, walls, foam-city, fog discs, entity colors) and
// the DOM (exposed as CSS custom properties so style.css can restyle instantly
// under [data-theme="dark"] / [data-theme="light"]).
//
// LIGHT == the original look (kept byte-for-byte from the old COLORS + CSS).
// DARK  == the moody-but-clean default.
//
// Scene color keys are stored as 0xRRGGBB numbers (drop straight into
// THREE.Color); the few DOM-only colors (panel rgba, blur, etc.) are strings,
// applied as CSS variables in applyTheme().

const THEMES = {
  dark: {
    // ---- scene (numbers -> THREE.Color) ----
    bgTop:     0x0b0f17,   // background gradient (top)
    bgBottom:  0x0e1320,   // background gradient (bottom)
    fog:       0x0e1320,   // scene fog color (far cubes fade into this)
    ground:    0x141a26,   // floor
    grid:      0x232c3d,   // tile lines
    outline:   0x2b3650,   // arena boundary square
    wall:      0x2b3446,   // chunky walls
    foamLo:    0x1a2130,   // foam-city jitter low
    foamHi:    0x2b3547,   // foam-city jitter high
    fogDisc:   0x96aad2,   // fog-of-war disc tint (rgba alpha applied in code)
    fogDiscAlpha: 0.08,
    // entities (brighter so they read as glowing on the dark floor)
    hider:     0x43b6ff,
    seeker:    0xff6b6b,
    box_light: 0xf5b73e,
    box_heavy: 0xef9a2b,
    ramp:      0xa78bfa,
    decoy:     0xc77dff,
    door:      0x56c7e6,
    spotted:   0xffd24a,
    muted:     0x8a98ad,
    edgeHeavy: 0xd9962f,   // heavy-box outline
    linkLine:  0xef9a2b,   // held-object link line
    // agents glow a touch more in dark
    agentEmissive: 0.26,
    decoyEmissive: 1.0,
    // tone-mapping exposure for this theme
    exposure: 1.0,
    // lights (moody-but-clean: lower ambient/hemi, keep a soft key)
    ambient: 0.32,
    hemiSky:  0xb9c8e6,
    hemiGround: 0x0c1019,
    hemi: 0.55,
    key:      0xdCE6FF,
    keyInt:   0.85,
    fill:     0x223049,
    fillInt:  0.30,
    groundRough: 0.92,
    // ---- DOM (strings -> CSS variables) ----
    dom: {
      "--bg": "#0b0f17",
      "--panel": "rgba(18, 24, 36, 0.74)",
      "--panel-solid": "rgba(16, 21, 32, 0.96)",
      "--panel-border": "rgba(120, 150, 200, 0.14)",
      "--panel-hover": "rgba(28, 36, 52, 0.86)",
      "--text": "#dbe5f3",
      "--muted": "#8a98ad",
      "--accent": "#5b9dff",
      "--accent-strong": "#4a86f0",
      "--hider": "#43b6ff",
      "--seeker": "#ff6b6b",
      "--box_light": "#f5b73e",
      "--box_heavy": "#ef9a2b",
      "--ramp": "#a78bfa",
      "--decoy": "#c77dff",
      "--wall": "#2b3446",
      "--door": "#56c7e6",
      "--spotted": "#ffd24a",
      "--amber": "#f5b73e",
      "--red": "#ff6b6b",
      "--chip-line": "rgba(120,150,200,0.20)",
      "--zebra": "rgba(255,255,255,0.025)",
      "--shadow": "0 14px 38px rgba(0, 0, 0, 0.45)",
      "--scrim": "rgba(8, 11, 18, 0.66)",
      "--cap-text": "#e7eefb",
      "--cap-halo": "rgba(8, 12, 20, 0.85)",
      "--track": "rgba(255,255,255,0.08)",
    },
  },

  light: {
    // ---- scene ---- (the ORIGINAL bright studio values)
    bgTop:     0xdce4ec,
    bgBottom:  0xeef2f6,
    fog:       0xe7edf3,
    ground:    0xf3f5f8,
    grid:      0xd6dde4,
    outline:   0xbac4cf,
    wall:      0xedf0f4,
    foamLo:    0xeef1f5,   // original foam used a single ~0xeef1f5 tone
    foamHi:    0xeef1f5,
    fogDisc:   0xcfd8e2,
    fogDiscAlpha: 0.10,
    hider:     0x2f9be8,
    seeker:    0xf2604d,
    box_light: 0xf2b441,
    box_heavy: 0xe89a2b,
    ramp:      0xb9a98f,
    decoy:     0x8b5cf6,
    door:      0xa9c6e2,
    spotted:   0xf59e2e,
    muted:     0x6c7785,
    edgeHeavy: 0xc98322,
    linkLine:  0xe89a2b,
    agentEmissive: 0.12,
    decoyEmissive: 0.9,
    exposure: 1.12,
    ambient: 0.45,
    hemiSky:  0xffffff,
    hemiGround: 0xdfe6ee,
    hemi: 1.1,
    key:      0xfff4e6,
    keyInt:   1.0,
    fill:     0xeaf1f8,
    fillInt:  0.35,
    groundRough: 0.95,
    dom: {
      "--bg": "#e7edf3",
      "--panel": "rgba(255, 255, 255, 0.72)",
      "--panel-solid": "rgba(255, 255, 255, 0.94)",
      "--panel-border": "rgba(15, 30, 55, 0.08)",
      "--panel-hover": "rgba(255, 255, 255, 0.92)",
      "--text": "#1f2733",
      "--muted": "#6c7785",
      "--accent": "#2f6bff",
      "--accent-strong": "#2a60e6",
      "--hider": "#2f9be8",
      "--seeker": "#f2604d",
      "--box_light": "#f2b441",
      "--box_heavy": "#e89a2b",
      "--ramp": "#b9a98f",
      "--decoy": "#8b5cf6",
      "--wall": "#9aa6b4",
      "--door": "#6fa0cf",
      "--spotted": "#f59e2e",
      "--amber": "#f2b441",
      "--red": "#f2604d",
      "--chip-line": "rgba(15,30,55,0.10)",
      "--zebra": "rgba(15,30,55,0.025)",
      "--shadow": "0 10px 30px rgba(31, 45, 70, 0.12)",
      "--scrim": "rgba(231, 237, 243, 0.7)",
      "--cap-text": "#2a3340",
      "--cap-halo": "rgba(255, 255, 255, 0.85)",
      "--track": "rgba(15,30,55,0.10)",
    },
  },
};

/** Persisted-theme storage key. */
const THEME_KEY = "hns2-theme";

/** The live "active palette": a flat copy of the current theme's scene colors. */
const PAL = {};

/** Current theme name. */
let currentTheme = "dark";

/** Read the persisted theme (default dark). */
function readStoredTheme() {
  try {
    const v = localStorage.getItem(THEME_KEY);
    if (v === "light" || v === "dark") return v;
  } catch (_) { /* private mode etc. */ }
  return "dark";
}

/** Copy a theme's scene colors into PAL (so PAL.x is always the active color). */
function loadPalette(name) {
  const t = THEMES[name] || THEMES.dark;
  for (const k in t) {
    if (k === "dom") continue;
    PAL[k] = t[k];
  }
}

/**
 * Live theme-color lookup handed to learning.js so its charts pull the current
 * team + chart colors. Returns a flat {key:number} map.
 */
function themeColorsForCharts() {
  return {
    hider: PAL.hider,
    seeker: PAL.seeker,
    chartGrid: PAL.grid,
    chartAxis: PAL.muted,
    chartText: PAL.muted,
  };
}

// `COLORS` is imported from trajectory.js (the schema mirror) but the viewer no
// longer reads scene colors from it -- everything routes through PAL so themes
// can swap live. We seed PAL immediately so module-level scene construction
// below sees real values.
loadPalette(readStoredTheme());
currentTheme = readStoredTheme();

// ============================================================================
// [B2] Constants & palette helpers
// ============================================================================

/** Default trajectory fetched on first load. */
const DEFAULT_TRAJ_URL = "./trajectories/demo_trajectory.json";

/** Vision-cone half-angle convenience (config.vision_cone_deg default 135deg). */
const VISION_CONE_DEG = 135;
const VISION_CONE_RANGE = 6.0; // world units the cone extends in front of an agent

/** Convert a 0xRRGGBB hex into a THREE.Color (cached). */
const _colorCache = new Map();
function col(hex) {
  if (!_colorCache.has(hex)) _colorCache.set(hex, new THREE.Color(hex));
  return _colorCache.get(hex);
}

/** Playback speed options offered in the transport bar. */
const SPEEDS = [0.5, 1, 2, 4];

/**
 * Tiny deterministic PRNG (mulberry32). Used ONLY for the static decorative
 * "foam city" backdrop so the cube field is stable across reloads.
 * @param {number} seed
 * @returns {() => number} a function returning floats in [0, 1)
 */
function makePRNG(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ============================================================================
// [C] Renderer, scene, camera, controls
// ============================================================================

const host = document.getElementById("canvas-host");

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;        // soft shadows
renderer.toneMapping = THREE.ACESFilmicToneMapping;       // gentle, clean tone curve
renderer.toneMappingExposure = PAL.exposure;              // per-theme exposure (see applyTheme)
renderer.outputColorSpace = THREE.SRGBColorSpace;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();

/** "#rrggbb" string for a 0xRRGGBB number. */
function hexStr(n) {
  return "#" + (n >>> 0).toString(16).padStart(6, "0");
}

/**
 * Build a soft vertical-gradient sky texture for the scene background using the
 * ACTIVE theme's bgTop -> bgBottom colors. Drawn into a CanvasTexture; rebuilt
 * on theme change.
 */
function makeBackgroundTexture() {
  const c = document.createElement("canvas");
  c.width = 16;
  c.height = 256;
  const ctx = c.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, 0, c.height);
  g.addColorStop(0, hexStr(PAL.bgTop));
  g.addColorStop(1, hexStr(PAL.bgBottom));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, c.width, c.height);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}
scene.background = makeBackgroundTexture();
// Gentle fog so the far "foam city" cubes fade softly into the backdrop.
scene.fog = new THREE.Fog(PAL.fog, 34, 120);

const camera = new THREE.PerspectiveCamera(
  42, window.innerWidth / window.innerHeight, 0.1, 600
);
// Closer, slightly lower 3/4 angle so the ARENA fills more of the frame and the
// white foam city forms a clean surrounding border rather than dominating it.
camera.position.set(10.5, 7.8, 13);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 4;
controls.maxDistance = 90;
controls.maxPolarAngle = Math.PI * 0.495; // keep camera above the ground
controls.target.set(0, 0.9, 0);           // raise target so the arena centers

// ============================================================================
// [D] Lights, ground, arena grid
// ============================================================================

// Bright, soft, low-contrast studio lighting (the OpenAI hide-and-seek look).
// The target is a BRIGHT, low-contrast, soft scene where nothing reads as
// black: heavy ambient + hemisphere fill, a gentler key light for low contrast.

// Flat ambient fill so shadowed vertical faces never crush toward black.
// (Intensity + colors are (re)applied per theme in applyTheme().)
const ambient = new THREE.AmbientLight(0xffffff, PAL.ambient);
scene.add(ambient);

// Hemisphere bounce: sky + ground term so downward-facing / away faces stay
// readable. Does most of the soft fill; dimmed in dark mode.
const hemi = new THREE.HemisphereLight(PAL.hemiSky, PAL.hemiGround, PAL.hemi);
scene.add(hemi);

// Key directional light, high and angled, casting SOFT shadows. Kept gentle so
// the lit/unlit contrast stays low; a soft warm key in light, cool key in dark.
const keyLight = new THREE.DirectionalLight(PAL.key, PAL.keyInt);
keyLight.position.set(16, 26, 12);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 1;
keyLight.shadow.camera.far = 90;
keyLight.shadow.camera.left = -22;
keyLight.shadow.camera.right = 22;
keyLight.shadow.camera.top = 22;
keyLight.shadow.camera.bottom = -22;
keyLight.shadow.bias = -0.0004;
keyLight.shadow.normalBias = 0.02;
keyLight.shadow.radius = 5;
scene.add(keyLight);

// A dim, shadowless fill from the opposite side to keep shadowed faces clean
// and low-contrast (no harsh black).
const fillLight = new THREE.DirectionalLight(PAL.fill, PAL.fillInt);
fillLight.position.set(-14, 9, -16);
scene.add(fillLight);

// Ground plane (large, bright). Lies in the XZ plane; world y is "up".
// IMPORTANT mapping: the trajectory's (x, y) are floor coordinates and z is
// elevation. We map traj.x -> three.x, traj.y -> three.z, traj.z -> three.y.
const groundMat = new THREE.MeshStandardMaterial({
  color: col(PAL.ground),
  roughness: PAL.groundRough,
  metalness: 0.0,
});
const groundGeo = new THREE.PlaneGeometry(600, 600);
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = 0;
ground.receiveShadow = true;
scene.add(ground);

// Grid + arena outline are rebuilt whenever a trajectory with a new arena loads.
let gridGroup = new THREE.Group();
scene.add(gridGroup);

// The static decorative "foam city" backdrop lives here (built once per arena).
let backdropGroup = new THREE.Group();
scene.add(backdropGroup);

/**
 * (Re)build the subtle tile grid and a faint arena boundary for a given arena
 * half-extent (bound). Understated light-grey tile lines, as in the reference.
 */
function buildArena(bound) {
  // Dispose the previous grid + outline (GridHelper buffers, line materials)
  // before rebuilding -- buildArena runs on every trajectory load AND every
  // theme toggle, so a raw Group.clear() here would steadily leak GPU buffers.
  clearGroup(gridGroup);

  // Fine 1-unit tile grid spanning the arena, in the theme tile-line color.
  const span = Math.ceil(bound) * 2;
  const grid = new THREE.GridHelper(span, span, col(PAL.grid), col(PAL.grid));
  grid.position.y = 0.002;
  grid.material.opacity = currentTheme === "dark" ? 0.5 : 0.6;
  grid.material.transparent = true;
  gridGroup.add(grid);

  // Faint arena boundary square (the actual playable bounds) -- understated.
  const b = bound;
  const pts = [
    new THREE.Vector3(-b, 0.01, -b),
    new THREE.Vector3(b, 0.01, -b),
    new THREE.Vector3(b, 0.01, b),
    new THREE.Vector3(-b, 0.01, b),
    new THREE.Vector3(-b, 0.01, -b),
  ];
  const outlineGeo = new THREE.BufferGeometry().setFromPoints(pts);
  const outline = new THREE.Line(
    outlineGeo,
    new THREE.LineBasicMaterial({
      color: col(PAL.outline),
      transparent: true,
      opacity: currentTheme === "dark" ? 0.7 : 0.5,
    })
  );
  gridGroup.add(outline);

  // (Re)build the foam-city backdrop sized to this arena.
  buildFoamCity(bound);
}

/**
 * Build the signature "foam city" backdrop: a dense, deterministic field of
 * randomly-sized light cubes scattered OUTSIDE the arena bounds, receding into
 * the soft background. One InstancedMesh for performance; static + decorative.
 *
 * @param {number} bound - arena half-extent; cubes are kept clear of it.
 */
function buildFoamCity(bound) {
  clearGroup(backdropGroup);

  const rng = makePRNG(0x9e3779b1); // fixed seed -> stable across reloads
  const COUNT = 560;
  const innerClear = bound + 1.6;   // keep cubes out of (and just off) the arena
  const outerRadius = bound + 46;   // pack out to a large radius

  const geo = new THREE.BoxGeometry(1, 1, 1);
  // Base material is a neutral white that gets MULTIPLIED by an explicit
  // per-instance color. CRITICAL: every instance is assigned a real color
  // below, so there is never an unset/black instanceColor (the old "black
  // skyline" bug). In LIGHT theme foamLo == foamHi, so the field is uniform
  // soft-white exactly like the original; in DARK theme each cube is tinted
  // between foamLo..foamHi for a subtle, moody city.
  const mat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.95,
    metalness: 0.0,
  });
  const mesh = new THREE.InstancedMesh(geo, mat, COUNT);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);

  const cLo = new THREE.Color(PAL.foamLo);
  const cHi = new THREE.Color(PAL.foamHi);
  const cTmp = new THREE.Color();

  const dummy = new THREE.Object3D();
  let placed = 0;
  let guard = 0;
  while (placed < COUNT && guard < COUNT * 20) {
    guard++;
    // Random point in a square region, rejected if it falls inside the arena.
    const x = (rng() * 2 - 1) * outerRadius;
    const z = (rng() * 2 - 1) * outerRadius;
    if (Math.abs(x) < innerClear && Math.abs(z) < innerClear) continue;
    // Density falloff: fewer cubes far out, so the field reads dense up close.
    const dist = Math.hypot(x, z);
    if (dist > outerRadius) continue;
    if (rng() > 1.0 - (outerRadius - dist) / outerRadius * 0.85) continue;

    const fw = 0.6 + rng() * 1.4;       // footprint 0.6..2.0
    const fd = 0.6 + rng() * 1.4;
    const h = 0.4 + rng() * 4.1;        // height 0.4..4.5
    dummy.position.set(x, h / 2, z);
    dummy.rotation.set(0, 0, 0);        // axis-aligned
    dummy.scale.set(fw, h, fd);
    dummy.updateMatrix();
    mesh.setMatrixAt(placed, dummy.matrix);
    // Per-instance tint: lerp foamLo..foamHi by a stable rng draw (this also
    // preserves the original deterministic placement sequence, which consumed
    // one rng() draw at this point).
    cTmp.copy(cLo).lerp(cHi, rng());
    mesh.setColorAt(placed, cTmp);
    placed++;
  }
  mesh.count = placed;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  backdropGroup.add(mesh);
}

// ============================================================================
// [D2] Theme application -- swap PAL, restyle scene + DOM, rebuild colored bits
// ============================================================================

/**
 * Apply a theme by name ("dark" | "light"):
 *   1. copy its scene colors into PAL,
 *   2. set document.documentElement.dataset.theme + all CSS variables so
 *      style.css restyles the DOM instantly,
 *   3. update the live Three.js scene: exposure, background gradient, fog,
 *      ground material, lights,
 *   4. rebuild the arena (grid/outline/foam-city) and entities so every
 *      scene color reflects the new palette.
 *
 * @param {string} name
 * @param {{persist?:boolean, rebuild?:boolean}} [opts]
 */
function applyTheme(name, opts = {}) {
  const theme = THEMES[name] ? name : "dark";
  currentTheme = theme;
  loadPalette(theme);

  // (2) DOM: dataset + CSS variables.
  const root = document.documentElement;
  root.dataset.theme = theme;
  const domVars = THEMES[theme].dom || {};
  for (const k in domVars) root.style.setProperty(k, domVars[k]);

  if (opts.persist !== false) {
    try { localStorage.setItem(THEME_KEY, theme); } catch (_) { /* ignore */ }
  }

  // (3) Live scene properties.
  renderer.toneMappingExposure = PAL.exposure;

  // Background gradient (dispose the old CanvasTexture to avoid a GPU leak).
  const oldBg = scene.background;
  scene.background = makeBackgroundTexture();
  if (oldBg && oldBg.isTexture) oldBg.dispose();

  // Fog + ground.
  if (scene.fog) scene.fog.color.set(PAL.fog);
  groundMat.color.set(PAL.ground);
  groundMat.roughness = PAL.groundRough;
  groundMat.needsUpdate = true;

  // Lights.
  ambient.intensity = PAL.ambient;
  hemi.color.set(PAL.hemiSky);
  hemi.groundColor.set(PAL.hemiGround);
  hemi.intensity = PAL.hemi;
  keyLight.color.set(PAL.key);
  keyLight.intensity = PAL.keyInt;
  fillLight.color.set(PAL.fill);
  fillLight.intensity = PAL.fillInt;

  // (4) Rebuild colored scene content (grid/outline/foam + entities). The arena
  // bound is taken from the loaded trajectory if present, else the boot default.
  if (opts.rebuild !== false) {
    const bound = state.traj ? state.traj.bound : 6;
    buildArena(bound);
    gridGroup.visible = state.showGrid;
    if (state.traj) {
      buildEntities(state.traj);
      // Re-pop entities in so the rebuild reads as intentional, not a flicker.
      popInEntities();
    }
  }

  // Refresh DOM bits the viewer paints inline (legend swatches use PAL).
  if (ui.legend) buildLegend();

  // Update the toggle button glyph/label + notify the learning dashboard.
  syncThemeButton();
  if (learning && learning.setTheme) learning.setTheme(theme);
}

// ============================================================================
// [E] Shared geometry / material caches (reused across all entities)
// ============================================================================

/**
 * Unit rounded-cube geometry for the boxes (soft, slightly-chamfered edges like
 * the reference). Uses the RoundedBoxGeometry addon if it loaded; otherwise a
 * plain BoxGeometry so the viewer still works without the addon.
 */
function makeUnitBoxGeo() {
  if (RoundedBoxGeometry) {
    try {
      return new RoundedBoxGeometry(1, 1, 1, 4, 0.12);
    } catch (_) {
      /* fall through to plain box */
    }
  }
  return new THREE.BoxGeometry(1, 1, 1);
}

// Build geometries once; entity meshes share them and only vary scale/material.
const GEO = {
  capsule: new THREE.CapsuleGeometry(0.32, 0.5, 8, 18),
  cube: new THREE.BoxGeometry(1, 1, 1),         // walls / doors (hard slabs)
  roundedCube: makeUnitBoxGeo(),                // boxes (soft edges)
  aura: new THREE.SphereGeometry(0.5, 20, 16),  // decoy glow shell
  auraDisc: new THREE.PlaneGeometry(1, 1),      // flat ground glow pool (agents)
  cylinder: new THREE.CylinderGeometry(0.32, 0.32, 0.9, 18),
  nose: new THREE.ConeGeometry(0.085, 0.22, 12),
  eye: new THREE.SphereGeometry(0.06, 10, 8),    // agent face dots
  ringMarker: new THREE.RingGeometry(0.5, 0.62, 32),
  decoyCore: new THREE.IcosahedronGeometry(0.22, 1),
  spottedRing: new THREE.RingGeometry(0.62, 0.78, 40),
};

/**
 * The set of shared, cache-owned geometries. These are reused by every entity
 * viz, so they must survive entity rebuilds -- disposeEntityGroup() skips them.
 */
const SHARED_GEOS = new Set(Object.values(GEO));

/**
 * Dispose one entity viz group: free every child's MATERIAL (always per-entity)
 * and any geometry that is NOT in the shared GEO cache (e.g. vision cones,
 * ramp wedges, heavy-box edges, decoy spheres/rings -- all built fresh per
 * entity). Shared geometries (capsule/cube/...) are left intact. Then detach
 * the children. Recurses for safety though entity groups are shallow.
 */
function disposeEntityGroup(group) {
  for (const c of [...group.children]) {
    if (c.children && c.children.length) disposeEntityGroup(c);
    if (c.geometry && !SHARED_GEOS.has(c.geometry)) c.geometry.dispose();
    if (c.material) {
      if (Array.isArray(c.material)) c.material.forEach((m) => m.dispose());
      else c.material.dispose();
    }
  }
  group.clear();
}

/**
 * Generate a soft CanvasTexture "lock badge" emblem applied to box side faces:
 * a rounded white square with a simple grey keyhole/lock glyph. Built once and
 * shared by all boxes. Returns a THREE.CanvasTexture.
 */
let _lockTex = {};
function lockEmblemTexture(color) {
  const key = color.getHexString();           // sRGB hex of the box's own colour
  if (_lockTex[key]) return _lockTex[key];
  const S = 256;
  const c = document.createElement("canvas");
  c.width = S; c.height = S;
  const ctx = c.getContext("2d");
  // OPAQUE base fill in the box's colour -- this IS the whole face (no black).
  ctx.fillStyle = "#" + key;
  ctx.fillRect(0, 0, S, S);
  // Stamped emblem: a soft inset panel + a muted lock glyph.
  const pad = 72, x = pad, y = pad, w = S - pad * 2, h = S - pad * 2;
  roundRect(ctx, x, y, w, h, 28);
  ctx.fillStyle = "rgba(255,255,255,0.15)"; ctx.fill();
  ctx.lineWidth = 5; ctx.strokeStyle = "rgba(60,46,16,0.30)"; ctx.stroke();
  const cx = S / 2;
  ctx.strokeStyle = "rgba(60,46,16,0.55)"; ctx.fillStyle = "rgba(60,46,16,0.55)";
  ctx.lineWidth = 13; ctx.lineCap = "round";
  ctx.beginPath(); ctx.arc(cx, 122, 24, Math.PI, 0, false); ctx.stroke();  // shackle
  roundRect(ctx, cx - 38, 122, 76, 56, 11); ctx.fill();                    // body
  ctx.fillStyle = "rgba(255,255,255,0.85)";                                // keyhole
  ctx.beginPath(); ctx.arc(cx, 150, 8, 0, Math.PI * 2); ctx.fill();
  ctx.fillRect(cx - 3.5, 150, 7, 19);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4; tex.needsUpdate = true;
  _lockTex[key] = tex;
  return tex;
}

/** Trace a rounded-rectangle path (caller fills/strokes). */
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Soft white radial-gradient sprite, shared by the agent ground-glow pools. */
let _glowTex = null;
function softGlowTexture() {
  if (_glowTex) return _glowTex;
  const S = 128;
  const c = document.createElement("canvas");
  c.width = S; c.height = S;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
  g.addColorStop(0.0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,0.45)");
  g.addColorStop(1.0, "rgba(255,255,255,0)");
  ctx.fillStyle = g; ctx.fillRect(0, 0, S, S);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
  _glowTex = tex;
  return tex;
}

// ============================================================================
// [F] Entity mesh factory & registry
// ============================================================================

/**
 * One visual record per entity slot. We create the mesh group once and update
 * its transform / material each frame for performance.
 * @typedef {{
 *   group: THREE.Group, body: THREE.Mesh, nose?: THREE.Mesh,
 *   cone?: THREE.Mesh, spotRing?: THREE.Mesh, marker?: THREE.Mesh,
 *   decoyRing?: THREE.Mesh, type: string, team: number, id: number,
 *   baseEmissive: number, isAgent: boolean
 * }} EntityViz
 */

/** Active state of the whole viewer (trajectory + meshes + UI flags). */
const state = {
  traj: /** @type {Trajectory|null} */ (null),
  vizById: /** @type {Map<number, EntityViz>} */ (new Map()),
  entityRoot: new THREE.Group(),
  linkRoot: new THREE.Group(),     // held-object link lines
  fogRoot: new THREE.Group(),      // fog discs
  trailRoot: new THREE.Group(),    // agent trails

  // playback
  pos: 0,            // fractional frame position
  playing: false,
  speed: 1,
  loop: true,
  lastTime: 0,

  // toggles (mirrors the left-panel checkboxes)
  showCones: false,
  showFog: true,
  godView: false,
  showGrid: true,
  showTrails: false,
  followCam: false,

  // interaction
  followTargetId: -1,
  hoverId: -1,
  pinnedId: -1,      // clicked entity stays in the inspector
  trails: /** @type {Map<number, THREE.Vector3[]>} */ (new Map()),

  // UI / animation
  tab: "watch",      // "watch" | "learning" | "about"
  popInStart: 0,     // performance.now() when the current entity pop-in began
  popInDur: 520,     // ms for the entity scale pop-in
  introActive: false,// camera establishing-shot in progress
};
scene.add(state.entityRoot);
scene.add(state.linkRoot);
scene.add(state.fogRoot);
scene.add(state.trailRoot);

// Respect the user's reduced-motion preference: skip the camera intro + big
// transitions and snap to final states instead.
const reducedMotionMQ = window.matchMedia
  ? window.matchMedia("(prefers-reduced-motion: reduce)")
  : { matches: false, addEventListener() {} };
let reducedMotion = reducedMotionMQ.matches;
if (reducedMotionMQ.addEventListener) {
  reducedMotionMQ.addEventListener("change", (e) => { reducedMotion = e.matches; });
}

/** The lazily-created Learning dashboard handle (see learning.js). */
let learning = null;

/** easeOutCubic for the camera intro + entity pop-in. */
function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

/**
 * Trigger the entity "pop-in": all active entity groups scale up from ~0 with a
 * quick eased overshoot when a scenario loads (or the scene is rebuilt). Driven
 * per-frame in applyFrame via entityPopScale().
 */
function popInEntities() {
  state.popInStart = reducedMotion ? 0 : performance.now();
}

/** The current pop-in scale multiplier (1 once finished / when reduced motion). */
function entityPopScale() {
  if (!state.popInStart) return 1;
  const t = (performance.now() - state.popInStart) / state.popInDur;
  if (t >= 1) { state.popInStart = 0; return 1; }
  // easeOutBack-ish: small overshoot then settle, clamped >= 0.
  const e = easeOutCubic(Math.max(0, t));
  const overshoot = 1 + 0.10 * Math.sin(Math.min(1, t) * Math.PI);
  return Math.max(0.001, e * overshoot);
}

/**
 * Create the reusable visual group for one entity slot, sized/colored by its
 * static metadata. Returns null for padded/never-active slots (caller filters).
 *
 * @param {object} meta - static entity metadata {id,type,team,size,mass,is_decoy}
 * @returns {EntityViz}
 */
function makeEntityViz(meta) {
  const group = new THREE.Group();
  const baseColor = entityColorFor(meta);
  const isAgent = meta.type === "hider" || meta.type === "seeker";
  const size = meta.size || 0.5;

  let body, nose, cone, decoyRing, aura;

  if (isAgent) {
    // Rounded capsule body, smooth & slightly glossy.
    const mat = new THREE.MeshStandardMaterial({
      color: baseColor,
      roughness: 0.4,
      metalness: 0.05,
      emissive: baseColor,
      emissiveIntensity: PAL.agentEmissive,
    });
    body = new THREE.Mesh(GEO.capsule, mat);
    body.scale.setScalar(size / 0.4); // agent_radius default 0.4
    body.position.y = 0.55 * (size / 0.4);
    body.castShadow = true;
    group.add(body);

    // Soft glow POOL on the floor under the agent (clean reference-style halo) --
    // a flat additive radial disc, NOT a 3D sphere blob.
    const auraMat = new THREE.MeshBasicMaterial({
      map: softGlowTexture(),
      color: baseColor,
      transparent: true,
      opacity: 0.42,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    aura = new THREE.Mesh(GEO.auraDisc, auraMat);
    aura.rotation.x = -Math.PI / 2;
    const auraScale = (size / 0.4) * 2.3;
    aura.scale.set(auraScale, auraScale, 1);
    aura.position.y = 0.03;
    group.add(aura);

    // Small heading "beak": a subtle body-tinted nub showing facing (+Z local).
    const noseMat = new THREE.MeshStandardMaterial({
      color: baseColor,
      emissive: baseColor,
      emissiveIntensity: PAL.agentEmissive,
      roughness: 0.5,
    });
    nose = new THREE.Mesh(GEO.nose, noseMat);
    nose.rotation.x = Math.PI / 2;            // point cone along +Z
    nose.position.set(0, 0.5 * (size / 0.4), 0.34 * (size / 0.4));
    group.add(nose);

    // Two small dark "eyes" on the front -> a friendly face (reference style).
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0x222831, roughness: 0.5 });
    for (const ex of [-0.12, 0.12]) {
      const eye = new THREE.Mesh(GEO.eye, eyeMat);
      const sc = size / 0.4;
      eye.scale.setScalar(sc);
      eye.position.set(ex * sc, 0.62 * sc, 0.27 * sc);
      group.add(eye);
    }

    // Vision cone (hidden unless toggled). Flat translucent wedge on the floor.
    cone = makeVisionCone(baseColor);
    cone.visible = false;
    group.add(cone);
  } else if (meta.type === "box_light" || meta.type === "box_heavy") {
    const heavy = meta.type === "box_heavy";
    // Warm crate with a stamped lock emblem. The emblem texture is OPAQUE and
    // baked in the box's own colour, so every face shows colour + emblem -- no
    // black faces (the old transparent map multiplied the side faces to black).
    const mat = new THREE.MeshStandardMaterial({
      map: lockEmblemTexture(baseColor),
      roughness: heavy ? 0.6 : 0.65,
      metalness: 0.0,
      emissiveIntensity: 0,   // keep baseEmissive 0 -> no self-glow on boxes
    });
    body = new THREE.Mesh(GEO.roundedCube, mat);
    const s = size * (heavy ? 1.3 : 1.0) * 2; // size is roughly a half-extent
    body.scale.set(s, s, s);
    body.position.y = s / 2;
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);

    // Heavy boxes get a faint darker amber outline -- signals "needs coop".
    if (heavy) {
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
        new THREE.LineBasicMaterial({ color: col(PAL.edgeHeavy), transparent: true, opacity: 0.5 })
      );
      edges.scale.copy(body.scale);
      edges.position.copy(body.position);
      group.add(edges);
    }
  } else if (meta.type === "ramp") {
    // A light warm/neutral wedge (inclined plane) so climbing reads in 3D.
    body = new THREE.Mesh(makeWedgeGeometry(size), new THREE.MeshStandardMaterial({
      color: baseColor, roughness: 0.85, metalness: 0.0,
    }));
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);
  } else if (meta.type === "wall") {
    // THICK, chunky white block -- beefier than a thin slab.
    const w = (size || 1) * 2;
    body = new THREE.Mesh(GEO.cube, new THREE.MeshStandardMaterial({
      color: baseColor, roughness: 0.9, metalness: 0.0,
    }));
    body.scale.set(w, 2.2, 0.7);
    body.position.y = 1.1;
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);
  } else if (meta.type === "door") {
    // A chunky cool-light slab that slides / fades as it deactivates (opens).
    const w = (size || 1) * 2;
    body = new THREE.Mesh(GEO.cube, new THREE.MeshStandardMaterial({
      color: baseColor, roughness: 0.7, metalness: 0.0,
      transparent: true, opacity: 0.95,
    }));
    body.scale.set(w, 2.0, 0.5);
    body.position.y = 1.0;
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);
  } else if (meta.type === "decoy") {
    // A soft glowing orb; an expanding ring pulse is added on top. Additive
    // glow halo keeps it readable against the bright background.
    body = new THREE.Mesh(new THREE.SphereGeometry(0.26, 20, 16),
      new THREE.MeshStandardMaterial({
        color: baseColor, emissive: baseColor, emissiveIntensity: PAL.decoyEmissive,
        roughness: 0.35, metalness: 0.1,
      }));
    body.position.y = 0.4;
    group.add(body);

    // Additive glow shell around the orb.
    const glow = new THREE.Mesh(GEO.aura, new THREE.MeshBasicMaterial({
      color: baseColor, transparent: true, opacity: 0.28,
      blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.BackSide,
    }));
    glow.scale.setScalar(1.1);
    glow.position.y = 0.4;
    group.add(glow);

    decoyRing = new THREE.Mesh(
      new THREE.RingGeometry(0.3, 0.42, 40),
      new THREE.MeshBasicMaterial({
        color: baseColor, transparent: true, opacity: 0.0,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide, depthWrite: false,
      })
    );
    decoyRing.rotation.x = -Math.PI / 2;
    decoyRing.position.y = 0.05;
    group.add(decoyRing);
  } else {
    // Fallback: a plain cube.
    body = new THREE.Mesh(GEO.cube, new THREE.MeshStandardMaterial({ color: baseColor }));
    body.position.y = 0.5;
    group.add(body);
  }

  // "Spotted" ground ring + emissive boost target -- hidden unless sn=1.
  // Additive amber/orange so it still reads as a warm glow on the white floor.
  const spotRing = new THREE.Mesh(
    GEO.spottedRing,
    new THREE.MeshBasicMaterial({
      color: col(PAL.spotted), transparent: true, opacity: 0.0,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide, depthWrite: false,
    })
  );
  spotRing.rotation.x = -Math.PI / 2;
  spotRing.position.y = 0.03;
  group.add(spotRing);

  // Make the body pickable for hover/click; tag it with the entity id.
  group.userData.entityId = meta.id;
  if (body) body.userData.entityId = meta.id;
  // (Box bodies use a material array; record a single representative material
  // for the spotted-glow emissive logic via baseEmissive below.)
  const bodyMat = body && Array.isArray(body.material) ? body.material[0] : body && body.material;

  return {
    group, body, nose, cone, spotRing, decoyRing, aura,
    type: meta.type, team: meta.team, id: meta.id,
    baseEmissive: bodyMat && bodyMat.emissiveIntensity !== undefined
      ? bodyMat.emissiveIntensity : 0,
    isAgent,
  };
}

/** Resolve an entity's THREE.Color from its static metadata (active theme). */
function entityColorFor(meta) {
  if (meta.type === "hider") return col(PAL.hider);
  if (meta.type === "seeker") return col(PAL.seeker);
  if (meta.type in PAL) return col(PAL[meta.type]);
  return col(PAL.muted);
}

/** Build a translucent flat vision wedge (fan) on the floor in front of an agent. */
function makeVisionCone(color) {
  const half = (VISION_CONE_DEG * Math.PI) / 180 / 2;
  const segments = 24;
  const r = VISION_CONE_RANGE;
  const positions = [0, 0, 0];
  for (let i = 0; i <= segments; i++) {
    const a = -half + (2 * half * i) / segments;
    // Local +Z is "forward"; fan opens around +Z.
    positions.push(Math.sin(a) * r, 0, Math.cos(a) * r);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const indices = [];
  for (let i = 1; i <= segments; i++) indices.push(0, i, i + 1);
  geo.setIndex(indices);
  geo.computeVertexNormals();
  const mat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.08,
    side: THREE.DoubleSide, depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.y = 0.03;
  return mesh;
}

/** Build a simple triangular-prism wedge geometry for ramps (climbable slope). */
function makeWedgeGeometry(size) {
  const w = (size || 1) * 1.4;   // width across
  const l = (size || 1) * 2.2;   // length of the slope run
  const h = (size || 1) * 1.2;   // peak height
  // Triangular prism: cross-section is a right triangle (rises along +Z).
  const verts = new Float32Array([
    // left side triangle
    -w, 0, -l,  -w, 0, l,  -w, h, l,
    // right side triangle
    w, 0, -l,   w, h, l,   w, 0, l,
    // sloped top quad (two tris)
    -w, 0, -l,  -w, h, l,  w, h, l,
    -w, 0, -l,  w, h, l,   w, 0, -l,
    // bottom quad
    -w, 0, -l,  w, 0, -l,  w, 0, l,
    -w, 0, -l,  w, 0, l,   -w, 0, l,
    // back (tall) quad
    -w, 0, l,   w, 0, l,   w, h, l,
    -w, 0, l,   w, h, l,   -w, h, l,
  ]);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geo.computeVertexNormals();
  return geo;
}

/**
 * Rebuild the entire entity registry for a freshly loaded trajectory. Disposes
 * the previous meshes and creates one viz per ever-active slot.
 */
function buildEntities(traj) {
  // Clear previous -- dispose each entity group's per-entity materials and any
  // NON-shared geometries before detaching, so rebuilds (trajectory load OR
  // theme toggle) don't leak GPU resources. IMPORTANT: the geometries in the
  // shared GEO cache (capsule/cube/etc.) are reused by every entity and must
  // NOT be disposed, or the next build would render with dead buffers.
  for (const viz of state.vizById.values()) {
    disposeEntityGroup(viz.group);
    state.entityRoot.remove(viz.group);
  }
  state.vizById.clear();
  clearGroup(state.linkRoot);
  clearGroup(state.fogRoot);
  clearGroup(state.trailRoot);
  state.trails.clear();

  for (const meta of traj.entities) {
    // Hide padded / never-active slots entirely.
    if (!traj.everActive[meta.id]) continue;
    const viz = makeEntityViz(meta);
    state.vizById.set(meta.id, viz);
    state.entityRoot.add(viz.group);
    if (viz.isAgent) state.trails.set(meta.id, []);
  }
}

// ============================================================================
// [G] Per-frame update -- map sampled trajectory state onto the meshes
// ============================================================================

// Reusable scratch objects to avoid per-frame allocation.
const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();

/**
 * Dispose every child's geometry + material, then detach them from the group.
 * Use this instead of Group.clear() for groups whose children are rebuilt each
 * frame (link lines, trails, fog): clear() alone leaks the underlying WebGL
 * buffers/programs, which steadily grows GPU memory over a long playback.
 * (Declared as a hoisted function so earlier callers like buildEntities resolve.)
 */
function clearGroup(g) {
  for (const c of g.children) {
    if (c.geometry) c.geometry.dispose();
    if (c.material) {
      if (Array.isArray(c.material)) c.material.forEach((m) => m.dispose());
      else c.material.dispose();
    }
  }
  g.clear();
}

/**
 * Apply a sampled frame to all entity meshes, link lines, fog and overlays.
 * @param {object} f - the result of Trajectory.sample(pos)
 * @param {number} timeSec - wall-clock seconds (for gentle anim phases)
 */
function applyFrame(f, timeSec) {
  const traj = state.traj;

  // Shared per-frame entity pop-in scale (1 once the pop-in animation ends).
  const popScale = entityPopScale();

  // ---- entities --------------------------------------------------------
  for (const en of f.ent) {
    const viz = state.vizById.get(en.id);
    if (!viz) continue;

    // Inactive slots: broken walls / opened doors / despawned props fade out.
    const active = !!en.a;
    if (!active && viz.type !== "door") {
      viz.group.visible = false;
      continue;
    }
    viz.group.visible = true;

    // World placement. traj (x, y, z) -> three (x, z=traj.y, y=traj.z).
    viz.group.position.set(en.x, en.z, en.y);
    // Heading: rotate the whole group about world-up (y). Traj heading is a
    // standard math angle in the floor (x, y) plane; map to a Y rotation.
    viz.group.rotation.y = -en.h;

    // Entity pop-in: scale the whole group up from ~0 when a scenario loads.
    viz.group.scale.setScalar(popScale);

    // Doors: as they deactivate (a=0) they "open" -- slide aside + fade out
    // instead of simply vanishing, so the chokepoint reads as cleared.
    if (viz.type === "door") {
      const open = !active;
      const m = viz.body.material;
      // Animate the slab toward its open pose (slid aside + faded), or back.
      const targetX = open ? viz.body.scale.x * 0.9 : 0;
      const targetOp = open ? 0.0 : 0.92;
      viz.body.position.x += (targetX - viz.body.position.x) * 0.18;
      m.opacity += (targetOp - m.opacity) * 0.12;
      // Fully cleared doors stop rendering once essentially invisible.
      if (open && m.opacity <= 0.03) viz.group.visible = false;
    }

    // "Spotted" highlight: emissive boost + ground ring when seen by the foe.
    const spotted = !!en.sn;
    if (viz.body && viz.body.material.emissive) {
      const target = spotted
        ? 0.9
        : viz.baseEmissive;
      // Lerp emissive intensity toward target for a soft glow-in.
      const cur = viz.body.material.emissiveIntensity;
      viz.body.material.emissiveIntensity = cur + (target - cur) * 0.25;
      if (spotted) viz.body.material.emissive.copy(col(PAL.spotted));
      else viz.body.material.emissive.copy(entityColorFor(traj.staticOf(en.id)));
    }
    if (viz.spotRing) {
      const targetOp = spotted ? 0.55 + 0.25 * Math.sin(timeSec * 6) : 0.0;
      const m = viz.spotRing.material;
      m.opacity += (targetOp - m.opacity) * 0.3;
    }

    // Vision cones (agents): show/hide per the toggle.
    if (viz.cone) viz.cone.visible = state.showCones;

    // Agent aura: gentle breathing glow; brightens softly when spotted.
    if (viz.aura) {
      const base = spotted ? 0.32 : 0.18;
      viz.aura.material.opacity = base + 0.05 * Math.sin(timeSec * 2 + en.id);
    }

    // Decoy pulse: expanding ring while dc=1, intensity scaled by noise.
    if (viz.decoyRing) {
      if (en.dc) {
        const period = 1.1;
        const phase = (timeSec % period) / period;
        const ringScale = 0.6 + phase * 4.5 * (0.5 + 0.5 * (en.no || 1));
        viz.decoyRing.scale.setScalar(ringScale);
        viz.decoyRing.material.opacity = (1 - phase) * 0.6;
        // Core glow throbs too.
        if (viz.body && viz.body.material) {
          viz.body.material.emissiveIntensity = 0.8 + 0.6 * Math.sin(timeSec * 8);
        }
      } else {
        viz.decoyRing.material.opacity *= 0.85;
        if (viz.body && viz.body.material) viz.body.material.emissiveIntensity = 0.4;
      }
    }

    // God-view: tint TRUE decoys (privileged identity) regardless of dc.
    if (state.godView) {
      const trueDecoy = traj.staticOf(en.id).is_decoy;
      if (trueDecoy && viz.body && viz.body.material.emissive) {
        viz.body.material.emissive.copy(col(PAL.decoy));
        viz.body.material.emissiveIntensity = Math.max(
          viz.body.material.emissiveIntensity, 0.6
        );
      }
    }
  }

  // ---- held-object link lines -----------------------------------------
  clearGroup(state.linkRoot);
  for (const en of f.ent) {
    if (!en.hd || en.hb < 0) continue;
    const holder = state.vizById.get(en.hb);
    const held = state.vizById.get(en.id);
    if (!holder || !held || !holder.group.visible || !held.group.visible) continue;
    _v.copy(held.group.position).y += 0.4;
    _v2.copy(holder.group.position).y += 0.5;
    const geo = new THREE.BufferGeometry().setFromPoints([_v.clone(), _v2.clone()]);
    const line = new THREE.Line(
      geo,
      new THREE.LineBasicMaterial({
        color: col(PAL.linkLine), transparent: true, opacity: 0.6,
      })
    );
    state.linkRoot.add(line);
  }

  // ---- fog discs -------------------------------------------------------
  updateFog(f.fog, timeSec);

  // ---- trails ----------------------------------------------------------
  updateTrails(f);
}

/** (Re)draw soft translucent fog discs at each (x, y, r) patch, gently breathing. */
function updateFog(patches, timeSec) {
  // Rebuild lazily only when count changes; else reuse + animate.
  if (!state.showFog) {
    state.fogRoot.visible = false;
    return;
  }
  state.fogRoot.visible = true;

  // Ensure we have one disc mesh per patch.
  while (state.fogRoot.children.length < patches.length) {
    const m = new THREE.Mesh(
      new THREE.CircleGeometry(1, 40),
      new THREE.MeshBasicMaterial({
        color: col(PAL.fogDisc), transparent: true, opacity: 0.1,
        side: THREE.DoubleSide, depthWrite: false,
      })
    );
    m.rotation.x = -Math.PI / 2;
    state.fogRoot.add(m);
  }
  while (state.fogRoot.children.length > patches.length) {
    state.fogRoot.remove(state.fogRoot.children[state.fogRoot.children.length - 1]);
  }

  patches.forEach((p, i) => {
    const m = state.fogRoot.children[i];
    const breathe = 1 + 0.05 * Math.sin(timeSec * 1.3 + i * 1.7);
    m.position.set(p[0], 0.04 + 0.01 * Math.sin(timeSec + i), p[1]);
    m.scale.setScalar(p[2] * breathe);
    m.material.opacity = PAL.fogDiscAlpha + 0.03 * Math.sin(timeSec * 0.9 + i);
  });
}

/** Accumulate fading path lines behind each agent when trails are enabled. */
function updateTrails(f) {
  clearGroup(state.trailRoot);
  if (!state.showTrails) return;

  const MAX = 60;
  for (const en of f.ent) {
    const viz = state.vizById.get(en.id);
    if (!viz || !viz.isAgent || !en.a) continue;
    const buf = state.trails.get(en.id);
    if (!buf) continue;
    const last = buf[buf.length - 1];
    const here = new THREE.Vector3(en.x, 0.05, en.y);
    if (!last || last.distanceToSquared(here) > 0.01) {
      buf.push(here);
      if (buf.length > MAX) buf.shift();
    }
    if (buf.length < 2) continue;
    const geo = new THREE.BufferGeometry().setFromPoints(buf);
    const base = en.id; // not used; color by team below
    const meta = state.traj.staticOf(en.id);
    const line = new THREE.Line(
      geo,
      new THREE.LineBasicMaterial({
        color: entityColorFor(meta), transparent: true, opacity: 0.4,
      })
    );
    state.trailRoot.add(line);
  }
}

// ============================================================================
// [H] HUD construction & wiring (built in code as glass overlays)
// ============================================================================

const ui = {}; // populated below with element references

/** Build the entire HUD DOM and attach it to #app. Idempotent. */
function buildHUD() {
  const app = document.getElementById("app");

  // ---- top HUD bar -----------------------------------------------------
  const top = el("div", "glass overlay", { id: "hud-top" });
  top.innerHTML = `
    <button class="btn panel-toggle watch-only" id="btn-panel" title="Toggle panel">
      <svg viewBox="0 0 24 24"><path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" stroke-width="2" fill="none"/></svg>
    </button>
    <div class="hud-title">
      <span class="t1" id="traj-title">Hide &amp; Seek 2.0</span>
      <span class="t2">Tactical Trajectory Viewer</span>
    </div>
    <div class="pill prep watch-only" id="phase-pill"><span class="dot"></span><span id="phase-text">PREP</span></div>

    <nav class="tab-bar" id="tab-bar" role="tablist">
      <button class="tab" data-tab="watch" role="tab">Watch</button>
      <button class="tab" data-tab="learning" role="tab">Learning</button>
      <button class="tab" data-tab="about" role="tab">About</button>
    </nav>

    <div class="hud-spacer"></div>
    <div class="scores watch-only">
      <div class="score hiders"><span class="lbl">Hiders</span><span class="val" id="score-h">0</span></div>
      <div class="score"><span class="vs">vs</span></div>
      <div class="score seekers"><span class="lbl">Seekers</span><span class="val" id="score-s">0</span></div>
    </div>
    <div class="step-counter watch-only">t <b id="step-cur">0</b> <span class="max">/ <span id="step-max">0</span></span></div>
    <div class="spotted-ind watch-only" id="spotted-ind"><span class="dot"></span>SPOTTED</div>
    <select class="scenario-select watch-only" id="scenario-select" title="Choose a scenario"></select>
    <button class="icon-btn watch-only" id="btn-load" title="Load a trajectory file">
      <svg viewBox="0 0 24 24"><path d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16" stroke="currentColor" stroke-width="2" fill="none"/></svg>
      Load
    </button>
    <button class="icon-btn theme-toggle" id="btn-theme" title="Toggle light / dark theme" aria-label="Toggle theme">
      <span class="theme-ic" id="theme-ic"></span>
    </button>
  `;
  app.appendChild(top);

  // ---- left control panel ---------------------------------------------
  const left = el("div", "glass overlay watch-only", { id: "panel-left" });
  left.innerHTML = `
    <div class="panel-head"><span>Controls</span></div>
    <div class="panel-body">
      <div class="panel-section">
        <div class="sec-title">Legend</div>
        <div class="legend" id="legend"></div>
      </div>
      <div class="panel-section">
        <div class="sec-title">Layers</div>
        ${toggleHTML("tg-cones", "Vision cones")}
        ${toggleHTML("tg-fog", "Fog of war", true)}
        ${toggleHTML("tg-decoys", "Reveal decoys (god)")}
        ${toggleHTML("tg-grid", "Grid", true)}
        ${toggleHTML("tg-trails", "Trails")}
        ${toggleHTML("tg-follow", "Follow camera")}
      </div>
    </div>
  `;
  app.appendChild(left);

  // ---- inspector -------------------------------------------------------
  const insp = el("div", "glass overlay watch-only", { id: "inspector" });
  insp.hidden = true;
  app.appendChild(insp);

  // ---- bottom transport ------------------------------------------------
  const tr = el("div", "glass overlay watch-only", { id: "transport" });
  tr.innerHTML = `
    <div class="transport-row">
      <div class="scrubber-wrap">
        <div class="scrubber-track"><div class="scrubber-fill" id="scrub-fill"></div></div>
        <div class="scrubber-tick" id="scrub-tick" style="left:0%"></div>
        <input type="range" class="scrubber" id="scrubber" min="0" max="100" step="0.01" value="0" />
      </div>
    </div>
    <div class="transport-row">
      <button class="btn primary" id="btn-play" title="Play / Pause (Space)">
        <svg viewBox="0 0 24 24" id="ic-play"><path d="M8 5v14l11-7z"/></svg>
      </button>
      <button class="btn" id="btn-prev" title="Step back (Left)">
        <svg viewBox="0 0 24 24"><path d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/></svg>
      </button>
      <button class="btn" id="btn-next" title="Step forward (Right)">
        <svg viewBox="0 0 24 24"><path d="M16 6h2v12h-2zM6 6l8.5 6L6 18z"/></svg>
      </button>
      <div class="speed-group" id="speed-group">
        ${SPEEDS.map((s) => `<button data-speed="${s}" class="${s === 1 ? "active" : ""}">${s}&times;</button>`).join("")}
      </div>
      <button class="btn active" id="btn-loop" title="Loop">
        <svg viewBox="0 0 24 24"><path d="M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z"/></svg>
      </button>
      <div class="frame-readout">frame <b id="fr-cur">0</b>/<span id="fr-max">0</span> &middot; <b id="fr-time">0.0s</b></div>
    </div>
  `;
  app.appendChild(tr);

  // ---- bottom-center scenario caption (OpenAI-clip style) -------------
  const cap = el("div", "watch-only", { id: "scenario-caption" });
  cap.innerHTML = `<span class="cap-title" id="cap-title"></span><span class="cap-sub" id="cap-sub"></span>`;
  app.appendChild(cap);

  // ---- Learning tab panel (scrollable dashboard; built lazily) --------
  const learn = el("div", "tab-panel", { id: "panel-learning" });
  learn.hidden = true;
  learn.innerHTML = `<div class="tab-scroll"><div id="learning-root" class="learning-root"></div></div>`;
  app.appendChild(learn);

  // ---- About tab panel -------------------------------------------------
  const about = el("div", "tab-panel", { id: "panel-about" });
  about.hidden = true;
  about.innerHTML = aboutHTML();
  app.appendChild(about);

  // ---- drag-drop hint overlay -----------------------------------------
  const drop = el("div", "", { id: "drop-hint" });
  drop.innerHTML = `<div class="drop-card">Drop an hns2-traj JSON file<small>to load it into the viewer</small></div>`;
  app.appendChild(drop);

  // ---- toast -----------------------------------------------------------
  const toast = el("div", "glass", { id: "toast" });
  app.appendChild(toast);

  // ---- hidden file input ----------------------------------------------
  const fileInput = el("input", "", { id: "file-input", type: "file", accept: ".json,application/json" });
  fileInput.style.display = "none";
  app.appendChild(fileInput);

  // ---- cache references ------------------------------------------------
  ui.title = byId("traj-title");
  ui.phasePill = byId("phase-pill");
  ui.phaseText = byId("phase-text");
  ui.scoreH = byId("score-h");
  ui.scoreS = byId("score-s");
  ui.stepCur = byId("step-cur");
  ui.stepMax = byId("step-max");
  ui.spotted = byId("spotted-ind");
  ui.scrubber = byId("scrubber");
  ui.scrubFill = byId("scrub-fill");
  ui.scrubTick = byId("scrub-tick");
  ui.btnPlay = byId("btn-play");
  ui.icPlay = byId("ic-play");
  ui.frCur = byId("fr-cur");
  ui.frMax = byId("fr-max");
  ui.frTime = byId("fr-time");
  ui.loop = byId("btn-loop");
  ui.left = byId("panel-left");
  ui.inspector = byId("inspector");
  ui.legend = byId("legend");
  ui.capTitle = byId("cap-title");
  ui.capSub = byId("cap-sub");
  ui.scenarioSelect = byId("scenario-select");
  ui.fileInput = fileInput;
  ui.drop = drop;
  ui.toast = toast;
  ui.tabBar = byId("tab-bar");
  ui.tabs = [...top.querySelectorAll(".tab")];
  ui.panelLearning = byId("panel-learning");
  ui.panelAbout = byId("panel-about");
  ui.learningRoot = byId("learning-root");
  ui.btnTheme = byId("btn-theme");
  ui.themeIc = byId("theme-ic");
  ui.transport = tr;

  wireHUD();
  buildLegend();
  syncThemeButton();
}

/** Friendly, concise About-tab content (the five 2.0 mechanics + links). */
function aboutHTML() {
  return `
  <div class="tab-scroll">
    <div class="about-wrap">
      <h1 class="about-title">Hide &amp; Seek <span class="about-badge">2.0</span></h1>
      <p class="about-lead">
        A tiny multi-agent world where two teams play hide-and-seek. The
        <b class="c-hider">Hiders</b> try to stay out of sight; the
        <b class="c-seeker">Seekers</b> try to find them. Nobody is told how to
        play — both teams <b>learn by self-play</b>, competing against past
        versions of themselves until clever tactics emerge on their own.
      </p>
      <p class="about-lead about-lead-sub">
        This viewer replays saved trajectories in 3D. Open
        <b>Watch</b> to scrub through a scenario, or <b>Learning</b> to see how
        the teams improved over training.
      </p>

      <h2 class="about-h2">The 2.0 mechanics</h2>
      <div class="about-grid">
        <div class="about-card">
          <div class="about-ic">\u{1F9F1}</div>
          <div class="about-card-t">Variable mass &amp; cooperative physics</div>
          <div class="about-card-d">Boxes have weight. A heavy box won't budge for
            one agent — two Hiders must push it together to build a fort.</div>
        </div>
        <div class="about-card">
          <div class="about-ic">\u{1FA84}</div>
          <div class="about-card-t">Decoys</div>
          <div class="about-card-d">Hiders can trigger decoys that look like them,
            sending Seekers chasing the wrong target.</div>
        </div>
        <div class="about-card">
          <div class="about-ic">\u{1F32B}️</div>
          <div class="about-card-t">Fog of war</div>
          <div class="about-card-d">Vision is limited and occluded. Each team only
            knows what it can actually see.</div>
        </div>
        <div class="about-card">
          <div class="about-ic">\u{1F6AA}</div>
          <div class="about-card-t">Destructible walls &amp; doors</div>
          <div class="about-card-d">Some walls can be broken and doors opened or
            jammed — chokepoints become contested ground.</div>
        </div>
        <div class="about-card">
          <div class="about-ic">\u{26A1}</div>
          <div class="about-card-t">Stamina</div>
          <div class="about-card-d">Sprinting drains stamina, so agents must spend
            their energy wisely during a chase.</div>
        </div>
      </div>

      <h2 class="about-h2">How they learn</h2>
      <p class="about-lead">
        The agents are trained by <b>self-play reinforcement learning</b>: they
        play millions of games against earlier copies of themselves. When one
        team discovers a new trick, the other is pressured to counter it — an
        ever-escalating arms race that produces surprisingly sophisticated,
        emergent behaviour.
      </p>

      <div class="about-links">
        <a class="about-link" href="https://github.com/GeFAA/hide-and-seek-2" target="_blank" rel="noopener">
          <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M12 .5A11.5 11.5 0 0 0 .5 12 11.5 11.5 0 0 0 8.4 23c.6.1.8-.3.8-.6v-2c-3.2.7-3.9-1.4-3.9-1.4-.5-1.3-1.3-1.7-1.3-1.7-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1 .1-.7.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.7 0-1.3.5-2.3 1.2-3.1-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0C17 4.2 18 4.5 18 4.5c.6 1.6.2 2.8.1 3.1.8.8 1.2 1.8 1.2 3.1 0 4.4-2.7 5.4-5.3 5.7.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A11.5 11.5 0 0 0 23.5 12 11.5 11.5 0 0 0 12 .5Z"/></svg>
          GitHub repo
        </a>
        <a class="about-link" href="https://github.com/GeFAA/hide-and-seek-2" target="_blank" rel="noopener">
          <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" d="M14 3h7v7m0-7L10 14M5 5h5M5 5v14h14v-5"/></svg>
          Live demo
        </a>
      </div>
    </div>
  </div>`;
}

/** Small DOM helpers. */
function el(tag, cls, attrs) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function byId(id) { return document.getElementById(id); }

/** HTML for one labeled checkbox toggle. */
function toggleHTML(id, label, checked) {
  return `<label class="toggle"><input type="checkbox" id="${id}"${checked ? " checked" : ""}/><span class="box"></span>${label}</label>`;
}

/** Populate the legend from the canonical entity types. */
function buildLegend() {
  const rows = [
    ["hider", "Hider"], ["seeker", "Seeker"],
    ["box_light", "Light box"], ["box_heavy", "Heavy box (coop)"],
    ["ramp", "Ramp"], ["decoy", "Decoy"],
    ["wall", "Wall"], ["door", "Door"],
    ["spotted", "Spotted"], ["fog", "Fog"],
  ];
  ui.legend.innerHTML = rows.map(([key, label]) => {
    const src = key === "fog" ? PAL.fogDisc : PAL[key];
    const hex = "#" + ((src != null ? src : PAL.muted) >>> 0).toString(16).padStart(6, "0");
    // A crisp 1px ring (via CSS var) so near-neutral swatches (wall/door) read
    // against the frosted panel in either theme.
    return `<div class="legend-row"><span class="swatch" style="background:${hex}"></span>${label}</div>`;
  }).join("");
}

/** Attach all event listeners for the HUD + keyboard + drag-drop. */
function wireHUD() {
  // transport
  ui.btnPlay.addEventListener("click", togglePlay);
  byId("btn-prev").addEventListener("click", () => stepBy(-1));
  byId("btn-next").addEventListener("click", () => stepBy(1));
  ui.loop.addEventListener("click", () => {
    state.loop = !state.loop;
    ui.loop.classList.toggle("active", state.loop);
  });
  byId("speed-group").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-speed]");
    if (!b) return;
    state.speed = parseFloat(b.dataset.speed);
    [...b.parentElement.children].forEach((c) => c.classList.toggle("active", c === b));
  });
  ui.scrubber.addEventListener("input", () => {
    if (!state.traj) return;
    const frac = parseFloat(ui.scrubber.value) / 100;
    state.pos = frac * (state.traj.nFrames - 1);
    // Scrubbing pauses playback so you can inspect a moment.
    if (state.playing) togglePlay();
  });

  // toggles
  bindToggle("tg-cones", "showCones");
  bindToggle("tg-fog", "showFog");
  bindToggle("tg-decoys", "godView");
  bindToggle("tg-grid", "showGrid", (v) => { gridGroup.visible = v; });
  bindToggle("tg-trails", "showTrails", (v) => { if (!v) clearGroup(state.trailRoot); });
  bindToggle("tg-follow", "followCam", (v) => {
    if (v && state.hoverId >= 0) state.followTargetId = state.hoverId;
  });

  // panel collapse
  byId("btn-panel").addEventListener("click", () => {
    ui.left.classList.toggle("collapsed");
  });

  // tab bar (Watch / Learning / About)
  ui.tabBar.addEventListener("click", (e) => {
    const b = e.target.closest(".tab[data-tab]");
    if (!b) return;
    switchTab(b.dataset.tab, { updateHash: true });
  });

  // theme toggle
  ui.btnTheme.addEventListener("click", () => {
    const next = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.classList.add("theme-anim"); // brief CSS cross-fade
    applyTheme(next);
    setTimeout(() => document.documentElement.classList.remove("theme-anim"), 360);
  });

  // file load
  byId("btn-load").addEventListener("click", () => ui.fileInput.click());
  ui.fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) loadFromFile(file);
    ui.fileInput.value = "";
  });

  // drag & drop (whole window) -- track an enter/leave DEPTH counter instead of
  // relying on relatedTarget (unreliable across browsers), so the full-screen
  // drop overlay never gets stuck covering the canvas.
  let _dragDepth = 0;
  window.addEventListener("dragenter", (e) => {
    e.preventDefault();
    _dragDepth++;
    ui.drop.classList.add("show");
  });
  window.addEventListener("dragover", (e) => { e.preventDefault(); });
  window.addEventListener("dragleave", (e) => {
    e.preventDefault();
    _dragDepth = Math.max(0, _dragDepth - 1);
    if (_dragDepth === 0) ui.drop.classList.remove("show");
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    _dragDepth = 0;
    ui.drop.classList.remove("show");
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) loadFromFile(file);
  });

  // keyboard: space = play/pause, arrows = step, T = toggle theme. Blur a
  // focused transport button first so Space isn't ALSO delivered as a click.
  window.addEventListener("keydown", (e) => {
    const tag = e.target && e.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    // Theme toggle works on every tab.
    if (e.code === "KeyT") {
      e.preventDefault();
      applyTheme(currentTheme === "dark" ? "light" : "dark");
      return;
    }
    // Playback keys only make sense on the Watch tab.
    if (state.tab !== "watch") return;
    if (e.code === "Space") {
      e.preventDefault();
      if (document.activeElement && typeof document.activeElement.blur === "function") {
        document.activeElement.blur();
      }
      togglePlay();
    } else if (e.code === "ArrowLeft") { e.preventDefault(); stepBy(-1); }
    else if (e.code === "ArrowRight") { e.preventDefault(); stepBy(1); }
  });

  // pointer picking for hover/click inspector
  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("click", onPointerClick);
}

/** Bind a checkbox to a boolean state field, with an optional side-effect. */
function bindToggle(id, field, onChange) {
  const cb = byId(id);
  if (!cb) return;
  // initialize element to current state
  cb.checked = !!state[field];
  cb.addEventListener("change", () => {
    state[field] = cb.checked;
    if (onChange) onChange(cb.checked);
  });
}

// ============================================================================
// [H2] Tabs, theme button, and hash routing
// ============================================================================

/** Update the theme toggle button icon (sun in dark mode, moon in light). */
function syncThemeButton() {
  if (!ui.themeIc) return;
  // Show the icon for the theme you would switch TO.
  const sun = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4.2" fill="currentColor"/><g stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="2.5" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="21.5"/><line x1="2.5" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="21.5" y2="12"/><line x1="4.9" y1="4.9" x2="6.7" y2="6.7"/><line x1="17.3" y1="17.3" x2="19.1" y2="19.1"/><line x1="4.9" y1="19.1" x2="6.7" y2="17.3"/><line x1="17.3" y1="6.7" x2="19.1" y2="4.9"/></g></svg>';
  const moon = '<svg viewBox="0 0 24 24"><path fill="currentColor" d="M20 14.5A8.5 8.5 0 0 1 9.5 4a7 7 0 1 0 10.5 10.5Z"/></svg>';
  ui.themeIc.innerHTML = currentTheme === "dark" ? sun : moon;
  ui.btnTheme.title = currentTheme === "dark"
    ? "Switch to light theme (T)" : "Switch to dark theme (T)";
}

/**
 * Switch the active tab. Watch shows the 3D canvas + overlays; Learning shows
 * the dashboard (canvas hidden, playback paused); About shows the explainer.
 * @param {"watch"|"learning"|"about"} name
 * @param {{updateHash?:boolean}} [opts]
 */
function switchTab(name, opts = {}) {
  const tab = ["watch", "learning", "about"].includes(name) ? name : "watch";
  state.tab = tab;

  // Tab button active states.
  if (ui.tabs) ui.tabs.forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));

  // Body class drives which overlays (watch-only) are visible + canvas dimming.
  document.body.dataset.tab = tab;

  const isWatch = tab === "watch";
  // Pause playback when leaving Watch so nothing animates off-screen.
  if (!isWatch && state.playing) togglePlay();

  // Toggle the two full-screen tab panels with a quick cross-fade.
  showPanel(ui.panelLearning, tab === "learning");
  showPanel(ui.panelAbout, tab === "about");

  // Lazily build / refresh the Learning dashboard the first time it's shown.
  if (tab === "learning") initLearning();

  if (isWatch) {
    // Reset the clock delta so playback doesn't jump after time off-screen.
    clock.getDelta();
    // Fire a queued camera intro, or just make sure controls are usable.
    if (cameraIntro.pending) startCameraIntro();
    else if (!cameraIntro.active) controls.enabled = true;
  }

  if (opts.updateHash !== false) updateHash();
}

/** Cross-fade a tab panel in/out (snaps if reduced motion). */
function showPanel(panelEl, show) {
  if (!panelEl) return;
  if (show) {
    panelEl.hidden = false;
    // force reflow so the opacity transition runs from 0
    void panelEl.offsetWidth;
    panelEl.classList.add("shown");
  } else {
    panelEl.classList.remove("shown");
    if (reducedMotion) {
      panelEl.hidden = true;
    } else {
      // hide after the fade so it doesn't intercept pointer events
      clearTimeout(panelEl.__hideT);
      panelEl.__hideT = setTimeout(() => { panelEl.hidden = true; }, 260);
    }
  }
}

/** Lazily init the Learning dashboard once; re-theme/redraw on later shows. */
function initLearning() {
  if (!ui.learningRoot) return;
  renderLearning(ui.learningRoot, {
    theme: currentTheme,
    getThemeColors: themeColorsForCharts,
    reducedMotion,
  }).then((inst) => { learning = inst; }).catch((err) => {
    console.error("Learning dashboard failed:", err);
  });
}

/**
 * Write the current tab + scenario into location.hash via replaceState (so it
 * doesn't spam history). Watch tab with a scenario -> "#scenario=<id>"; other
 * tabs -> "#tab=<name>". Combined form kept compact.
 */
function updateHash() {
  const parts = [];
  const tab = state.tab;
  const scn = ui.scenarioSelect && ui.scenarioSelect.value;
  if (tab === "watch") {
    // A bare scenario hash implies Watch -- keep it short & shareable.
    if (scn) parts.push("scenario=" + scn);
    else parts.push("tab=watch");
  } else {
    parts.push("tab=" + tab);
    if (scn) parts.push("scenario=" + scn);
  }
  const hash = "#" + parts.join("&");
  try {
    if (location.hash !== hash) history.replaceState(null, "", hash);
  } catch (_) { /* ignore */ }
}

/** Parse the location hash into {tab, scenario}. */
function parseHash() {
  const h = (location.hash || "").replace(/^#/, "");
  const out = { tab: null, scenario: null };
  for (const kv of h.split("&")) {
    const [k, v] = kv.split("=");
    if (k === "tab" && v) out.tab = decodeURIComponent(v);
    else if (k === "scenario" && v) out.scenario = decodeURIComponent(v);
  }
  // A bare "#scenario=fort" implies the Watch tab.
  if (!out.tab && out.scenario) out.tab = "watch";
  if (!["watch", "learning", "about"].includes(out.tab)) out.tab = out.tab ? "watch" : null;
  return out;
}

// ---- playback transport ----------------------------------------------------

function togglePlay() {
  state.playing = !state.playing;
  ui.icPlay.innerHTML = state.playing
    ? '<path d="M6 5h4v14H6zm8 0h4v14h-4z"/>'   // pause icon
    : '<path d="M8 5v14l11-7z"/>';               // play icon
  state.lastTime = performance.now();
}

function stepBy(n) {
  if (!state.traj) return;
  if (state.playing) togglePlay();
  state.pos = Math.max(0, Math.min(state.traj.nFrames - 1, Math.round(state.pos) + n));
}

// ============================================================================
// [I] Pointer picking + inspector
// ============================================================================

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function pickEntity(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const meshes = [];
  for (const viz of state.vizById.values()) {
    if (viz.group.visible && viz.body) meshes.push(viz.body);
  }
  const hits = raycaster.intersectObjects(meshes, false);
  return hits.length ? hits[0].object.userData.entityId : -1;
}

function onPointerMove(ev) {
  const id = pickEntity(ev);
  state.hoverId = id;
  renderer.domElement.style.cursor = id >= 0 ? "pointer" : "default";
  if (state.pinnedId < 0) updateInspector(id);
}

function onPointerClick(ev) {
  const id = pickEntity(ev);
  if (id >= 0) {
    state.pinnedId = id;
    if (state.followCam) state.followTargetId = id;
    updateInspector(id);
  } else {
    state.pinnedId = -1;
    updateInspector(-1);
  }
}

/** Render the inspector panel for an entity id (-1 hides it). */
function updateInspector(id) {
  if (id < 0 || !state.traj) { ui.inspector.hidden = true; return; }
  const meta = state.traj.staticOf(id);
  if (!meta) { ui.inspector.hidden = true; return; }
  // Current dynamic state at the displayed frame.
  const f = state.traj.frameAt(Math.round(state.pos));
  const en = f.ent[id];
  const color = "#" + (entityColorFor(meta).getHex() >>> 0).toString(16).padStart(6, "0");
  const teamName = meta.team === 0 ? "Hider" : meta.team === 1 ? "Seeker" : "—";

  const flag = (v) => `<span class="v ${v ? "flag-on" : "flag-off"}">${v ? "yes" : "no"}</span>`;
  const stam = meta.team >= 0 && en.st >= 0
    ? `<span class="stamina-bar"><i style="width:${Math.round(en.st * 100)}%"></i></span>`
    : `<span class="v">—</span>`;

  let decoyRow = "";
  if (state.godView) {
    decoyRow = `<div class="insp-row"><span class="k">decoy?</span>` +
      `<span class="v ${meta.is_decoy ? "decoy-true" : ""}">${meta.is_decoy ? "TRUE DECOY" : "real"}</span></div>`;
  }

  ui.inspector.innerHTML = `
    <div class="insp-head">
      <span class="swatch" style="background:${color};box-shadow:0 0 8px ${color}"></span>
      <span class="insp-type">${meta.type.replace("_", " ")}</span>
    </div>
    <div class="insp-row"><span class="k">id</span><span class="v">${meta.id}</span></div>
    <div class="insp-row"><span class="k">team</span><span class="v">${teamName}</span></div>
    <div class="insp-row"><span class="k">mass</span><span class="v">${meta.mass.toFixed(1)}</span></div>
    <div class="insp-row"><span class="k">size</span><span class="v">${meta.size.toFixed(2)}</span></div>
    <div class="insp-row"><span class="k">locked</span>${flag(en.lk)}</div>
    <div class="insp-row"><span class="k">grounded</span>${flag(en.gr)}</div>
    <div class="insp-row"><span class="k">held</span>${flag(en.hd)}</div>
    <div class="insp-row"><span class="k">stamina</span>${stam}</div>
    ${decoyRow}
  `;
  ui.inspector.hidden = false;
}

// ============================================================================
// [J] HUD refresh (per-frame text) + toast
// ============================================================================

/** Push the sampled frame's status into the HUD text/widgets. */
function refreshHUD(f) {
  const traj = state.traj;
  const prep = f.phase === "prep";
  ui.phasePill.className = "pill " + (prep ? "prep" : "main");
  ui.phaseText.textContent = prep ? "PREP" : "SEEK";
  if (ui.capSub) ui.capSub.textContent = prep ? "Preparation phase" : "Seek phase";
  // Scores read from the interpolated sample, so they already glide smoothly;
  // give the digits a tiny "pop" whenever the rounded value actually changes.
  setScore(ui.scoreH, Math.round(f.sh));
  setScore(ui.scoreS, Math.round(f.ss));
  ui.stepCur.textContent = f.t;
  ui.stepMax.textContent = traj.maxSteps;
  ui.spotted.classList.toggle("live", !!f.seen_any);

  const frac = traj.nFrames > 1 ? state.pos / (traj.nFrames - 1) : 0;
  ui.scrubber.value = (frac * 100).toFixed(2);
  ui.scrubFill.style.width = (frac * 100).toFixed(2) + "%";
  ui.frCur.textContent = Math.round(state.pos);
  ui.frMax.textContent = traj.nFrames - 1;
  ui.frTime.textContent = (state.pos * traj.dt).toFixed(1) + "s";

  // refresh pinned inspector continuously (stamina etc. change per frame)
  if (state.pinnedId >= 0) updateInspector(state.pinnedId);
}

/** Set a score readout, pulsing the digits when the integer value changes. */
function setScore(node, val) {
  if (!node) return;
  if (node.__val === val) return;
  node.__val = val;
  node.textContent = val;
  if (reducedMotion) return;
  node.classList.remove("bump");
  void node.offsetWidth; // restart the CSS animation
  node.classList.add("bump");
}

/** Flash a transient status message. kind: "ok" | "warn" | "err". */
let _toastTimer = null;
function toast(msg, kind = "ok") {
  ui.toast.textContent = msg;
  ui.toast.className = "glass " + kind + " show";
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { ui.toast.className = "glass " + kind; }, 2600);
}

// ============================================================================
// [K] Loading: default fetch / file picker / drag-drop
// ============================================================================

/** Install a freshly parsed Trajectory into the scene and reset playback. */
function installTrajectory(traj) {
  state.traj = traj;
  buildArena(traj.bound);
  buildEntities(traj);
  state.pos = 0;
  state.lastEvFrame = -1;
  state.pinnedId = -1;
  state.followTargetId = -1;
  updateInspector(-1);

  // Frame the camera target on the arena center (raised slightly so the arena
  // sits centered in the closer 3/4 view).
  controls.target.set(0, 0.9, 0);

  // Entities pop in with a quick eased scale.
  popInEntities();

  // Position the phase-boundary tick on the scrubber.
  const bf = traj.phaseBoundaryFrame;
  const pct = traj.nFrames > 1 ? (bf / (traj.nFrames - 1)) * 100 : 0;
  ui.scrubTick.style.left = pct + "%";
  ui.scrubTick.style.display = bf > 0 && bf < traj.nFrames - 1 ? "block" : "none";

  ui.title.textContent = traj.title;
  if (ui.capTitle) ui.capTitle.textContent = traj.title;
  if (ui.capSub) ui.capSub.textContent = "";   // scenario loader sets a description

  // Auto-play on load -- but only on the Watch tab (so opening on #tab=learning
  // doesn't run playback off-screen).
  if (state.tab === "watch" && !state.playing) togglePlay();

  toast(`Loaded ${traj.title} — ${traj.nFrames} frames, ${state.vizById.size} entities`, "ok");
}

let _manifest = null;

/** Fetch the scenario manifest, populate the picker, and load the default one. */
async function loadScenarios() {
  try {
    const res = await fetch("./trajectories/manifest.json", { cache: "no-cache" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const man = await res.json();
    if (!man || !Array.isArray(man.scenarios) || !man.scenarios.length) {
      throw new Error("empty manifest");
    }
    _manifest = man;
    const sel = ui.scenarioSelect;
    if (sel) {
      sel.innerHTML = man.scenarios
        .map((s) => `<option value="${s.id}">${s.title}</option>`)
        .join("");
      sel.addEventListener("change", () => loadScenarioById(sel.value));
    }
    const hasDefault = man.default && man.scenarios.some((s) => s.id === man.default);
    const route = parseHash();
    const hashId = route.scenario;
    const wanted = hashId && man.scenarios.some((s) => s.id === hashId) ? hashId
                 : (hasDefault ? man.default : man.scenarios[0].id);
    if (sel) sel.value = wanted;
    // Don't write the hash yet -- boot() applies tab routing right after this.
    await loadScenarioById(wanted, { updateHash: false });
    return true;
  } catch (err) {
    console.warn("No scenario manifest:", err);
    return false;
  }
}

/**
 * Load one scenario by manifest id and set the caption description.
 * @param {string} id
 * @param {{updateHash?:boolean}} [opts] - pass {updateHash:false} during boot,
 *   before tab routing has run, so we don't clobber the incoming hash.
 */
async function loadScenarioById(id, opts = {}) {
  const entry = _manifest && _manifest.scenarios.find((s) => s.id === id);
  if (!entry) return;
  try {
    const res = await fetch("./trajectories/" + entry.file, { cache: "no-cache" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const doc = await res.json();
    installTrajectory(Trajectory.parse(doc));
    if (ui.scenarioSelect) ui.scenarioSelect.value = id;
    if (ui.capSub) ui.capSub.textContent = entry.description || "";
    if (opts.updateHash !== false) updateHash();
  } catch (err) {
    toast("Could not load scenario: " + (err.message || id), "err");
    console.error(err);
  }
}

/** Fetch + parse the default trajectory JSON (fallback when no manifest). */
async function loadDefault() {
  try {
    const res = await fetch(DEFAULT_TRAJ_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const doc = await res.json();
    installTrajectory(Trajectory.parse(doc));
  } catch (err) {
    // Not fatal: the scene still runs; invite the user to load a file.
    toast(
      "No demo trajectory found — drop or load an hns2-traj file.",
      "warn"
    );
    console.warn("Default trajectory not loaded:", err);
  }
}

/** Read + parse a user-selected / dropped File. */
function loadFromFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const doc = JSON.parse(reader.result);
      installTrajectory(Trajectory.parse(doc));
    } catch (err) {
      toast("Could not load file: " + (err.message || "invalid JSON"), "err");
      console.error(err);
    }
  };
  reader.onerror = () => toast("Failed to read file.", "err");
  reader.readAsText(file);
}

/** Toast brief on-screen cues when key events happen between integer frames. */
function emitEvents(idx) {
  const traj = state.traj;
  if (!traj || idx < 1) return;
  const cur = traj.frameAt(idx);
  const prev = traj.frameAt(idx - 1);
  if (!cur || !prev || !cur.ent || !prev.ent) return;
  if (prev.phase === "prep" && cur.phase === "main") toast("Seekers released!", "ev");
  const pm = new Map(prev.ent.map((e) => [e.id, e]));
  for (const e of cur.ent) {
    const p = pm.get(e.id);
    if (!p) continue;
    const meta = traj.staticOf(e.id);
    if (p.a && !e.a) {
      if (meta && meta.type === "door") toast("Door opened", "ev");
      else if (meta && meta.type === "wall") toast("Wall broken!", "ev");
    }
    if (!p.dc && e.dc) toast("Decoy activated", "ev");
    if (!p.sn && e.sn) toast("Hider spotted!", "ev");
  }
}

// ============================================================================
// [L0] Camera intro -- establishing shot easing into the framed 3/4 view
// ============================================================================

// The "home" framed pose (matches the initial camera.position / target).
const HOME_CAM = new THREE.Vector3(10.5, 7.8, 13);
const HOME_TARGET = new THREE.Vector3(0, 0.9, 0);
// The wider/higher establishing pose the intro eases FROM.
const INTRO_CAM = new THREE.Vector3(20, 17, 25);

const cameraIntro = {
  active: false,
  pending: false,    // queued (e.g. booted into a non-Watch tab) until Watch shows
  done: false,       // the intro has already played once this session
  start: 0,
  dur: 1200,
  from: new THREE.Vector3(),
  to: new THREE.Vector3(),
  fromTarget: new THREE.Vector3(),
  toTarget: new THREE.Vector3(),
};

/**
 * Kick off the camera establishing-shot -> framed-view ease. Runs once per
 * session, and only while the Watch tab is visible; if requested off-Watch it
 * is queued (cameraIntro.pending) and fires when Watch is shown.
 */
function startCameraIntro() {
  if (cameraIntro.done) return;
  if (state.tab !== "watch") { cameraIntro.pending = true; return; }
  cameraIntro.pending = false;
  cameraIntro.done = true;
  if (reducedMotion) {
    // Snap straight to the framed view; no animation.
    camera.position.copy(HOME_CAM);
    controls.target.copy(HOME_TARGET);
    controls.enabled = true;
    controls.update();
    return;
  }
  cameraIntro.active = true;
  state.introActive = true;
  cameraIntro.start = performance.now();
  cameraIntro.from.copy(INTRO_CAM);
  cameraIntro.to.copy(HOME_CAM);
  cameraIntro.fromTarget.set(0, 1.6, 0);
  cameraIntro.toTarget.copy(HOME_TARGET);
  controls.enabled = false; // hand control back to the user when the ease ends
}

/** Per-frame camera intro update; returns true while the intro is running. */
function updateCameraIntro() {
  if (!cameraIntro.active) return false;
  const t = Math.min(1, (performance.now() - cameraIntro.start) / cameraIntro.dur);
  const e = easeOutCubic(t);
  camera.position.lerpVectors(cameraIntro.from, cameraIntro.to, e);
  controls.target.lerpVectors(cameraIntro.fromTarget, cameraIntro.toTarget, e);
  camera.lookAt(controls.target);
  if (t >= 1) {
    cameraIntro.active = false;
    state.introActive = false;
    controls.enabled = true;
  }
  return true;
}

// ============================================================================
// [L] Main animation loop
// ============================================================================

const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dtWall = clock.getDelta();
  const timeSec = clock.elapsedTime;

  // While the Learning/About tabs are open the 3D canvas is hidden; skip the
  // heavy per-frame scene work (but keep the rAF loop alive so returning to
  // Watch resumes instantly).
  const watching = state.tab === "watch";
  if (!watching) {
    // Nothing visible to draw; yield the frame.
    return;
  }

  // Advance playback position by real time, paced by meta.dt and speed.
  if (state.traj && state.playing) {
    const framesPerSec = (1 / state.traj.dt) * state.speed;
    state.pos += dtWall * framesPerSec;
    if (state.pos >= state.traj.nFrames - 1) {
      if (state.loop) state.pos = 0;
      else { state.pos = state.traj.nFrames - 1; if (state.playing) togglePlay(); }
    }
  }

  // On-screen event cues at integer-frame transitions during forward playback.
  if (state.traj) {
    const fi = Math.round(state.pos);
    if (fi !== state.lastEvFrame) {
      if (state.lastEvFrame >= 0 && fi === state.lastEvFrame + 1) emitEvents(fi);
      state.lastEvFrame = fi;
    }
  }

  if (state.traj) {
    const f = state.traj.sample(state.pos);
    applyFrame(f, timeSec);
    refreshHUD(f);

    // Follow-camera: keep the controls target on the followed/hovered agent.
    // (Suppressed while the camera intro is still flying in.)
    if (state.followCam && !cameraIntro.active) {
      const tid = state.followTargetId >= 0 ? state.followTargetId
                : state.hoverId >= 0 ? state.hoverId : -1;
      const viz = tid >= 0 ? state.vizById.get(tid) : null;
      if (viz && viz.group.visible) {
        controls.target.lerp(viz.group.position, 0.12);
      }
    }
  }

  // Camera intro takes precedence over OrbitControls until it completes.
  if (!updateCameraIntro()) controls.update();
  renderer.render(scene, camera);
}

// ============================================================================
// [M] Resize handling
// ============================================================================

function onResize() {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
}
window.addEventListener("resize", onResize);

// ============================================================================
// [N] Boot
// ============================================================================

async function boot() {
  try {
    buildHUD();

    // Apply the persisted (default dark) theme: this seeds the DOM CSS vars +
    // scene materials/lights and builds the arena/foam in the right palette.
    applyTheme(readStoredTheme(), { persist: false, rebuild: false });
    buildArena(state.traj ? state.traj.bound : 6); // default arena until a traj loads
    gridGroup.visible = state.showGrid;

    // UI enter animation: panels fade/slide in on boot.
    document.body.classList.add("booting");
    requestAnimationFrame(() => {
      requestAnimationFrame(() => document.body.classList.remove("booting"));
    });

    animate();

    // Decide the initial tab + scenario from the hash BEFORE loading scenarios
    // so auto-play only kicks in on Watch.
    const route = parseHash();
    state.tab = route.tab || "watch";

    const ok = await loadScenarios();
    if (!ok) await loadDefault();

    // Now that a scenario is loaded, formally switch to the routed tab (this
    // shows the right panels, lazily builds Learning if needed, and writes the
    // canonical hash via replaceState).
    switchTab(state.tab, { updateHash: true });

    // Camera establishing-shot -> framed view (skipped under reduced motion).
    startCameraIntro();

    // React to manual hash edits / back-forward navigation.
    window.addEventListener("hashchange", onHashChange);

    window.__hns2Booted = true; // cancel the index.html watchdog
  } catch (err) {
    showBootError("The viewer failed to initialize: " + (err.message || err));
    throw err;
  }
}

/** Handle external hash changes (user edits the URL, or uses back/forward). */
function onHashChange() {
  const route = parseHash();
  // Scenario change?
  if (route.scenario && _manifest &&
      _manifest.scenarios.some((s) => s.id === route.scenario) &&
      ui.scenarioSelect && ui.scenarioSelect.value !== route.scenario) {
    loadScenarioById(route.scenario, { updateHash: false });
  }
  // Tab change?
  const tab = route.tab || "watch";
  if (tab !== state.tab) switchTab(tab, { updateHash: false });
}

boot();
