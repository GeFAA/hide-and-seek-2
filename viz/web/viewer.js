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

import { Trajectory, COLORS, ENTITY_TYPES } from "./trajectory.js";

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
// [B] Constants & palette helpers
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
renderer.toneMappingExposure = 1.12;                      // keep whites bright (not crushed)
renderer.outputColorSpace = THREE.SRGBColorSpace;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();

/**
 * Build a soft vertical-gradient sky texture for the scene background:
 * #dce4ec (top) -> #eef2f6 (bottom). Drawn once into a CanvasTexture.
 */
function makeBackgroundTexture() {
  const c = document.createElement("canvas");
  c.width = 16;
  c.height = 256;
  const ctx = c.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, 0, c.height);
  g.addColorStop(0, "#dce4ec");
  g.addColorStop(1, "#eef2f6");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, c.width, c.height);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}
scene.background = makeBackgroundTexture();
// Gentle light fog so the far "foam city" cubes fade softly into the backdrop.
scene.fog = new THREE.Fog(COLORS.bg, 34, 120);

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
const ambient = new THREE.AmbientLight(0xffffff, 0.45);
scene.add(ambient);

// Hemisphere bounce: white sky, LIGHT ground term so downward-facing / away
// faces stay light. Does most of the soft fill.
const hemi = new THREE.HemisphereLight(0xffffff, 0xdfe6ee, 1.1);
scene.add(hemi);

// Warm key directional light, high and angled, casting SOFT shadows. Kept
// gentle (~1.0) so the lit/unlit contrast stays low and whites read as white.
const keyLight = new THREE.DirectionalLight(0xfff4e6, 1.0);
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
const fillLight = new THREE.DirectionalLight(0xeaf1f8, 0.35);
fillLight.position.set(-14, 9, -16);
scene.add(fillLight);

// Ground plane (large, bright). Lies in the XZ plane; world y is "up".
// IMPORTANT mapping: the trajectory's (x, y) are floor coordinates and z is
// elevation. We map traj.x -> three.x, traj.y -> three.z, traj.z -> three.y.
const groundMat = new THREE.MeshStandardMaterial({
  color: col(COLORS.ground),
  roughness: 0.95,
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
  gridGroup.clear();

  // Fine 1-unit tile grid spanning the arena, in the light tile-line color.
  const span = Math.ceil(bound) * 2;
  const grid = new THREE.GridHelper(span, span, col(COLORS.grid), col(COLORS.grid));
  grid.position.y = 0.002;
  grid.material.opacity = 0.6;
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
      color: col(0xbac4cf),
      transparent: true,
      opacity: 0.5,
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
  // SINGLE uniform soft-white material -- exactly the OpenAI reference look.
  // No per-instance coloring: an unset/black instanceColor (or vertexColors
  // with no per-vertex color attribute) multiplies the lit result to near-
  // black, which is the "black skyline" bug. A uniform light MeshStandard,
  // lit by the bright ambient + hemisphere fill, keeps every cube clean white.
  const mat = new THREE.MeshStandardMaterial({
    color: 0xeef1f5,
    roughness: 0.95,
    metalness: 0.0,
  });
  const mesh = new THREE.InstancedMesh(geo, mat, COUNT);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);

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
    // Keep one rng() draw here so the deterministic placement sequence (and
    // thus the stable cube field across reloads) is unchanged after dropping
    // the per-instance tint. The value is intentionally discarded.
    rng();
    placed++;
  }
  mesh.count = placed;
  mesh.instanceMatrix.needsUpdate = true;
  backdropGroup.add(mesh);
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
};
scene.add(state.entityRoot);
scene.add(state.linkRoot);
scene.add(state.fogRoot);
scene.add(state.trailRoot);

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
      emissiveIntensity: 0.12,
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
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    aura = new THREE.Mesh(GEO.auraDisc, auraMat);
    aura.rotation.x = -Math.PI / 2;
    const auraScale = (size / 0.4) * 2.7;
    aura.scale.set(auraScale, auraScale, 1);
    aura.position.y = 0.03;
    group.add(aura);

    // Small heading "beak": a subtle body-tinted nub showing facing (+Z local).
    const noseMat = new THREE.MeshStandardMaterial({
      color: baseColor,
      emissive: baseColor,
      emissiveIntensity: 0.12,
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
        new THREE.LineBasicMaterial({ color: col(0xc98322), transparent: true, opacity: 0.5 })
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
        color: baseColor, emissive: baseColor, emissiveIntensity: 0.9,
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
      color: col(COLORS.spotted), transparent: true, opacity: 0.0,
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

/** Resolve an entity's THREE.Color from its static metadata. */
function entityColorFor(meta) {
  if (meta.type === "hider") return col(COLORS.hider);
  if (meta.type === "seeker") return col(COLORS.seeker);
  if (meta.type in COLORS) return col(COLORS[meta.type]);
  return col(COLORS.muted);
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
  // Clear previous.
  for (const viz of state.vizById.values()) state.entityRoot.remove(viz.group);
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
      if (spotted) viz.body.material.emissive.copy(col(COLORS.spotted));
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
        viz.body.material.emissive.copy(col(COLORS.decoy));
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
        color: col(COLORS.box_heavy), transparent: true, opacity: 0.6,
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
        color: col(COLORS.fog), transparent: true, opacity: 0.1,
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
    m.material.opacity = 0.10 + 0.03 * Math.sin(timeSec * 0.9 + i);
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
    <button class="btn panel-toggle" id="btn-panel" title="Toggle panel">
      <svg viewBox="0 0 24 24"><path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" stroke-width="2" fill="none"/></svg>
    </button>
    <div class="hud-title">
      <span class="t1" id="traj-title">Hide &amp; Seek 2.0</span>
      <span class="t2">Tactical Trajectory Viewer</span>
    </div>
    <div class="pill prep" id="phase-pill"><span class="dot"></span><span id="phase-text">PREP</span></div>
    <div class="hud-spacer"></div>
    <div class="scores glass" style="box-shadow:none;background:transparent;border:none">
      <div class="score hiders"><span class="lbl">Hiders</span><span class="val" id="score-h">0</span></div>
      <div class="score"><span class="vs">vs</span></div>
      <div class="score seekers"><span class="lbl">Seekers</span><span class="val" id="score-s">0</span></div>
    </div>
    <div class="step-counter">t <b id="step-cur">0</b> <span class="max">/ <span id="step-max">0</span></span></div>
    <div class="spotted-ind" id="spotted-ind"><span class="dot"></span>SPOTTED</div>
    <select class="scenario-select" id="scenario-select" title="Choose a scenario"
      style="font:inherit;font-size:12px;color:#1f2733;background:rgba(255,255,255,0.55);border:1px solid rgba(15,30,55,0.14);border-radius:8px;padding:4px 8px;margin-right:8px;cursor:pointer;outline:none;max-width:190px"></select>
    <button class="icon-btn" id="btn-load" title="Load a trajectory file">
      <svg viewBox="0 0 24 24"><path d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16" stroke="currentColor" stroke-width="2" fill="none"/></svg>
      Load
    </button>
  `;
  app.appendChild(top);

  // ---- left control panel ---------------------------------------------
  const left = el("div", "glass overlay", { id: "panel-left" });
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
  const insp = el("div", "glass overlay", { id: "inspector" });
  insp.hidden = true;
  app.appendChild(insp);

  // ---- bottom transport ------------------------------------------------
  const tr = el("div", "glass overlay", { id: "transport" });
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
  const cap = el("div", "", { id: "scenario-caption" });
  cap.innerHTML = `<span class="cap-title" id="cap-title"></span><span class="cap-sub" id="cap-sub"></span>`;
  app.appendChild(cap);

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

  wireHUD();
  buildLegend();
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
    const hex = "#" + (COLORS[key] >>> 0).toString(16).padStart(6, "0");
    // Light theme: a crisp 1px ring instead of a glow so near-white swatches
    // (wall/door) still read against the frosted panel.
    return `<div class="legend-row"><span class="swatch" style="background:${hex};box-shadow:0 0 0 1px rgba(15,30,55,0.12)"></span>${label}</div>`;
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

  // keyboard: space = play/pause, arrows = step. Blur a focused transport
  // button first so Space isn't ALSO delivered as that button's click.
  window.addEventListener("keydown", (e) => {
    const tag = e.target && e.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
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
  ui.scoreH.textContent = Math.round(f.sh);
  ui.scoreS.textContent = Math.round(f.ss);
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

  // Position the phase-boundary tick on the scrubber.
  const bf = traj.phaseBoundaryFrame;
  const pct = traj.nFrames > 1 ? (bf / (traj.nFrames - 1)) * 100 : 0;
  ui.scrubTick.style.left = pct + "%";
  ui.scrubTick.style.display = bf > 0 && bf < traj.nFrames - 1 ? "block" : "none";

  ui.title.textContent = traj.title;
  if (ui.capTitle) ui.capTitle.textContent = traj.title;
  if (ui.capSub) ui.capSub.textContent = "";   // scenario loader sets a description

  // Auto-play on load.
  if (!state.playing) togglePlay();

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
    const hashId = (location.hash.match(/scenario=([\w-]+)/) || [])[1];
    const wanted = hashId && man.scenarios.some((s) => s.id === hashId) ? hashId
                 : (hasDefault ? man.default : man.scenarios[0].id);
    if (sel) sel.value = wanted;
    await loadScenarioById(wanted);
    return true;
  } catch (err) {
    console.warn("No scenario manifest:", err);
    return false;
  }
}

/** Load one scenario by manifest id and set the caption description. */
async function loadScenarioById(id) {
  const entry = _manifest && _manifest.scenarios.find((s) => s.id === id);
  if (!entry) return;
  try {
    const res = await fetch("./trajectories/" + entry.file, { cache: "no-cache" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const doc = await res.json();
    installTrajectory(Trajectory.parse(doc));
    if (ui.scenarioSelect) ui.scenarioSelect.value = id;
    if (ui.capSub) ui.capSub.textContent = entry.description || "";
    try { history.replaceState(null, "", "#scenario=" + id); } catch (e) { /* ignore */ }
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
// [L] Main animation loop
// ============================================================================

const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dtWall = clock.getDelta();
  const timeSec = clock.elapsedTime;

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
    if (state.followCam) {
      const tid = state.followTargetId >= 0 ? state.followTargetId
                : state.hoverId >= 0 ? state.hoverId : -1;
      const viz = tid >= 0 ? state.vizById.get(tid) : null;
      if (viz && viz.group.visible) {
        controls.target.lerp(viz.group.position, 0.12);
      }
    }
  }

  controls.update();
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
    buildArena(6); // default until a trajectory loads (arena_size 12 -> bound 6)
    gridGroup.visible = state.showGrid;
    animate();
    const ok = await loadScenarios();
    if (!ok) await loadDefault();
    window.__hns2Booted = true; // cancel the index.html watchdog
  } catch (err) {
    showBootError("The viewer failed to initialize: " + (err.message || err));
    throw err;
  }
}

boot();
