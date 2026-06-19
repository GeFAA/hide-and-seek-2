/**
 * trajectory.js -- Parser / validator / sampler for the **hns2-traj** format.
 *
 * This module is the JavaScript mirror of `viz/schema.py`. It knows nothing about
 * Three.js or the DOM; its only job is to take a parsed JSON document, prove that
 * it conforms to the trajectory contract, and then hand the viewer a clean,
 * ergonomic interface for reading frames -- including smooth LINEAR INTERPOLATION
 * between integer frames so playback does not look like a flip-book.
 *
 * The contract (see viz/schema.py for the authoritative version):
 *
 *   { format:"hns2-traj", version:1,
 *     meta:   { arena_size, dt, max_steps, prep_steps, entity_types[], ... },
 *     entities:[ { id, type, team, size, mass, is_decoy } ... ]   // static, id-indexed
 *     frames:  [ { t, phase, sh, ss, seen_any, fog:[[x,y,r]...], ent:[ ... ] } ... ]
 *   }
 *
 * Per-frame entity keys (FRAME_ENT_KEYS):
 *   id, x, y, z(elevation), h(heading rad), a(active), lk(locked), hd(held),
 *   hb(held_by id / -1), no(noise 0..1), dc(decoy active), gr(grounded),
 *   st(stamina 0..1 / -1), sn(seen by opposing team)
 *
 * All numeric / boolean fields are stored as-is; interpolation only touches the
 * continuous channels (x, y, z, h) -- everything boolean is step-held to the
 * floor frame so a flag never flickers "half on".
 */

// ----------------------------------------------------------------------------
// Contract constants (kept in lock-step with viz/schema.py / config.py).
// ----------------------------------------------------------------------------

export const FORMAT = "hns2-traj";
export const VERSION = 1;

/** Canonical entity taxonomy -- order defines the one-hot / type index. */
export const ENTITY_TYPES = [
  "hider", "seeker", "box_light", "box_heavy",
  "ramp", "decoy", "wall", "door",
];

/** Per-frame dynamic entity keys, in documentation order. */
export const FRAME_ENT_KEYS = [
  "id", "x", "y", "z", "h", "a", "lk", "hd", "hb", "no", "dc", "gr", "st", "sn",
];

// Team ids.
export const TEAM_HIDER = 0;
export const TEAM_SEEKER = 1;
export const TEAM_NONE = -1;

// ----------------------------------------------------------------------------
// Palette -- the bright, clean OpenAI "Emergent Tool Use" hide-and-seek look,
// shared with style.css and the legend swatches. Stored as hex numbers so they
// drop straight into THREE.Color(...). Light, low-contrast, friendly, slightly
// desaturated: red/coral seeker, blue/cyan hiders, warm yellow boxes, white
// chunky arena & foam-city backdrop.
// ----------------------------------------------------------------------------

export const COLORS = {
  bg:        0xe7edf3,   // soft light studio background / fog tint
  ground:    0xf3f5f8,   // bright tiled floor
  grid:      0xd6dde4,   // thin light-grey tile lines
  hider:     0x2f9be8,   // friendly blue/cyan hider
  seeker:    0xf2604d,   // warm red/coral seeker
  box_light: 0xf2b441,   // warm yellow light box
  box_heavy: 0xe89a2b,   // deeper amber heavy box (needs cooperation)
  ramp:      0xb9a98f,   // light warm/neutral ramp wedge
  decoy:     0x8b5cf6,   // soft glowing violet decoy
  wall:      0xedf0f4,   // thick chunky white walls
  door:      0xa9c6e2,   // cool light door tint
  fog:       0xcfd8e2,   // soft light fog discs
  spotted:   0xf59e2e,   // warm amber/orange "spotted" glow
  text:      0x1f2733,   // dark slate UI text
  muted:     0x6c7785,   // muted UI text
};

/**
 * Resolve the display color for an entity, honoring team for agents.
 * Agents (hider/seeker) take their team color; props take their type color.
 *
 * @param {{type:string, team:number}} meta - static entity metadata
 * @returns {number} a 0xRRGGBB hex color
 */
export function entityColor(meta) {
  if (meta.type === "hider") return COLORS.hider;
  if (meta.type === "seeker") return COLORS.seeker;
  // Non-agent props are colored by type; team is -1 and irrelevant.
  if (meta.type in COLORS) return COLORS[meta.type];
  return COLORS.muted;
}

// ----------------------------------------------------------------------------
// Small math helpers.
// ----------------------------------------------------------------------------

/** Standard linear interpolation. */
function lerp(a, b, t) {
  return a + (b - a) * t;
}

/**
 * Shortest-arc angular interpolation. Heading is in radians and wraps at 2*pi;
 * naive lerp would spin the long way round when crossing the +/-pi seam, so we
 * fold the delta into [-pi, pi] before blending.
 */
function lerpAngle(a, b, t) {
  let d = (b - a) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
}

// ----------------------------------------------------------------------------
// Validation -- a JS port of validate_trajectory() in schema.py. Returns a list
// of human-readable problem strings; an empty list means the document is valid.
// We keep this cheap and structural: enough to catch the mistakes that silently
// break the viewer (wrong format, id misalignment, missing keys, length drift).
// ----------------------------------------------------------------------------

/**
 * Structurally validate a parsed hns2-traj document.
 * @param {any} doc
 * @returns {string[]} problems (empty == valid)
 */
export function validateTrajectory(doc) {
  const p = [];
  if (doc === null || typeof doc !== "object" || Array.isArray(doc)) {
    return ["top-level value is not an object"];
  }
  if (doc.format !== FORMAT) {
    p.push(`format must be "${FORMAT}", got ${JSON.stringify(doc.format)}`);
  }
  if (doc.version !== VERSION) {
    p.push(`version must be ${VERSION}, got ${JSON.stringify(doc.version)}`);
  }

  const meta = doc.meta || {};
  for (const k of ["arena_size", "dt", "max_steps", "prep_steps",
                   "entity_types", "max_agents", "max_entities"]) {
    if (!(k in meta)) p.push(`meta.${k} missing`);
  }
  if (Array.isArray(meta.entity_types) &&
      meta.entity_types.join(",") !== ENTITY_TYPES.join(",")) {
    p.push("meta.entity_types does not match the canonical ENTITY_TYPES order");
  }

  const entities = doc.entities;
  if (!Array.isArray(entities) || entities.length === 0) {
    p.push("entities must be a non-empty list");
    return p; // can't validate frames against entities without them
  }
  const E = entities.length;
  for (let i = 0; i < E; i++) {
    const e = entities[i];
    if (e.id !== i) p.push(`entities[${i}].id must equal its index (${i}), got ${JSON.stringify(e.id)}`);
    if (!ENTITY_TYPES.includes(e.type)) p.push(`entities[${i}].type invalid: ${JSON.stringify(e.type)}`);
  }

  const frames = doc.frames;
  if (!Array.isArray(frames) || frames.length === 0) {
    p.push("frames must be a non-empty list");
    return p;
  }
  for (let fi = 0; fi < frames.length; fi++) {
    const fr = frames[fi];
    if (fr.phase !== "prep" && fr.phase !== "main") {
      p.push(`frames[${fi}].phase invalid: ${JSON.stringify(fr.phase)}`);
    }
    const ent = fr.ent;
    if (!Array.isArray(ent) || ent.length !== E) {
      p.push(`frames[${fi}].ent must have length ${E} (got ${Array.isArray(ent) ? ent.length : "n/a"})`);
      continue;
    }
    for (let ei = 0; ei < ent.length; ei++) {
      const en = ent[ei];
      if (en.id !== ei) {
        p.push(`frames[${fi}].ent[${ei}].id misaligned (expected ${ei}, got ${JSON.stringify(en.id)})`);
      }
      const missing = FRAME_ENT_KEYS.filter((k) => !(k in en));
      if (missing.length) {
        p.push(`frames[${fi}].ent[${ei}] missing keys: ${missing.join(", ")}`);
        break; // one report per entity is plenty
      }
    }
  }
  return p;
}

// ----------------------------------------------------------------------------
// Trajectory -- the high-level, validated wrapper the viewer actually uses.
// ----------------------------------------------------------------------------

/**
 * A loaded, validated trajectory with convenient frame access + interpolation.
 *
 * Construct via the static `Trajectory.parse(doc)` factory, which throws a
 * descriptive Error if the document does not conform to the contract.
 */
export class Trajectory {
  /**
   * @param {object} doc - a validated hns2-traj document
   */
  constructor(doc) {
    this.doc = doc;
    this.meta = doc.meta;
    this.entities = doc.entities;       // static per-slot metadata (length E)
    this.frames = doc.frames;           // dynamic frames (length n_frames)
    this.E = doc.entities.length;       // entity count
    this.nFrames = doc.frames.length;   // frame count

    // Convenience metadata with sane fallbacks.
    this.arenaSize = Number(this.meta.arena_size) || 12.0;
    this.dt = Number(this.meta.dt) || 0.1;
    this.maxSteps = Number(this.meta.max_steps) || (this.nFrames - 1);
    this.prepSteps = Number(this.meta.prep_steps) || 0;
    this.title = this.meta.title || "Hide & Seek 2.0";
    this.outcome = doc.outcome || null;   // {winner, reason, step} (who wins)

    // id -> static metadata map (id == index, but the map keeps intent explicit).
    /** @type {Map<number, object>} */
    this.metaById = new Map();
    for (const e of this.entities) this.metaById.set(e.id, e);

    // Pre-compute which slots are ever active. Padded / never-spawned slots
    // (a==0 in every frame) are hidden entirely by the viewer.
    /** @type {boolean[]} */
    this.everActive = new Array(this.E).fill(false);
    for (const fr of this.frames) {
      for (let i = 0; i < this.E; i++) {
        if (fr.ent[i].a) this.everActive[i] = true;
      }
    }
  }

  /**
   * Parse + validate a raw JSON object into a Trajectory.
   * @param {any} doc
   * @returns {Trajectory}
   * @throws {Error} with all collected validation problems if invalid
   */
  static parse(doc) {
    const problems = validateTrajectory(doc);
    if (problems.length) {
      throw new Error("Invalid hns2-traj file:\n  - " + problems.join("\n  - "));
    }
    return new Trajectory(doc);
  }

  /** Arena half-extent: world spans [-bound, +bound] in x and y. */
  get bound() {
    return this.arenaSize / 2;
  }

  /** Static metadata for an entity id (or undefined). */
  staticOf(id) {
    return this.metaById.get(id);
  }

  /** The integer-frame index of the prep/main phase boundary (for the scrubber tick). */
  get phaseBoundaryFrame() {
    // Find the first frame whose phase is "main"; fall back to prepSteps.
    for (let i = 0; i < this.nFrames; i++) {
      if (this.frames[i].phase === "main") return i;
    }
    return Math.min(this.prepSteps, this.nFrames - 1);
  }

  /**
   * Sample the trajectory at a (possibly fractional) frame position.
   *
   * Continuous channels (x, y, z, h) are linearly interpolated between the two
   * bracketing integer frames -- heading via shortest-arc -- so motion is smooth.
   * Boolean / discrete channels (a, lk, hd, hb, dc, gr, sn, no, st) are
   * step-held to the FLOOR frame, because a half-open door or a half-spotted
   * agent is meaningless and would flicker.
   *
   * @param {number} pos - fractional frame index in [0, nFrames-1]
   * @returns {{
   *   t:number, phase:string, sh:number, ss:number, seen_any:boolean,
   *   fog:number[][], ent: Array<object>, frameIndex:number, alpha:number
   * }}
   */
  sample(pos) {
    const maxIdx = this.nFrames - 1;
    const clamped = Math.max(0, Math.min(maxIdx, pos));
    const i0 = Math.floor(clamped);
    const i1 = Math.min(maxIdx, i0 + 1);
    const alpha = clamped - i0;

    const f0 = this.frames[i0];
    const f1 = this.frames[i1];

    // Per-entity interpolation.
    const ent = new Array(this.E);
    for (let i = 0; i < this.E; i++) {
      const a0 = f0.ent[i];
      const a1 = f1.ent[i];
      // If the entity is inactive in the floor frame, do not interpolate it
      // toward an active future pose (it would "teleport in"); just step-hold.
      const blend = a0.a && a1.a ? alpha : 0;
      ent[i] = {
        id: a0.id,
        x: lerp(a0.x, a1.x, blend),
        y: lerp(a0.y, a1.y, blend),
        z: lerp(a0.z, a1.z, blend),
        h: lerpAngle(a0.h, a1.h, blend),
        // Step-held discrete / boolean channels (floor frame is authoritative).
        a: a0.a,
        lk: a0.lk,
        hd: a0.hd,
        hb: a0.hb,
        no: a0.no,
        dc: a0.dc,
        gr: a0.gr,
        st: a0.st,
        sn: a0.sn,
      };
    }

    return {
      t: f0.t,
      phase: f0.phase,
      sh: lerp(f0.sh, f1.sh, alpha),
      ss: lerp(f0.ss, f1.ss, alpha),
      seen_any: f0.seen_any,
      fog: f0.fog,            // fog patches step-held (their own gentle anim lives in the viewer)
      ent,
      frameIndex: i0,
      alpha,
    };
  }

  /** Direct (non-interpolated) access to an integer frame. */
  frameAt(i) {
    return this.frames[Math.max(0, Math.min(this.nFrames - 1, i | 0))];
  }
}

export default Trajectory;
