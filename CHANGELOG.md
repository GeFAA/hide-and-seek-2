# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.0] -- 2026-06-19

### Changed

- **Professional visual overhaul of the 3D viewer.** Agents are now designed
  little robot characters (rounded body, face with eyes, stubby arms, antenna,
  and a glowing base ring) instead of bare capsules. The scene gained image-based
  lighting (PBR materials + environment reflections), a post-processing pipeline
  (subtle bloom + SMAA, with a safe fallback to direct rendering), a three-point
  light rig with soft shadows, a low-roughness reflective floor, and characterful
  animations (idle breathing / squash-stretch, lean-into-motion, eye blinks, and
  a softly pulsing base glow).

### Added

- **Key art and a proper GitHub presentation.** A rendered hero banner and a
  gallery (character key art + action scenes), plus a refreshed screenshot of the
  actual viewer.

## [1.6.0] -- 2026-06-13

### Added

- **Dark mode (now the default).** The whole viewer — 3D scene and UI — has a
  clean dark theme, with a one-click light/dark toggle (the `T` key) that is
  remembered between visits.
- **Tabs: Watch / Learning / About.** A tab bar (with `#tab=…` deep links)
  switches between the 3D replay, a learning dashboard, and a beginner-friendly
  explainer of the project and its 2.0 mechanics.
- **Learning dashboard.** A newcomer-friendly view of the self-play
  autocurriculum: team skill cards (ELO + win rate), an "arms race" win-rate
  chart, an ELO-over-time chart, and a table of emergent behaviours with the
  training step at which each was learned. Driven by `viz/make_learning_data.py`
  (pure stdlib, illustrative synthetic data).

### Changed

- **Animations reworked** throughout: an easing camera intro, cross-fading tab
  switches, fading/sliding panels, animated number count-ups and progress bars,
  drawn-in charts, and smoother in-scene effects — all respecting
  `prefers-reduced-motion`.

## [1.5.1] -- 2026-06-13

### Fixed

- **Box-carry visuals.** Hiders now visibly *push* a carried box ahead of them
  instead of rendering inside it (the box was previously pinned to the carrier's
  exact position, so the agent disappeared into the crate while hauling). The
  cooperative heavy-box flank was widened so both pushers sit clearly beside the
  box, and the agent ground-glow was softened so it no longer washes out
  capsules behind it.

## [1.5.0] -- 2026-06-13

### Added

- **Named viewer scenarios.** The 3D viewer now ships with a set of curated,
  hand-authored episodes — **Fort Building**, **Ramp Use**, **Running &
  Chasing**, **Door Blocking**, and **Sensory Deception** — each isolating one
  emergent behaviour from the original *Emergent Tool Use* work.
- **In-viewer scenario picker.** Switch between scenarios from a picker in the
  viewer itself, with no reload (you can still load any trajectory file
  manually).
- **Live GitHub Pages demo.** Try the viewer in your browser with no install at
  <https://gefaa.github.io/hide-and-seek-2/>.
- **Continuous integration.** A GitHub Actions workflow byte-compiles every
  package and runs the JAX-free configuration tests on each push and pull
  request.
- **Project docs.** Added this `CHANGELOG.md` and a `CONTRIBUTING.md` guide.
- **Agent faces & on-screen event cues.** Agents now show simple faces, and the
  viewer surfaces on-screen cues as key events happen during an episode.

## [1.0.1] -- 2026-06-13

### Fixed

- **3D viewer overhaul** to a clean, OpenAI-style scene:
  - a clean square arena bounded by correctly oriented border walls;
  - an opaque box lock-emblem, fixing the previously black box faces;
  - soft ground-glow agent halos that replace the old sphere "blob";
  - a subtler heading indicator.

## [1.0.0] -- 2026-06-13

### Added

- **Initial public release.** A JAX/Flax multi-agent RL scaffold — MAPPO with
  Centralized-Training / Decentralized-Execution, an entity Transformer with GRU
  memory, ELO historical self-play, and the Hide & Seek 2.0 mechanics — together
  with the browser-based 3D replay viewer.

[1.5.0]: https://github.com/GeFAA/hide-and-seek-2/releases/tag/v1.5.0
[1.0.1]: https://github.com/GeFAA/hide-and-seek-2/releases/tag/v1.0.1
[1.0.0]: https://github.com/GeFAA/hide-and-seek-2/releases/tag/v1.0.0
