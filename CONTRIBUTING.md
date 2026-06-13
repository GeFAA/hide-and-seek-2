# Contributing to Hide & Seek 2.0

Thanks for taking an interest! This is a small, friendly research project and
contributions of all sizes are welcome — bug reports, doc fixes, new viewer
scenarios, or core improvements.

## Heads up: research scaffold

Hide & Seek 2.0 is a **research scaffold**. It is engineered to be correct,
importable, and faithful to current JAX idioms — but the JAX training paths
(`config.py`, `train.py`, `envs/`, `models/`, `trainers/`, `utils/`) are
intended to run on a **GPU/accelerator env**. CPU works for imports, the test
suite, and the viewer; full-scale training (`num_envs = 2048`) is not. The 3D
viewer (`viz/`) is fully decoupled and needs **no JAX at all**.

## Dev setup

From the repository root (Python ≥ 3.10):

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The editable `[dev]` install pulls in `pytest`, `ruff`, and `black`.

## Run the tests

Tests live in `tests/`. Run them from the repository root with `PYTHONPATH=.`
so the absolute imports (e.g. `from config import Config`) resolve:

```bash
PYTHONPATH=. pytest
```

```powershell
# Windows PowerShell
$env:PYTHONPATH = "."
pytest
```

`tests/test_config.py` imports only the pure-Python `config` module and runs
without JAX — it is the same test CI runs on every push.

## Run the viewer

The viewer is browser-based, needs no build step, and needs no JAX:

```bash
# 1. Generate a demo trajectory (pure stdlib — no jax/numpy needed)
python viz/make_demo_trajectory.py
# 2. Serve viz/web on http://localhost:8000 (stdlib server, no deps)
python -m viz.serve
# 3. Open http://localhost:8000  (must be http://, not file://)
```

Or just try the hosted build: <https://gefaa.github.io/hide-and-seek-2/>.

## Code style

- Format with **black** and lint with **ruff**: `black . && ruff check --fix .`
  (or `make format`).
- **Line length is 88** for both tools — see `[tool.black]` and `[tool.ruff]`
  in [`pyproject.toml`](pyproject.toml), the single source of truth for style.
- Keep the `viz/` viewer and helper scripts **stdlib-only** where they already
  are: `make_demo_trajectory.py`, `schema.py`, and `serve.py` must not import
  jax or numpy (only `recorder.py` may, since it handles real rollouts).

## Add a new viewer scenario

Scenarios are self-contained trajectory JSON files that the viewer plays back;
adding one needs no changes to the training code. Two steps:

1. **Add a builder** in [`viz/make_demo_trajectory.py`](viz/make_demo_trajectory.py)
   that constructs the episode and writes a schema-valid trajectory (see
   [`viz/schema.py`](viz/schema.py) for the contract) into
   `viz/web/trajectories/<your_scenario>.json`.
2. **Register it in the manifest** — add an entry (name + one-line description +
   file) to `viz/web/trajectories/manifest.json` so the in-viewer scenario
   picker lists it.

Keep scenarios short and legible: each should isolate one behaviour (a fort
going up, a ramp being used, a door being blocked) so it reads clearly in a few
seconds. See [`viz/README.md`](viz/README.md) for the trajectory format,
controls, and palette legend.

## Pull requests

- Branch from `main`, keep PRs focused, and describe the change.
- Run `PYTHONPATH=. pytest` and `ruff check .` before opening a PR.
- Update [`CHANGELOG.md`](CHANGELOG.md) under a suitable heading for
  user-facing changes.

Happy hacking!
