"""
tests/ -- pytest sanity suite for Hide & Seek 2.0.

These tests are intentionally lightweight: they assert the *contract* (derived
dimensions, pytree layout conventions, ELO math, visibility geometry, and an
end-to-end environment smoke step) rather than learning behavior.

Run from the repository root so the absolute imports resolve::

    PYTHONPATH=. pytest            # or: make test

The configuration test (:mod:`tests.test_config`) needs **no third-party
dependencies** -- it imports only :mod:`config`, which is pure Python. Every
other test guards its JAX requirement with
``jax = pytest.importorskip("jax")`` at module top, so collecting the suite
without JAX installed *skips* those tests cleanly instead of erroring.
"""
