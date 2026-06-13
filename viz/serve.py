"""
viz/serve.py -- a tiny static file server for the Hide & Seek 2.0 3D viewer.

The viewer (``viz/web/index.html``) loads ES modules and uses an importmap to
resolve Three.js. Browsers refuse to evaluate ES modules from the ``file://``
scheme (CORS / module-loading restrictions), so opening ``index.html`` directly
will fail with an opaque module error. This server exists purely to serve the
``viz/web`` directory over plain ``http://localhost:PORT/`` so the viewer loads.

Run it::

    python -m viz.serve                 # serves viz/web on :8000
    python -m viz.serve --port 9000     # custom port
    python -m viz.serve --dir viz/web   # custom directory

It is a thin wrapper over :mod:`http.server` -- no third-party dependencies,
nothing fancy. Press Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
import sys
from typing import List, Optional


# Default directory: the sibling ``web/`` folder next to this module.
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with a couple of viewer-friendly tweaks.

    * Adds a permissive ``Access-Control-Allow-Origin`` header so a trajectory
      fetched by the viewer is never blocked by an over-eager browser.
    * Ensures ``.js``/``.mjs``/``.json`` are served with sensible MIME types
      (older Python builds occasionally mis-type ``.mjs``).
    """

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".css": "text/css",
        ".svg": "image/svg+xml",
        ".wasm": "application/wasm",
    }

    def end_headers(self) -> None:  # noqa: D102 (inherited semantics)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def serve(directory: str, port: int) -> int:
    """Serve ``directory`` over HTTP on ``port`` until interrupted.

    Parameters
    ----------
    directory:
        Folder to serve as the web root (the viewer's ``web/`` dir).
    port:
        TCP port to bind on localhost.

    Returns
    -------
    Process exit code (0 on clean Ctrl-C shutdown, non-zero on bind failure).
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        print(f"error: directory does not exist: {directory}", file=sys.stderr)
        return 2

    # Bind the handler to the chosen directory (Python 3.7+ supports `directory=`).
    handler = functools.partial(_QuietHandler, directory=directory)

    # Allow quick rebinds during development (avoids "address already in use").
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.TCPServer(("", port), handler)
    except OSError as exc:
        print(f"error: could not bind port {port}: {exc}", file=sys.stderr)
        print("       try a different --port.", file=sys.stderr)
        return 2

    url = f"http://localhost:{port}/"
    index = os.path.join(directory, "index.html")
    print("=" * 64)
    print("Hide & Seek 2.0 -- 3D viewer server")
    print("-" * 64)
    print(f"  serving : {directory}")
    print(f"  open    : {url}")
    if not os.path.exists(index):
        print(f"  note    : no index.html found in {directory} yet "
              "(the web viewer module owns it).")
    print()
    print("  IMPORTANT: the viewer MUST be loaded over http:// (this URL), not")
    print("  by double-clicking index.html. Browsers block ES modules and")
    print("  importmaps under the file:// scheme, so file:// loads will fail.")
    print("=" * 64)
    print("Press Ctrl-C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        httpd.server_close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: parse args and start the server.

    Parameters
    ----------
    argv:
        Optional argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m viz.serve",
        description="Serve the Hide & Seek 2.0 3D viewer over http:// "
                    "(required for ES module / importmap loading).",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="port to serve on (default 8000)",
    )
    parser.add_argument(
        "--dir", default=DEFAULT_DIR,
        help="directory to serve (default: viz/web next to this module)",
    )
    args = parser.parse_args(argv)
    return serve(args.dir, args.port)


if __name__ == "__main__":
    sys.exit(main())
