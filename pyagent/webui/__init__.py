"""Packaged browser UI assets for ``pyagent serve``."""

from pathlib import Path


DIST_DIR = Path(__file__).resolve().parent / "dist"
INDEX_FILE = DIST_DIR / "index.html"
ASSETS_DIR = DIST_DIR / "assets"
