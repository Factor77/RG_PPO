"""Pytest bootstrap for Tail_GAE unit tests.

Injects the Tail_GAE src directory (flat-import universe) and the project
root (package-style universe) into sys.path so that tests can import both
``from risk_oracle import ...`` and ``from scripts.Tail_GAE.src...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[3]

for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
