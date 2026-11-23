"""Compatibility wrapper for the packaged ``atomzero-audit`` command.

Use ``atomzero-audit`` (or ``python -m atomzero.audit``) when Atom Zero is
installed. This script keeps the original checkout-local entry point working:

    python scripts/aisc_audit.py --aisc-path "C:\path\to\ai-systems-algebra"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomzero.audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
