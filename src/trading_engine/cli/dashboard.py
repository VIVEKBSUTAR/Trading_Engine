"""Launch the Streamlit live intelligence dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Launch Streamlit using the packaged dashboard app."""
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
