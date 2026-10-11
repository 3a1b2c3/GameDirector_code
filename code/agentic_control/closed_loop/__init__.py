"""Shared closed-loop agent for NRFTW, VRising, and Hollow Knight death data."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../agentic_control
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .loop import ClosedLoopAgent

__all__ = ["ClosedLoopAgent"]
