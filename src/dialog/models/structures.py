"""TODO: Add docstring."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeneBackgroundPMFs:
    """Mapping from gene to background mutation rate PMF P(B = k)."""

    mapping: dict[str, list[float]]
