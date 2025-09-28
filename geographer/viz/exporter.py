"""Export utilities for GeoGrapher figures and computation payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import plotly.io as pio

from geographer.core.utils import ComputationResult


def export_figure(figure, path: str, format: str = "png") -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format = format.lower()
    if format not in {"png", "svg", "pdf"}:
        raise ValueError("Unsupported export format")
    pio.write_image(figure, str(output_path), format=format)
    return str(output_path)


def export_json(result: ComputationResult, path: str) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)
    return str(output_path)


def export_payload(payload: Dict[str, Any], path: str) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(output_path)
