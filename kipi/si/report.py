"""Signal Integrity results to files.

* :func:`save_results` / :func:`load_results`  reopen solved channels (and the last Quick
  Scan) without solving again, as JSON and arrays in a zip (:mod:`kipi.core.results_io`).
* :func:`export_touchstone`  a channel's S-parameters at its die ports, for other tools.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Optional

from kipi import __version__
from kipi.core import results_io

RESULTS_FORMAT = 1
RESULTS_KIND = "si"
RESULTS_SUFFIX = ".kipi-si-results"
OPEN_FILTER = f"KiPI Signal Integrity results (*{RESULTS_SUFFIX})"


def save_results(path, results: Dict[str, object], scan, board_name: str, setup) -> Path:
    """Write every solved channel's :class:`~kipi.si.solver.ChannelResult` (keyed by
    channel name) and the Quick Scan's :class:`~kipi.si.scan.ScanResult` (or None)."""
    header = {"format": RESULTS_FORMAT, "kind": RESULTS_KIND, "kipi": __version__,
              "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "board": board_name,
              "setup": setup.to_dict() if setup is not None else None}
    return results_io.write(path, header, {"results": dict(results), "scan": scan})


def load_results(path) -> dict:
    """``{"results", "scan", "setup", "board", "saved", ...}`` from a results file."""
    if results_io.is_legacy(path):
        raise ValueError("Not a Signal Integrity results file")
    payload = results_io.read(path)
    if payload.get("format", 0) > RESULTS_FORMAT:
        raise ValueError("These results are from a newer KiPI")
    kind = payload.get("kind")
    if kind != RESULTS_KIND:
        mode = results_io.mode_name(kind)
        raise ValueError(f"This holds {mode} results: open it in {mode} mode")
    from kipi.dc.pi.model import PISetup
    data = payload.pop("data")
    payload["results"] = data.get("results", {})
    payload["scan"] = data.get("scan")
    payload["setup"] = PISetup.from_dict(payload["setup"]) if payload.get("setup") else None
    for result in payload["results"].values():
        report = getattr(result, "report_path", "")
        if report and not Path(report).is_file():
            result.report_path = ""
    return payload


def spice_name(stem: str, channel: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", channel).strip("_") or "channel"
    return f"{stem}-{safe}.lib"


def export_spice(path, result) -> tuple:
    """Fit ``result``'s S-parameters with a passive rational model and write it as a SPICE
    subcircuit to ``path``; returns (the path, the model's notes)."""
    from kipi.si.vectfit import fit_sparams, spice_subckt
    model = fit_sparams(result.freqs, result.s)
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", result.channel.name).strip("_") or "channel"
    text = spice_subckt(model, f"kipi_{safe}", result.port_names, 50.0,
                        f"KiPI {__version__}: channel {result.channel.name}\n"
                        + "\n".join(model.notes))
    path = Path(path)
    path.write_text(text, encoding="utf-8")
    return path, model.notes


def touchstone_name(stem: str, channel: str, ports: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", channel).strip("_") or "channel"
    return f"{stem}-{safe}.s{ports}p"


def export_touchstone(path, result, comment: Optional[str] = None) -> Path:
    """Write ``result``'s single-ended S-parameters (50 Ω, at its die ports) to ``path``."""
    from kipi.si.sparams import write_touchstone
    return write_touchstone(path, result.freqs, result.s, 50.0, result.port_names,
                            comment or f"KiPI {__version__}: channel {result.channel.name}")
