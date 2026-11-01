"""Signal Integrity plots drawn the same in the SI plots view and in the report: the solved
channels side by side."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class Ink:
    """A plot's colours: its series, text, faint lines, and PASS / WARN / FAIL."""
    colors: Tuple[str, ...]
    text: str
    faint: str
    status: Tuple[str, str, str] = ("#60d060", "#e0c040", "#ff6060")

    def of(self, status: Optional[str]) -> str:
        return {"PASS": self.status[0], "WARN": self.status[1],
                "FAIL": self.status[2]}.get(status or "", self.faint)


# What each comparison panel shows: its title, the check whose status colours it, and
# (value, limit) from a result.
def _eye_h(r):
    e = r.eye
    return (e.height_v * 1e3 if e is not None else np.nan,
            r.channel.value("eye_h_v") * 1e3 if r.channel.value("eye_h_v") else None)


def _eye_w(r):
    e = r.eye
    return (e.width_ui if e is not None else np.nan, r.channel.value("eye_w_ui"))


def _il(r):
    k = int(np.argmin(np.abs(r.freqs - r.nyquist_hz)))
    return float(r.il_db[k]) if r.il_db is not None else np.nan, r.channel.value("il_max_db")


def _skew(r):
    if r.skew_s is None:
        return np.nan, None
    return 100 * r.skew_s * r.channel.value("rate_bps"), 5.0


COMPARE = (("Eye height\n(mV)", "eye", _eye_h), ("Eye width\n(UI)", "eye", _eye_w),
           ("Insertion loss\nat Nyquist (dB)", "insertion loss", _il),
           ("P/N skew\n(% of a UI)", "skew", _skew))


def draw_compare(fig, results: Sequence, ink: Ink, style=None):
    """The solved ``results`` side by side: a bar each in each panel of :data:`COMPARE`,
    coloured by that check's status, with its limit as a tick. ``style(ax)`` styles each
    panel."""
    fig.clear()
    names = [r.channel.name for r in results]
    y = np.arange(len(results))
    axes = fig.subplots(1, len(COMPARE), sharey=True)
    for ax, (title, check, get) in zip(axes, COMPARE):
        vals, lims, cols = [], [], []
        for r in results:
            v, lim = get(r)
            st = next((c.status for c in r.checks if c.check == check), None)
            vals.append(v)
            lims.append(np.nan if lim is None else lim)
            cols.append(ink.of(st))
        vals = np.array(vals, float)
        ax.barh(y, np.nan_to_num(vals), color=cols, height=0.6)
        lims = np.array(lims, float)
        if np.isfinite(lims).any():
            ax.plot(lims, y, "|", color=ink.text, ms=14, mew=2, label="Limit")
        for k, v in enumerate(vals):
            if np.isfinite(v):
                text = f" {v:.0f} " if abs(v) >= 100 else f" {v:.3g} "
                ax.text(v, k, text, va="center", fontsize=7, color=ink.text,
                        ha="left" if v >= 0 else "right")
            else:
                ax.text(0, k, " none", va="center", fontsize=7, color=ink.faint)
        ax.set_title(title, fontsize=9)
        ax.axvline(0, color=ink.faint, lw=0.8)
        span = np.concatenate([vals[np.isfinite(vals)], lims[np.isfinite(lims)], [0.0]])
        lo, hi = span.min(), span.max()
        pad = 0.25 * (hi - lo or 1.0)
        ax.set_xlim(lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0))
        if style is not None:
            style(ax)
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].invert_yaxis()                      # the first channel at the top
    return axes
