"""An eye diagram on a matplotlib figure, for the SI plots and the report alike.

A statistical eye shows the probability of each voltage at each phase (log scale), its
inner edges at several BERs (the target's thicker), the eye height and width at the
target BER as arrows, the receiver's mask and, under it, the bathtub: the BER of a bit
sampled at the threshold against phase. A waveform eye shows how often a simulated PRBS
waveform passes each point, its traces drawn as lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Theme:
    bg: str
    fg: str
    grid: str
    spine: str
    contour: Tuple[str, ...]          # lightest BER to the target's
    mask: str
    marker: str


DARK = Theme("#1e1e1e", "#dddddd", "#444444", "#888888",
             ("#2f6d8a", "#3a8fb5", "#48b2df", "#7fd4ff", "#b8e9ff"), "#60d060", "#ffffff")
# The eye's own axes are black on either theme: its contours and marks are bright on both.
LIGHT = Theme("#ffffff", "#222222", "#cccccc", "#666666", DARK.contour, "#2ca02c", "#ffffff")

LOG_FLOOR = -12                     # the colour scale's bottom, log10 of probability


def _closed(ph: np.ndarray, edge: np.ndarray, centre: float):
    """The open run of ``edge`` (phases, 2) around ``centre`` as one closed outline (x, y),
    the eye's ends where its edges meet."""
    ok = np.isfinite(edge[:, 0]) & (edge[:, 1] > edge[:, 0])
    if not ok.any():
        return None
    k = int(np.argmin(np.abs(ph - centre)))
    if not ok[k]:
        cand = np.flatnonzero(ok)
        k = int(cand[np.argmin(np.abs(ph[cand] - centre))])
    i0 = i1 = k
    while i0 > 0 and ok[i0 - 1]:
        i0 -= 1
    while i1 < len(ok) - 1 and ok[i1 + 1]:
        i1 += 1
    lo, hi = edge[i0:i1 + 1, 0], edge[i0:i1 + 1, 1]
    x = ph[i0:i1 + 1]
    # Close the ends half a phase step out, at the middle of the last open edges.
    step = (ph[1] - ph[0]) / 2 if len(ph) > 1 else 0.0
    xs = np.concatenate([[x[0] - step], x, [x[-1] + step], x[::-1], [x[0] - step]])
    ys = np.concatenate([[0.5 * (lo[0] + hi[0])], lo, [0.5 * (lo[-1] + hi[-1])], hi[::-1],
                         [0.5 * (lo[0] + hi[0])]])
    return xs, ys


def _style(ax, th: Theme):
    ax.set_facecolor(th.bg)
    for s in ax.spines.values():
        s.set_color(th.spine)
    ax.tick_params(colors=th.fg, which="both", labelsize=8)
    ax.xaxis.label.set_color(th.fg)
    ax.yaxis.label.set_color(th.fg)
    ax.title.set_color(th.fg)


def _legend(ax, th: Theme):
    if ax.get_legend_handles_labels()[0]:
        leg = ax.legend(loc="upper right", fontsize=7, facecolor=th.bg, edgecolor=th.spine,
                        framealpha=0.85)
        for t in leg.get_texts():
            t.set_color(th.fg)


def _arrow(ax, xy0, xy1, text, th: Theme, offset, ha):
    ax.annotate("", xy=xy1, xytext=xy0,
                arrowprops=dict(arrowstyle="<->", color=th.marker, lw=1.0, shrinkA=0,
                                shrinkB=0))
    mx, my = 0.5 * (xy0[0] + xy1[0]), 0.5 * (xy0[1] + xy1[1])
    ax.annotate(text, (mx, my), xytext=offset, textcoords="offset points", ha=ha,
                va="center", fontsize=8, color=th.marker,
                bbox=dict(boxstyle="round,pad=0.2", fc="#000000", ec="none", alpha=0.6))


def _limits(e) -> Tuple[float, float]:
    """The voltage span the eye's density reaches, padded."""
    d = e.density
    seen = np.flatnonzero(d.max(axis=0) > d.max() * 10.0 ** LOG_FLOOR)
    lv = e.levels_v
    lo, hi = (lv[seen[0]], lv[seen[-1]]) if len(seen) else (lv[0], lv[-1])
    pad = 0.12 * (hi - lo + 1e-12)
    return max(lo - pad, lv[0]) * 1e3, min(hi + pad, lv[-1]) * 1e3


def draw_eye(fig, e, mask: Optional[Tuple[float, float]] = None, theme: Theme = DARK,
             bathtub: bool = True, title: str = "") -> None:
    """Draw eye ``e`` (:class:`kipi.si.eye.EyeResult`) on ``fig`` (cleared first).
    ``mask``: (height V, width UI) of the receiver's eye mask, drawn at the eye's middle."""
    fig.clear()
    fig.set_layout_engine(None)
    fig.subplots_adjust(left=0.09, right=0.92, top=0.93 if title else 0.97, bottom=0.14)
    stat = e.kind == "statistical"
    tub = bathtub and stat and e.bathtub is not None
    if tub:
        gs = fig.add_gridspec(2, 2, height_ratios=(3.2, 1), width_ratios=(40, 1), hspace=0.08,
                              wspace=0.03)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        bx = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=(40, 1), wspace=0.03)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        bx = None
    ph, lv = e.phases_ui, e.levels_v
    d = np.asarray(e.density, float)
    if stat:
        # Each column is a probability density: a little smoothing along the levels
        # turns its discrete ISI levels into the bands a scope would show.
        from scipy.ndimage import gaussian_filter1d
        d = gaussian_filter1d(d, 1.0, axis=1)
        z = np.log10(d / max(d.max(), 1e-300) + 1e-300)
        vmin, label = LOG_FLOOR, "log₁₀ probability"
    else:
        from scipy.ndimage import gaussian_filter
        d = gaussian_filter(d, 0.7)          # anti-aliased traces
        z = np.log10(d / max(d.max(), 1e-300) * 1e3 + 1.0)
        z = np.where(d > 1e-3 * d.max(), z, np.nan)
        vmin, label = 0.0, "hits (log scale)"
    import matplotlib
    cmap = matplotlib.colormaps["inferno"].with_extremes(bad="#000000")
    img = ax.imshow(z.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin,
                    vmax=max(np.nanmax(z) if np.isfinite(z).any() else 1.0, vmin + 1),
                    extent=(ph[0], ph[-1], lv[0] * 1e3, lv[-1] * 1e3),
                    interpolation="bilinear")
    ax.set_facecolor("#000000")
    cb = fig.colorbar(img, cax=cax)
    cb.set_label(label, color=theme.fg, fontsize=8)
    cb.ax.tick_params(colors=theme.fg, labelsize=7)
    cb.outline.set_edgecolor(theme.spine)

    # BER contours, the lightest the highest BER, the target's the thickest.
    if stat and e.contours is not None and len(e.contour_bers):
        n = len(e.contour_bers)
        cols = theme.contour[-n:] if n <= len(theme.contour) else theme.contour
        for k, (b, edge) in enumerate(zip(e.contour_bers, e.contours)):
            out = _closed(ph, edge, e.center_ui)
            if out is None:
                continue
            target = abs(np.log10(b) - np.log10(e.ber)) < 1e-6
            col = "#40d0ff" if target else cols[min(k, len(cols) - 1)]
            ax.plot(*(out[0], out[1] * 1e3), color=col, lw=1.8 if target else 0.9,
                    ls="-" if target else "--", label=f"BER {b:.0e}")
    if mask and mask[0] and mask[1] and e.height_v > 0:
        eh, ew = mask
        c, thr = e.center_ui, e.threshold_v * 1e3
        ax.fill([c - ew / 2, c, c + ew / 2, c, c - ew / 2],
                [thr, thr + eh * 500, thr, thr - eh * 500, thr], facecolor=theme.mask,
                alpha=0.22, edgecolor=theme.mask, lw=1.2, label="Receiver mask")
    # Height and width at the target BER.
    if e.height_v > 0:
        c, thr = e.center_ui, e.threshold_v * 1e3
        if stat and e.ber_contour is not None:
            k = int(np.argmin(np.abs(ph - c)))
            lo, hi = e.ber_contour[k] * 1e3
        else:
            lo, hi = thr - e.height_v * 500, thr + e.height_v * 500
        if np.isfinite(lo) and np.isfinite(hi):
            _arrow(ax, (c, lo), (c, hi), f"{e.height_v * 1e3:.0f} mV", theme, (6, 14), "left")
        if e.width_ui > 0:
            x0, x1 = c - e.width_ui / 2, c + e.width_ui / 2
            if stat and e.ber_contour is not None:
                out = _closed(ph, e.ber_contour, c)
                if out is not None:
                    x0, x1 = out[0].min(), out[0].max()
            _arrow(ax, (x0, thr), (x1, thr), f"{e.width_ui:.2f} UI", theme, (-28, -9), "center")
    ax.set_xlim(ph[0], ph[-1])
    ax.set_ylim(*_limits(e))
    ax.set_ylabel("Voltage at the receiver (mV)", fontsize=9)
    ax.grid(True, color=theme.grid, lw=0.4, alpha=0.6)
    if title:
        ax.set_title(title, fontsize=10)
    _style(ax, theme)
    ax.set_facecolor("#000000")
    _legend(ax, theme)
    if bx is not None:
        tub_v = np.clip(e.bathtub, 1e-30, 0.5)
        bx.semilogy(ph, tub_v, color="#08519c" if theme is LIGHT else "#40d0ff", lw=1.3)
        bx.axhline(e.ber, color=theme.mask, ls=":", lw=1)
        bx.text(ph[0], e.ber, f" BER {e.ber:.0e}", color=theme.mask, fontsize=7, va="bottom")
        bx.set_ylim(min(e.ber / 1e3, 1e-3), 1.0)
        bx.set_ylabel("BER", fontsize=9)
        bx.set_xlabel("Time (UI)", fontsize=9)
        bx.grid(True, color=theme.grid, lw=0.4, which="major")
        low = int(np.floor(np.log10(bx.get_ylim()[0])))
        bx.set_yticks([10.0 ** k for k in range(0, low - 1, -6 if low < -9 else -3)])
        bx.minorticks_off()
        ax.tick_params(labelbottom=False)
        _style(bx, theme)
    else:
        ax.set_xlabel("Time (UI)", fontsize=9)
    fig.patch.set_facecolor(theme.bg)
