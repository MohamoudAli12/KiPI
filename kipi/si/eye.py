"""Eye diagrams, equalisation and a channel operating margin, from a channel's response.

**Patterns.** PRBS7 to PRBS31 from their standard LFSRs.

**Linear eyes.** A linear channel's received waveform is its pulse response p(t) (the
response to one bit) superposed bit by bit: v(t) = Σ_n b_n·p(t − nT). The waveform eye
folds a PRBS's waveform on the UI. The **statistical eye** (StatEye's method) takes the
pulse response's cursors at each sampling phase φ, h_k = p(φ + kT), and convolves their
two-point distributions {0, h_k} into the inter-symbol interference's exact PDF, for a
'1' and a '0' sent; Gaussian noise widens it, and jitter (dual-Dirac DJ plus Gaussian
RJ) smears it across phases. Its contour at a BER (1e-12 by default) is the eye at that
BER, and its opening the eye height and width.

**Equalisation.** A transmitter FFE (taps on the bits, pre- and post-cursor), a receiver
CTLE (IEEE 802.3's form, one zero and two poles, peaking at the Nyquist frequency by its
DC gain), and an ideal DFE (its taps cancel the first post-cursors). :func:`optimize`
picks the CTLE gain and FFE taps that open the statistical eye most.

**COM** (after IEEE 802.3 Annex 93A, simplified): 20·log10(As/Ani), the main cursor's
height over the noise at the detector error ratio, the noise being the residual ISI,
crosstalk, jitter turned into noise by the pulse's slope, and Gaussian noise. 3 dB or
more passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

PRBS_TAPS = {7: (7, 6), 9: (9, 5), 11: (11, 9), 15: (15, 14), 23: (23, 18), 31: (31, 28)}


def prbs(order: int, n_bits: Optional[int] = None, seed: int = 1) -> np.ndarray:
    """A PRBS of ``order`` (x^a + x^b + 1), ``n_bits`` long (default one period)."""
    a, b = PRBS_TAPS[order]
    n = n_bits or (2 ** order - 1)
    state = seed or 1
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        bit = ((state >> (a - 1)) ^ (state >> (b - 1))) & 1
        state = ((state << 1) | bit) & ((1 << a) - 1)
        out[i] = bit
    return out


# ------------------------------------------------------------- equalisers
@dataclass
class Equalizer:
    ffe: Tuple[float, ...] = (1.0,)        # TX taps, the main one at index ffe_main
    ffe_main: int = 0
    ctle_db: float = 0.0                   # CTLE DC gain (≤ 0 dB: peaking of −ctle_db)
    dfe_taps: int = 0
    dfe_limit: float = 1.0                 # each tap at most this share of the main cursor

    def describe(self) -> str:
        parts = []
        if len(self.ffe) > 1:
            parts.append("FFE [" + ", ".join(f"{c:+.2f}" for c in self.ffe) + "]")
        if self.ctle_db:
            parts.append(f"CTLE {self.ctle_db:+.1f} dB")
        if self.dfe_taps:
            parts.append(f"DFE {self.dfe_taps} taps")
        return ", ".join(parts) or "none"


def ctle(freqs: np.ndarray, gdc_db: float, f_baud: float) -> np.ndarray:
    """IEEE 802.3's CTLE: (g + jf/fz)·fp1/... with fz = fp1 = f_baud/4 and fp2 = f_baud,
    normalised to unity at high frequency's peak: H = (g + jf/fp1)/((1 + jf/fp1)(1 + jf/fp2))."""
    g = 10 ** (gdc_db / 20)
    fp1, fp2 = f_baud / 4, f_baud
    jf = 1j * np.asarray(freqs)
    return (g + jf / fp1) / ((1 + jf / fp1) * (1 + jf / fp2))


def apply_ffe(pulse: np.ndarray, spui: int, eq: Equalizer) -> np.ndarray:
    out = np.zeros_like(pulse)
    for k, c in enumerate(eq.ffe):
        shift = (k - eq.ffe_main) * spui
        if shift >= 0:
            out[shift:] += c * pulse[:len(pulse) - shift]
        else:
            out[:shift] += c * pulse[-shift:]
    return out


# ------------------------------------------------------------ pulse, eye
@dataclass
class Pulse:
    """A pulse response sampled ``spui`` times a UI: v (volts per unit of swing) on t."""
    t: np.ndarray
    v: np.ndarray
    ui_s: float
    spui: int

    @property
    def main(self) -> int:
        return int(np.argmax(self.v))


def pulse_from_step(t: np.ndarray, step: np.ndarray, ui_s: float, spui: int = 32) -> Pulse:
    """The pulse response of one UI from a step response, resampled to ``spui`` per UI."""
    dt = ui_s / spui
    tt = np.arange(0, t[-1] - ui_s, dt)
    s = np.interp(tt, t, step)
    s_late = np.interp(tt - ui_s, t, step, left=step[0])
    return Pulse(tt, s - s_late, ui_s, spui)


def cursors(p: Pulse, phase: int, pre: int = 3, post: Optional[int] = None) -> np.ndarray:
    """The pulse at phase ``phase`` (sample index of the main cursor) and every UI
    around it: [h_-pre, ..., h_0, ..., h_post], zero where the pulse's window ends."""
    start = phase - pre * p.spui
    idx = start + p.spui * np.arange(pre + 1 + (post if post is not None else
                                                 max((len(p.v) - phase) // p.spui - 1, 0)))
    inside = (idx >= 0) & (idx < len(p.v))
    return np.where(inside, p.v[np.clip(idx, 0, len(p.v) - 1)], 0.0)


def resample(p: Pulse, spui: int) -> Pulse:
    """``p`` sampled ``spui`` times a UI (cubic between its samples)."""
    if spui == p.spui:
        return p
    from scipy.interpolate import CubicSpline
    t = np.arange(0.0, p.t[-1], p.ui_s / spui)
    return Pulse(t, CubicSpline(p.t, p.v)(t), p.ui_s, spui)


@dataclass
class EyeResult:
    """A statistical or waveform eye on ±1 UI around its middle (phase 0). ``density``
    (phases × levels) is the probability (statistical) or hit count (waveform)."""
    phases_ui: np.ndarray
    levels_v: np.ndarray
    density: np.ndarray
    ber_contour: Optional[np.ndarray] = None    # (phases, 2): lower and upper eye edge
    # The eye's inner edges at more BERs than the target (statistical eyes), and the
    # bathtubs: the BER sampling at each phase at the threshold, and at each threshold at
    # the eye's middle.
    contour_bers: List[float] = field(default_factory=list)
    contours: Optional[np.ndarray] = None       # (len(contour_bers), phases, 2)
    bathtub: Optional[np.ndarray] = None        # (phases,)
    vbathtub: Optional[np.ndarray] = None       # (levels,)
    height_v: float = 0.0
    width_ui: float = 0.0
    center_ui: float = 0.0
    threshold_v: float = 0.0
    ber: float = 1e-12
    main_cursor_v: float = 0.0
    com_db: Optional[float] = None
    eq: Equalizer = field(default_factory=Equalizer)
    kind: str = "statistical"
    notes: List[str] = field(default_factory=list)


def waveform_eye(t: np.ndarray, v: np.ndarray, ui_s: float, t0: float = 0.0,
                 levels: int = 400, phases: int = 512, skip_ui: int = 4,
                 upsample: int = 8) -> EyeResult:
    """A waveform eye: the waveform folded on two UIs around the eye's middle (phase 0),
    drawn as lines (``upsample`` points between samples) so each trace is continuous."""
    keep = t >= t0 + skip_ui * ui_s
    tt, vv = t[keep] - t0, v[keep]
    ph = ((tt / ui_s) + 0.5) % 2.0 - 0.5
    res = EyeResult(np.zeros(0), np.zeros(0), np.zeros((0, 0)), kind="waveform")
    _measure_waveform(res, ph, vv)
    c = res.center_ui
    lo, hi = vv.min(), vv.max()
    pad = 0.08 * (hi - lo + 1e-12)
    lv = np.linspace(lo - pad, hi + pad, levels)
    pr = np.linspace(-1.0, 1.0, phases)
    dp, dl = pr[1] - pr[0], lv[1] - lv[0]
    if upsample > 1 and len(tt) > 1:
        # Draw each step as a line: as many points along it as the bins it crosses.
        steps = np.maximum(np.abs(np.diff(vv)) / dl, np.diff(tt) / ui_s / dp)
        n = np.clip(np.ceil(steps).astype(int), 1, 64 * upsample)
        frac = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        frac = frac / np.repeat(n, n)
        k = np.repeat(np.arange(len(n)), n)
        tt = np.append(tt[k] + frac * (tt[k + 1] - tt[k]), tt[-1])
        vv = np.append(vv[k] + frac * (vv[k + 1] - vv[k]), vv[-1])
    ph = ((tt / ui_s) - c + 1.0) % 2.0 - 1.0              # the eye's middle at 0
    d, _, _ = np.histogram2d(ph, vv, bins=[np.append(pr, pr[-1] + dp) - dp / 2,
                                           np.append(lv, lv[-1] + dl) - dl / 2])
    res.phases_ui, res.levels_v, res.density = pr, lv, d
    res.center_ui = 0.0
    return res


def _measure_waveform(res: EyeResult, ph: np.ndarray, vv: np.ndarray):
    """The eye's middle half a UI from its threshold crossings (their circular mean), its
    width between the crossings either side of it and its height there (the inner edges
    of the samples within 1/128 UI, or half a sample, of it)."""
    thr = 0.5 * (vv.max() + vv.min())
    res.threshold_v = thr
    cross = np.flatnonzero(np.diff(np.sign(vv - thr)) != 0)
    cph = np.zeros(0)
    if len(cross):
        frac = (thr - vv[cross]) / (vv[cross + 1] - vv[cross])
        cph = (ph[cross] + frac * ((ph[cross + 1] - ph[cross] + 1) % 2.0 - 1)) % 1.0
        ang = np.angle(np.mean(np.exp(2j * np.pi * cph)))
        centre = (ang / (2 * np.pi) + 0.5) % 1.0
    else:
        centre = 0.5
    res.center_ui = float(centre)
    step = float(np.median(np.abs(np.diff(ph)))) if len(ph) > 1 else 0.0
    near = np.abs((ph - centre + 0.5) % 1.0 - 0.5) <= max(1 / 128, 0.51 * step)
    hi = vv[near & (vv > thr)]
    lo = vv[near & (vv <= thr)]
    res.height_v = float(max(hi.min() - lo.max(), 0.0)) if len(hi) and len(lo) else 0.0
    if len(cph):
        d = (cph - centre + 0.5) % 1.0 - 0.5
        left, right = d[d < 0], d[d >= 0]
        lo_w = left.max() if len(left) else -0.5
        hi_w = right.min() if len(right) else 0.5
        res.width_ui = float(max(hi_w - lo_w, 0.0)) if res.height_v > 0 else 0.0
    else:
        res.width_ui = 1.0 if res.height_v > 0 else 0.0


CONTOUR_BERS = (1e-3, 1e-6, 1e-9, 1e-12, 1e-15)


def statistical_eye(p: Pulse, ber: float = 1e-12, rj_ui: float = 0.0, dj_ui: float = 0.0,
                    noise_v: float = 0.0, levels: int = 512, eq: Optional[Equalizer] = None,
                    xtalk: Sequence[Pulse] = (), swing: float = 1.0,
                    phases_per_ui: Optional[int] = None) -> EyeResult:
    """The statistical eye of NRZ 0/1 through pulse response ``p`` (per unit swing), on
    ``phases_per_ui`` phases a UI (the pulse's own sampling if None; more for a smooth
    plot)."""
    eq = eq or Equalizer()
    if phases_per_ui and phases_per_ui != p.spui:
        p = resample(p, phases_per_ui)
        xtalk = [resample(x, phases_per_ui) for x in xtalk]
    spui = p.spui
    pv = p.v * swing
    main = _centre(pv)
    pre = 3
    span = np.max(np.abs(pv))
    vmin = -0.6 * span - 4 * noise_v
    vmax = 1.6 * span + 4 * noise_v
    lv = np.linspace(vmin, vmax, levels)
    dv = lv[1] - lv[0]
    phases = np.arange(-spui, spui + 1) + main     # ±1 UI around the eye's middle
    pdf1 = np.zeros((len(phases), levels))
    pdf0 = np.zeros((len(phases), levels))
    for k, ph in enumerate(phases):
        # Before the pulse's record starts the main cursor is 0 and the bits before it
        # still count (cursors() takes 0 outside the record): a slow line's UI is long
        # next to its delay, so its eye's first half UI lies there.
        if ph >= len(pv):
            continue
        c = cursors(Pulse(p.t, pv, p.ui_s, spui), ph, pre=pre)
        h0 = c[pre]
        others = np.delete(c, pre)
        if eq.dfe_taps:
            post = others[pre:pre + eq.dfe_taps].copy()
            lim = eq.dfe_limit * abs(h0)
            others[pre:pre + eq.dfe_taps] = post - np.clip(post, -lim, lim)
        isi = _isi_pdf(others, lv, dv)
        for x in xtalk:              # each aggressor bit moves the victim by ±h/2
            isi = _convolve_pdf(isi, 0.5 * swing * x.v[ph % spui::spui], lv)
        if noise_v > 0:
            isi = _gauss(isi, noise_v / dv)
        shift = int(round(h0 / dv))
        pdf0[k] = isi
        pdf1[k] = np.roll(isi, shift)
        if shift > 0:
            pdf1[k, :shift] = 0
    if rj_ui > 0 or dj_ui > 0:
        pdf0 = _jitter(pdf0, spui, rj_ui, dj_ui)
        pdf1 = _jitter(pdf1, spui, rj_ui, dj_ui)
    density = 0.5 * (pdf0 + pdf1)
    res = EyeResult((phases - main) / spui, lv, density, eq=eq, ber=ber,
                    main_cursor_v=float(pv[main]))
    _ber_contour(res, pdf0, pdf1, ber)
    return res


def _centre(pv: np.ndarray) -> int:
    """The middle of the pulse's top: of the run above half its peak around the peak
    (the eye's middle, where a flat-topped pulse's peak can be anywhere along it)."""
    peak = int(np.argmax(pv))
    above = pv >= 0.5 * pv[peak]
    i0, i1 = peak, peak
    while i0 > 0 and above[i0 - 1]:
        i0 -= 1
    while i1 < len(pv) - 1 and above[i1 + 1]:
        i1 += 1
    return (i0 + i1) // 2


def _isi_pdf(c: np.ndarray, lv: np.ndarray, dv: float) -> np.ndarray:
    """The PDF over levels ``lv`` of Σ b_k·c_k, b_k ∈ {0, 1} equally likely."""
    n = len(lv)
    pdf = np.zeros(n)
    zero = int(round(-lv[0] / dv))
    pdf[min(max(zero, 0), n - 1)] = 1.0
    for h in c:
        s = int(round(h / dv))
        if s == 0:
            continue
        moved = np.zeros(n)
        if s > 0:
            moved[s:] = pdf[:-s]
        else:
            moved[:s] = pdf[-s:]
        pdf = 0.5 * pdf + 0.5 * moved
    return pdf


def _convolve_pdf(pdf, cursors_x, lv):
    dv = lv[1] - lv[0]
    for h in np.atleast_1d(cursors_x):
        s = int(round(h / dv))
        if s == 0:
            continue
        a = np.roll(pdf, s)
        b = np.roll(pdf, -s)
        pdf = 0.5 * (a + b)
    return pdf


def _gauss(pdf: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins < 0.3:
        return pdf
    half = int(math.ceil(5 * sigma_bins))
    x = np.arange(-half, half + 1)
    g = np.exp(-0.5 * (x / sigma_bins) ** 2)
    g /= g.sum()
    return np.convolve(pdf, g, mode="same")


def _jitter(pdf: np.ndarray, spui: int, rj_ui: float, dj_ui: float) -> np.ndarray:
    """Smear across phases: dual-Dirac ±DJ/2 and Gaussian RJ (σ in UI)."""
    from scipy.ndimage import convolve1d
    half = int(math.ceil(0.5 * dj_ui * spui + 5 * rj_ui * spui)) + 1
    kern = np.zeros(2 * half + 1)
    for d in (-0.5 * dj_ui, 0.5 * dj_ui):
        kern[half + int(round(d * spui))] += 0.5
    if rj_ui > 0:
        x = np.arange(-half, half + 1)
        g = np.exp(-0.5 * (x / (rj_ui * spui)) ** 2)
        kern = np.convolve(kern, g / g.sum(), mode="same")
    return convolve1d(pdf, kern, axis=0, mode="constant")


def _edges(lv: np.ndarray, cdf1: np.ndarray, t0: np.ndarray, ber: float):
    """(contour (phases, 2), heights (phases,)) of the eye's inner edges at ``ber``."""
    n = cdf1.shape[0]
    contour = np.full((n, 2), np.nan)
    heights = np.zeros(n)
    for k in range(n):
        c1, z0 = cdf1[k], t0[k]
        if not (np.isfinite(c1[-1]) and np.isfinite(z0[0])):
            continue
        upper = lv[np.searchsorted(c1, ber)] if c1.max() >= ber else lv[-1]
        lower_idx = np.flatnonzero(z0 >= ber)
        lower = lv[lower_idx[-1]] if len(lower_idx) else lv[0]
        if upper > lower:
            contour[k] = (lower, upper)
            heights[k] = upper - lower
    return contour, heights


def _ber_contour(res: EyeResult, pdf0: np.ndarray, pdf1: np.ndarray, ber: float):
    """The eye's inner edges at ``ber`` (and at :data:`CONTOUR_BERS` above it) at each
    phase, its height and width, and the bathtubs through its middle."""
    lv = res.levels_v
    tot1 = pdf1.sum(axis=1, keepdims=True)
    tot0 = pdf0.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cdf1 = np.cumsum(pdf1, axis=1) / np.where(tot1 > 0, tot1, np.nan)  # P(1 ≤ level)
        t0 = (tot0 - np.cumsum(pdf0, axis=1) + pdf0) / np.where(tot0 > 0, tot0, np.nan)
    contour, heights = _edges(lv, cdf1, t0, ber)
    res.ber_contour = contour
    res.contour_bers = sorted({b for b in CONTOUR_BERS if b > ber} | {ber}, reverse=True)
    res.contours = np.array([contour if b == ber else _edges(lv, cdf1, t0, b)[0]
                             for b in res.contour_bers])
    open_ = heights > 0
    best = int(np.argmax(heights))
    if open_.any():
        # The open run around the tallest phase; the eye's middle is the run's middle,
        # where its height is measured (as a scope's eye measurement does).
        i0 = best
        while i0 > 0 and open_[i0 - 1]:
            i0 -= 1
        i1 = best
        while i1 < len(open_) - 1 and open_[i1 + 1]:
            i1 += 1
        mid = (i0 + i1) // 2
        res.width_ui = float(res.phases_ui[i1] - res.phases_ui[i0] + 1.0 / max(
            1, round(1 / (res.phases_ui[1] - res.phases_ui[0]))))
        res.height_v = float(heights[mid])
        res.center_ui = float(res.phases_ui[mid])
        res.threshold_v = float(np.mean(contour[mid]))
    else:
        res.width_ui = res.height_v = 0.0
        res.center_ui = float(res.phases_ui[best])
        res.threshold_v = 0.0
    # The bathtub: a bit's error probability at each phase, sampled at the threshold.
    j = int(np.clip(np.searchsorted(lv, res.threshold_v), 0, len(lv) - 1))
    tub = 0.5 * (np.nan_to_num(cdf1[:, j], nan=1.0) + np.nan_to_num(t0[:, j], nan=1.0))
    res.bathtub = np.clip(tub, 1e-300, 0.5)
    # The voltage bathtub: the error probability at the eye's middle for each threshold.
    k = int(np.argmin(np.abs(res.phases_ui - res.center_ui)))
    vtub = 0.5 * (np.nan_to_num(cdf1[k], nan=1.0) + np.nan_to_num(t0[k], nan=1.0))
    res.vbathtub = np.clip(vtub, 1e-300, 0.5)


def com(p: Pulse, der: float = 1e-12, rj_ui: float = 0.01, dj_ui: float = 0.05,
        noise_v: float = 0.0, xtalk: Sequence[Pulse] = (), eq: Optional[Equalizer] = None,
        swing: float = 1.0) -> Tuple[float, float, float]:
    """(COM dB, As, Ani): the main cursor over the noise at the detector error ratio."""
    eq = eq or Equalizer()
    pv = p.v * swing
    main = int(np.argmax(pv))
    c = cursors(Pulse(p.t, pv, p.ui_s, p.spui), main, pre=3)
    a_s = c[3]
    others = np.delete(c, 3)
    if eq.dfe_taps:
        post = others[3:3 + eq.dfe_taps]
        lim = eq.dfe_limit * abs(a_s)
        others[3:3 + eq.dfe_taps] = post - np.clip(post, -lim, lim)
    # NRZ ±A/2 symbols: each cursor contributes ±h/2.
    span = 2 * (np.abs(others).sum() / 2 + np.abs(a_s) + 6 * noise_v + 1e-12)
    lv = np.linspace(-span, span, 4096)
    dv = lv[1] - lv[0]
    pdf = np.zeros(len(lv))
    pdf[len(lv) // 2] = 1.0
    pdf = _convolve_pdf(pdf, 0.5 * others, lv)
    for x in xtalk:
        xs = x.v[main % x.spui::x.spui] * swing
        pdf = _convolve_pdf(pdf, 0.5 * xs, lv)
    slope = np.gradient(pv, p.t)[main] * p.ui_s
    sj = math.hypot(noise_v, slope * rj_ui)
    pdf = _gauss(pdf, sj / dv)
    if dj_ui > 0:
        pdf = _convolve_pdf(pdf, [0.5 * slope * dj_ui], lv)
    cdf = np.cumsum(pdf[::-1])[::-1]
    idx = np.flatnonzero(cdf >= der)
    a_ni = lv[idx[-1]] if len(idx) else lv[-1]
    a_ni = max(a_ni, 1e-12)
    return 20 * math.log10(max(a_s / 2, 1e-15) / a_ni), float(a_s / 2), float(a_ni)


def optimize(freqs: np.ndarray, h: np.ndarray, ui_s: float, rise_s: float,
             ctle_range=(0.0, -15.0), ffe_presets: Sequence[Tuple[float, ...]] = ((1.0,),),
             dfe_taps: int = 0, spui: int = 32) -> Tuple[Equalizer, Pulse]:
    """The CTLE gain (in 1 dB steps over ``ctle_range``) and FFE preset that maximise the
    statistical eye height (no jitter) of transfer function ``h`` (F,)."""
    from kipi.si.sparams import step
    best = None
    lo, hi = sorted(ctle_range)
    for g in np.arange(hi, lo - 0.5, -1.0):
        hc = h * (ctle(freqs, g, 1 / ui_s) if g != 0 else 1.0)
        t, s = step(freqs, hc, rise_s, oversample=4)
        base = pulse_from_step(t, s, ui_s, spui)
        for taps in ffe_presets:
            main_i = int(np.argmax(np.abs(taps)))
            eq = Equalizer(tuple(taps), main_i, float(g), dfe_taps)
            pl = Pulse(base.t, apply_ffe(base.v, spui, eq), ui_s, spui)
            e = statistical_eye(pl, 1e-12, eq=eq, levels=256)
            if best is None or e.height_v > best[0]:
                best = (e.height_v, eq, pl)
    return best[1], best[2]
