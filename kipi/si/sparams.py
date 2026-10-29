"""S-parameter utilities: conversions, mixed-mode, interpolation, time-domain responses
(TDR, step and pulse), Touchstone output, passivity and causality checks.

Time-domain responses come from a uniform frequency grid f_k = k·Δf (k ≥ 1) with the DC
point extrapolated: the inverse real FFT of the response times a Gaussian edge filter
(the 10–90 % rise time asked for) gives the impulse response on Δt = 1/(2·f_max) over
T = 1/Δf; its running sum is the step response.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _sqrt_z0(z0, p):
    z0 = np.broadcast_to(np.asarray(z0, dtype=float), (p,))
    return np.diag(np.sqrt(z0))


def s_from_y(y: np.ndarray, z0=50.0) -> np.ndarray:
    p = y.shape[-1]
    rz = _sqrt_z0(z0, p)
    yn = rz @ y @ rz
    eye = np.eye(p)
    return np.linalg.solve((eye + yn).transpose(0, 2, 1), (eye - yn).transpose(0, 2, 1)
                           ).transpose(0, 2, 1)


def y_from_s(s: np.ndarray, z0=50.0) -> np.ndarray:
    p = s.shape[-1]
    rz = _sqrt_z0(z0, p)
    ri = np.linalg.inv(rz)
    eye = np.eye(p)
    return ri @ np.linalg.solve(eye + s, eye - s) @ ri


def z_from_s(s: np.ndarray, z0=50.0) -> np.ndarray:
    return np.linalg.inv(y_from_s(s, z0))


def interpolate(freqs: np.ndarray, data: np.ndarray, f_new: np.ndarray) -> np.ndarray:
    """Data (F, ...) on ``freqs`` at ``f_new``: magnitude and unwrapped phase linearly,
    held at the end values outside."""
    f_new = np.asarray(f_new, dtype=float)
    shape = data.shape[1:]
    flat = data.reshape(len(freqs), -1)
    mag = np.abs(flat)
    ph = np.unwrap(np.angle(flat), axis=0)
    out = np.empty((len(f_new), flat.shape[1]), complex)
    for j in range(flat.shape[1]):
        m = np.interp(f_new, freqs, mag[:, j])
        a = np.interp(f_new, freqs, ph[:, j])
        out[:, j] = m * np.exp(1j * a)
    return out.reshape((len(f_new), *shape))


def mixed_mode(s: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Mixed-mode S of 2·K single-ended ports grouped in ``pairs`` (p, n): ordered
    [d1..dK, c1..cK], so Sdd = mm[:, :K, :K], Sdc = mm[:, :K, K:], Scd = mm[:, K:, :K],
    Scc = mm[:, K:, K:]."""
    k = len(pairs)
    m = np.zeros((2 * k, s.shape[-1]))
    for i, (p, n) in enumerate(pairs):
        m[i, p], m[i, n] = 1 / math.sqrt(2), -1 / math.sqrt(2)
        m[k + i, p], m[k + i, n] = 1 / math.sqrt(2), 1 / math.sqrt(2)
    return m @ s @ m.T


def db(x) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(x), 1e-30))


def uniform_grid(f_max_hz: float, df_hz: float) -> np.ndarray:
    """f_k = k·Δf, k = 1..K, up to ``f_max_hz``: the grid the time-domain responses need."""
    k = int(math.ceil(f_max_hz / df_hz))
    return df_hz * np.arange(1, k + 1)


def _with_dc(freqs: np.ndarray, h: np.ndarray) -> np.ndarray:
    """``h`` (F,) on f_k = k·Δf with a DC point in front: the lowest point's real part
    (a response is real at DC), linearly extrapolated in magnitude."""
    h0 = h[0].real if len(h) < 2 else (2 * h[0] - h[1]).real
    if np.sign(h0) != np.sign(h[0].real) and abs(h[0].real) > 0:
        h0 = h[0].real
    return np.concatenate([[h0], h])


def gaussian_filter(freqs: np.ndarray, rise_s: float) -> np.ndarray:
    """A Gaussian edge of 10–90 % rise time ``rise_s`` in frequency."""
    if rise_s <= 0:
        return np.ones_like(freqs)
    sigma_t = rise_s / 2.563                    # 10–90 % of a Gaussian step
    return np.exp(-0.5 * (2 * math.pi * freqs * sigma_t) ** 2)


def impulse(freqs: np.ndarray, h: np.ndarray, rise_s: float = 0.0, oversample: int = 1):
    """(t, impulse response per time step) of the response ``h`` (F,) on a uniform grid
    f_k = k·Δf; the samples sum to the step response."""
    df = freqs[0]
    hf = _with_dc(freqs, h) * gaussian_filter(np.concatenate([[0], freqs]), rise_s)
    n = 2 * (len(hf) - 1) * oversample
    x = np.fft.irfft(hf, n)
    dt = 1.0 / (n * df)
    return np.arange(n) * dt, x


def step(freqs: np.ndarray, h: np.ndarray, rise_s: float = 0.0, oversample: int = 1):
    """(t, step response) of ``h`` with a Gaussian edge of ``rise_s``. The edge is
    centred on t = 0, so a response that starts at once (a reflection at the port, near-end
    crosstalk) has half its first edge before 0, wrapped to the record's end: that part is
    summed in first, else the step would settle at half its height."""
    t, x = impulse(freqs, h, rise_s, oversample)
    dt = t[1] - t[0] if len(t) > 1 else 0.0
    pre = int(math.ceil(5 * rise_s / 2.563 / dt)) if rise_s > 0 and dt > 0 else 0
    pre = min(pre, len(x) // 4)
    if pre == 0:
        return t, np.cumsum(x)
    s = np.cumsum(np.roll(x, pre))[pre:]
    return t[:len(s)], s


def tdr(freqs: np.ndarray, s11: np.ndarray, z0: float = 50.0, rise_s: float = 35e-12):
    """(t, impedance seen into port 1) of a TDR with a ``rise_s`` step: the reflected
    step ρ(t) as Z = z0·(1 + ρ)/(1 − ρ). Time is round trip."""
    t, rho = step(freqs, s11, rise_s, oversample=2)
    rho = np.clip(rho, -0.999, 0.999)
    return t, z0 * (1 + rho) / (1 - rho)


def passivity(s: np.ndarray) -> float:
    """The largest singular value of S over frequency: above 1 is not passive."""
    return float(np.linalg.svd(s, compute_uv=False).max())


def reciprocity(s: np.ndarray) -> float:
    """The largest |Sij − Sji| over frequency."""
    return float(np.abs(s - s.transpose(0, 2, 1)).max())


def causality(freqs: np.ndarray, h: np.ndarray) -> float:
    """How non-causal a response on a uniform grid is: the energy of its impulse
    response in the second half of the time window (where a causal response's would have
    to wrap around), over the whole."""
    _, x = impulse(freqs, h)
    half = len(x) // 2
    return float(np.sum(x[half + half // 2:] ** 2) / max(np.sum(x ** 2), 1e-300))


def write_touchstone(path, freqs_hz: np.ndarray, s: np.ndarray, z0: float = 50.0,
                     port_names: Optional[List[str]] = None, comment: str = "") -> Path:
    """Write S (F, P, P) as Touchstone 1 ``.sPp`` in RI format, port names as comments."""
    p = s.shape[-1]
    path = Path(path)
    if path.suffix.lower() != f".s{p}p":
        path = path.with_suffix(f".s{p}p")
    lines = [f"! {line}" for line in (comment or "Written by KiPI").splitlines()]
    for i, n in enumerate(port_names or []):
        lines.append(f"! Port[{i + 1}] = {n}")
    lines.append(f"# Hz S RI R {z0:g}")
    for f, m in zip(freqs_hz, s):
        # Touchstone 1 writes a 2-port column-major (S11 S21 S12 S22), the rest row-major.
        vals = m.T.ravel() if p == 2 else m.ravel()
        parts = [f"{f:.9g}"] + [f"{v.real:.9e} {v.imag:.9e}" for v in vals]
        if p <= 2:
            lines.append(" ".join(parts))
        else:                       # each row on new lines, 4 pairs a line, f on the first
            first = True
            for r in range(p):
                row = parts[1 + r * p: 1 + (r + 1) * p]
                for k in range(0, len(row), 4):
                    lines.append((parts[0] + " " if first else "  ") + " ".join(row[k:k + 4]))
                    first = False
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
