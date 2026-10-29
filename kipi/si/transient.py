"""Time-domain simulation of a channel with nonlinear IBIS buffers at its ports.

The channel is linear: its port S-parameters (reference R0) on a uniform frequency grid
give impulse responses h_ij(t) (an inverse FFT, tapered near f_max so the band edge does
not ring). With incident and reflected waves a = (v + R0·i)/(2√R0) and
b = (v − R0·i)/(2√R0) at each port (i into the channel),

    b(n) = H0·a(n) + Σ_k≥1 H_k·a(n − k),

so at each time step the channel is a Norton equivalent i = G·v + i_hist at its ports.
Each port's device (a driver, a receiver, a termination) adds its own current, and the
pad voltages are solved by Newton's method; C_comp is integrated by backward Euler. The
history starts from the DC operating point, so a run starts settled.

**Linear runs.** With linear terminations the response is exact in the frequency domain
instead: (Yp + Yt)·V = I_src at each frequency (:func:`port_transfer`), for the pulse and
step responses the eye diagrams are built from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from kipi.si.ibis import Buffer
from kipi.si.sparams import s_from_y

R0 = 50.0


@dataclass
class Stimulus:
    """A driver's data: ``bits`` (0/1) at ``ui_s`` each, starting ``delay_s`` in."""
    bits: Sequence[int]
    ui_s: float
    delay_s: float = 0.0
    invert: bool = False

    def edges(self):
        """(time, "rise"/"fall") of each change, and the level before the first."""
        b = [1 - x if self.invert else x for x in self.bits]
        out = []
        for k in range(1, len(b)):
            if b[k] != b[k - 1]:
                out.append((self.delay_s + k * self.ui_s, "rise" if b[k] else "fall"))
        return b[0], out


@dataclass
class Device:
    """What sits at a port: an IBIS ``buffer`` (driven by ``stimulus`` if it drives,
    else a receiver held by its clamps), and/or a termination ``r_term`` to ``v_term``,
    and extra capacitance to ground."""
    buffer: Optional[Buffer] = None
    stimulus: Optional[Stimulus] = None
    r_term: Optional[float] = None
    v_term: float = 0.0
    c_extra: float = 0.0
    hold_low: bool = False              # a driver that does not switch: held low
    hold_high: bool = False

    @property
    def c(self) -> float:
        return (self.buffer.c_comp if self.buffer else 0.0) + self.c_extra


class _DriverState:
    def __init__(self, dev: Device):
        self.dev = dev
        b = dev.buffer
        if dev.stimulus is not None:
            first, self.edges = dev.stimulus.edges()
        else:
            first, self.edges = (1 if dev.hold_high else 0), []
        self.k = (1.0, 0.0) if first else (0.0, 1.0)
        self.start = self.k
        self.edge = None
        self.t_edge = 0.0
        self.i = 0
        self.b = b

    def coefficients(self, t: float):
        while self.i < len(self.edges) and self.edges[self.i][0] <= t:
            self.start = self.coefficients_now(self.edges[self.i][0])
            self.t_edge, self.edge = self.edges[self.i]
            self.i += 1
        return self.coefficients_now(t)

    def coefficients_now(self, t: float):
        if self.edge is None:
            return self.k
        ku_e, kd_e = self.b.coefficients(self.edge, np.array([t - self.t_edge]))
        ku_e, kd_e = float(ku_e[0]), float(kd_e[0])
        if self.edge == "rise":
            tgt_u, tgt_d, e0u, e0d = 1.0, 0.0, 0.0, 1.0
        else:
            tgt_u, tgt_d, e0u, e0d = 0.0, 1.0, 1.0, 0.0
        su, sd = self.start
        ku = su + (ku_e - e0u) * (tgt_u - su) / (tgt_u - e0u)
        kd = sd + (kd_e - e0d) * (tgt_d - sd) / (tgt_d - e0d)
        return ku, kd


def impulse_responses(freqs: np.ndarray, s: np.ndarray, dt: float, taper: float = 0.25):
    """h (P, P, K) of S (F, P, P) on f_k = k·Δf, sampled at ``dt`` (zero-padded), tapered
    over the top ``taper`` of the band, and trimmed where they have died away."""
    df = freqs[0]
    f_all = np.concatenate([[0.0], freqs])
    s0 = s[0].real if len(s) < 2 else (2 * s[0] - s[1]).real
    sf = np.concatenate([s0[None], s], axis=0)
    w = np.ones(len(f_all))
    f_max = f_all[-1]
    edge = f_all > (1 - taper) * f_max
    w[edge] = 0.5 + 0.5 * np.cos(np.pi * (f_all[edge] - (1 - taper) * f_max) / (taper * f_max))
    n = int(2 ** math.ceil(math.log2(max(1.0 / (df * dt), 2 * (len(f_all) - 1)))))
    h = np.fft.irfft(sf * w[:, None, None], n, axis=0)        # (n, P, P), per step
    dt_eff = 1.0 / (n * df)
    h = h.transpose(1, 2, 0)
    # Keep the causal part, until the responses have died away.
    keep = n // 2
    energy = np.cumsum((h[:, :, :keep] ** 2).sum(axis=(0, 1)))
    if energy[-1] > 0:
        k_end = int(np.searchsorted(energy, energy[-1] * (1 - 1e-9))) + 1
        keep = max(min(keep, k_end + 16), 8)
    return h[:, :, :keep], dt_eff


@dataclass
class TransientResult:
    t: np.ndarray
    v: np.ndarray                       # (T, P) pad voltages
    i: np.ndarray                       # (T, P) currents into the channel
    port_names: List[str] = field(default_factory=list)


def simulate(freqs: np.ndarray, y: np.ndarray, devices: Sequence[Device], t_stop: float,
             dt: float, port_names: Sequence[str] = (), check_cancel=None) -> TransientResult:
    """Run the channel (port admittance ``y`` (F, P, P) on a uniform grid) with
    ``devices`` at its ports for ``t_stop`` seconds, steps of about ``dt``."""
    p = y.shape[-1]
    s = s_from_y(y, R0)
    h, dt = impulse_responses(freqs, s, dt)
    k = h.shape[-1]
    sq = math.sqrt(R0)
    eye = np.eye(p)
    h0 = h[:, :, 0]
    m = np.linalg.inv(eye + h0)
    g_lin = (eye - h0) @ m / R0
    hk = h[:, :, 1:]
    states = [_DriverState(d) if d.buffer is not None and d.buffer.model.is_driver else None
              for d in devices]
    caps = np.array([d.c for d in devices])

    def dev_current(v, t):
        i = np.zeros(p)
        g = np.zeros(p)
        for j, d in enumerate(devices):
            if d.buffer is not None:
                if states[j] is not None:
                    ku, kd = states[j].coefficients(t)
                    ij, gj = d.buffer.drive(np.array([v[j]]), ku, kd)
                else:
                    ij, gj = d.buffer.static(np.array([v[j]]))
                i[j] += ij[0]
                g[j] += gj[0]
            if d.r_term:
                i[j] += (v[j] - d.v_term) / d.r_term
                g[j] += 1.0 / d.r_term
        return i, g

    # DC operating point: the channel's DC admittance.
    y_dc = y[0].real
    v = np.zeros(p)
    for _ in range(60):
        i_dev, g_dev = dev_current(v, -1.0)
        f = y_dc @ v + i_dev
        jac = y_dc + np.diag(g_dev + 1e-12)
        dv = np.linalg.solve(jac, -f)
        dv = np.clip(dv, -0.5, 0.5)
        v = v + dv
        if np.abs(dv).max() < 1e-9:
            break
    i_ch = y_dc @ v
    a_ss = (v + R0 * i_ch) / (2 * sq)
    n_steps = int(math.ceil(t_stop / dt))
    a = np.zeros((p, n_steps + k))
    a[:, :k] = a_ss[:, None]
    t_out = np.arange(n_steps) * dt
    v_out = np.zeros((n_steps, p))
    i_out = np.zeros((n_steps, p))
    v_prev = v.copy()
    for n in range(n_steps):
        if check_cancel and n % 2000 == 0:
            check_cancel()
        t = t_out[n]
        col = k + n
        past = a[:, col - 1:col - k:-1] if k > 1 else np.zeros((p, 0))
        hist = np.einsum("ijk,jk->i", hk, past) if k > 1 else np.zeros(p)
        i0 = ((eye - h0) @ (m @ (-hist)) - hist) / sq
        for _ in range(30):
            i_dev, g_dev = dev_current(v, t)
            f = g_lin @ v + i0 + i_dev + caps * (v - v_prev) / dt
            jac = g_lin + np.diag(g_dev + caps / dt)
            dv = np.linalg.solve(jac, -f)
            dv = np.clip(dv, -0.3, 0.3)
            v = v + dv
            if np.abs(dv).max() < 1e-7:
                break
        an = m @ (v / sq - hist)
        a[:, col] = an
        v_out[n] = v
        i_out[n] = g_lin @ v + i0
        v_prev = v.copy()
    return TransientResult(t_out, v_out, i_out, list(port_names))


# ------------------------------------------------------------------ linear
@dataclass
class LinearPort:
    """A linear termination at a port: a source of impedance ``r_source`` (with
    ``c_shunt`` beside it) driven by weight ``drive`` (0: none, ±1: this port's share of
    the source), or a load of ``r_load`` and ``c_shunt`` to ground."""
    drive: float = 0.0
    r_source: float = 50.0
    r_load: Optional[float] = None
    c_shunt: float = 0.0

    def admittance(self, w):
        y = 1j * w * self.c_shunt + 0j
        if self.drive:
            y = y + 1.0 / self.r_source
        elif self.r_load:
            y = y + 1.0 / self.r_load
        return y


def port_transfer(freqs: np.ndarray, y: np.ndarray, ports: Sequence[LinearPort]) -> np.ndarray:
    """Each port's voltage (F, P) per volt of source, with ``ports`` terminating it."""
    w = 2 * np.pi * freqs
    p = y.shape[-1]
    yt = np.stack([pt.admittance(w) * np.ones_like(w) for pt in ports], axis=-1)   # (F, P)
    src = np.array([pt.drive / pt.r_source if pt.drive else 0.0 for pt in ports])
    a = y + yt[:, :, None] * np.eye(p)[None]
    return np.linalg.solve(a, np.broadcast_to(src, (len(freqs), p))[..., None])[..., 0]
