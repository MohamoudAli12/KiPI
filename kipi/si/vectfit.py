"""Rational macromodels of a channel: vector fitting (Gustavsen and Semlyen's relaxed form,
common poles for every element), passivity, and a SPICE subcircuit that any SPICE
simulator runs (resistors, capacitors, voltage-controlled current and voltage sources).

S-parameters (F, P, P) are fitted as

    S(s) ≈ D + Σ_k R_k / (s − a_k)

with stable poles a_k (real, or complex conjugate pairs) shared by every element, then
scaled if need be so that no singular value exceeds 1 from DC to twice the band (passive).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class RationalModel:
    poles: np.ndarray                  # (N,) complex; a pair appears as a, conj(a) in a row
    residues: np.ndarray               # (N, P, P) complex
    d: np.ndarray                      # (P, P) real
    rms_error: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def ports(self) -> int:
        return self.d.shape[0]

    def __call__(self, freqs: np.ndarray) -> np.ndarray:
        s = 2j * np.pi * np.asarray(freqs, float)
        out = np.broadcast_to(self.d, (len(s),) + self.d.shape).astype(complex)
        for a, r in zip(self.poles, self.residues):
            out = out + r[None] / (s[:, None, None] - a)
        return out


def _start_poles(f_lo: float, f_hi: float, n: int) -> np.ndarray:
    """n (even) complex poles, in pairs, spread linearly over the band, lightly damped."""
    beta = np.linspace(2 * np.pi * max(f_lo, f_hi * 1e-3), 2 * np.pi * f_hi, n // 2)
    out = []
    for b in beta:
        a = -b / 100 + 1j * b
        out += [a, np.conj(a)]
    return np.array(out)


def _basis(s: np.ndarray, poles: np.ndarray) -> np.ndarray:
    """The real basis (F, N): 1/(s−a) for a real pole; for a pair a, a*:
    1/(s−a) + 1/(s−a*) and j/(s−a) − j/(s−a*)."""
    cols = []
    k = 0
    while k < len(poles):
        a = poles[k]
        if abs(a.imag) < 1e-12 * max(abs(a), 1.0):
            cols.append(1 / (s - a.real))
            k += 1
        else:
            cols.append(1 / (s - a) + 1 / (s - np.conj(a)))
            cols.append(1j / (s - a) - 1j / (s - np.conj(a)))
            k += 2
    return np.stack(cols, axis=1)


def _realify(m: np.ndarray) -> np.ndarray:
    return np.concatenate([m.real, m.imag], axis=0)


def _update_poles(poles, c, d_sigma):
    """The zeros of σ(s) = d + Σ c/(s−a): the new poles."""
    n = len(poles)
    a = np.zeros((n, n))
    b = np.zeros(n)
    k = 0
    while k < n:
        p = poles[k]
        if abs(p.imag) < 1e-12 * max(abs(p), 1.0):
            a[k, k], b[k] = p.real, 1.0
            k += 1
        else:
            a[k:k + 2, k:k + 2] = [[p.real, p.imag], [-p.imag, p.real]]
            b[k], b[k + 1] = 2.0, 0.0
            k += 2
    z = np.linalg.eigvals(a - np.outer(b, c) / d_sigma)
    z = np.where(z.real > 0, -z.real + 1j * z.imag, z)        # flip unstable poles
    return _sort_pairs(z)


def _sort_pairs(z: np.ndarray) -> np.ndarray:
    """Real poles first, then pairs as a (Im > 0), conj(a)."""
    tol = 1e-9
    real = sorted([x.real for x in z if abs(x.imag) <= tol * max(abs(x), 1.0)])
    up = sorted([x for x in z if x.imag > tol * max(abs(x), 1.0)], key=lambda x: x.imag)
    out = [complex(x) for x in real]
    for x in up:
        out += [x, np.conj(x)]
    return np.array(out, complex)


def fit(freqs: np.ndarray, h: np.ndarray, n_poles: int = 20, iterations: int = 8,
        weight: Optional[np.ndarray] = None) -> RationalModel:
    """Vector-fit ``h`` (F, P, P) over ``freqs`` (Hz) with ``n_poles`` common poles."""
    freqs = np.asarray(freqs, float)
    s = 2j * np.pi * freqs
    nf, p, _ = h.shape
    cols = h.reshape(nf, p * p)
    w = np.ones(nf) if weight is None else np.asarray(weight, float)
    poles = _start_poles(freqs[0], freqs[-1], n_poles + (n_poles % 2))
    n = len(poles)
    for _ in range(iterations):
        phi = _basis(s, poles)                                 # (F, N)
        phi1 = np.concatenate([phi, np.ones((nf, 1))], axis=1)  # with d
        rows = []
        for m in range(cols.shape[1]):
            hm = cols[:, m]
            a = np.concatenate([phi1 * w[:, None], -(hm * w)[:, None] * phi1], axis=1)
            ar = _realify(a)
            q, r = np.linalg.qr(ar)
            rows.append(r[n + 1:, n + 1:])                     # the σ unknowns' block
        # Relaxation: the mean of σ over the band is fixed (Gustavsen 2006).
        relax = np.concatenate([np.real(phi1).sum(axis=0)]) * (np.sqrt(cols.shape[1]) / nf)
        big = np.vstack(rows + [relax[None, :]])
        rhs = np.zeros(big.shape[0])
        rhs[-1] = np.sqrt(cols.shape[1])
        x, *_ = np.linalg.lstsq(big, rhs, rcond=None)
        c, d_sigma = x[:n], x[n]
        if abs(d_sigma) < 1e-8:
            d_sigma = 1e-8 if d_sigma >= 0 else -1e-8
        poles = _update_poles(poles, c, d_sigma)
    # Residues and D with the final poles.
    phi = _basis(s, poles)
    phi1 = np.concatenate([phi, np.ones((nf, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(_realify(phi1 * w[:, None]), _realify(cols * w[:, None]),
                               rcond=None)                    # (N+1, M)
    res = np.zeros((n, p * p), complex)
    k = 0
    while k < n:
        a = poles[k]
        if abs(a.imag) < 1e-12 * max(abs(a), 1.0):
            res[k] = coef[k]
            k += 1
        else:
            res[k] = coef[k] + 1j * coef[k + 1]
            res[k + 1] = np.conj(res[k])
            k += 2
    model = RationalModel(poles, res.reshape(n, p, p), coef[n].reshape(p, p))
    err = model(freqs) - h
    model.rms_error = float(np.sqrt(np.mean(np.abs(err) ** 2)))
    return model


FIT_POINTS = 400                     # frequencies fitted at most (the error is on all)


def fit_sparams(freqs: np.ndarray, s: np.ndarray, n_poles: int = 0,
                tol: float = 2e-3) -> RationalModel:
    """Fit S-parameters ``s`` (F, P, P); with ``n_poles`` 0, add poles (in pairs) until the
    RMS error is under ``tol``. The model is then made passive (:func:`make_passive`).
    S is fitted rather than Y: it is bounded (|S| ≤ 1), where a nearly lossless line's Y
    is nearly singular in its real part, so a small error there breaks passivity."""
    keep = freqs > 0
    f, h = freqs[keep], s[keep]
    idx = np.unique(np.linspace(0, len(f) - 1, min(len(f), FIT_POINTS)).round().astype(int))

    def full_error(m):
        m.rms_error = float(np.sqrt(np.mean(np.abs(m(f) - h) ** 2)))
        return m

    if n_poles:
        best = full_error(fit(f[idx], h[idx], n_poles))
    else:
        best = None
        for n in range(8, 97, 8):
            m = full_error(fit(f[idx], h[idx], n))
            if best is None or m.rms_error < best.rms_error:
                best = m
            if m.rms_error < tol:
                break
    best.notes.append(f"{len(best.poles)} poles, RMS error {best.rms_error:.2e}")
    make_passive(best, f)
    return best


def _check_grid(freqs: np.ndarray) -> np.ndarray:
    """DC to twice the band: where the model must stay passive."""
    return np.concatenate([[0.0], np.linspace(freqs[-1] / 4000, freqs[-1], 4000),
                           np.linspace(freqs[-1], 2 * freqs[-1], 800)[1:]])


def passivity(model: RationalModel, freqs: np.ndarray) -> float:
    """The largest singular value of the S model over ``freqs``: above 1 it generates
    power."""
    return float(np.max(np.linalg.svd(model(freqs), compute_uv=False)))


def make_passive(model: RationalModel, freqs: np.ndarray) -> float:
    """Scale the model so its largest singular value (DC to twice the band) is under 1;
    returns the scale (1 if none was needed)."""
    worst = passivity(model, _check_grid(freqs))
    if worst < 1.0:
        return 1.0
    k = 0.999 / worst
    model.residues = model.residues * k
    model.d = model.d * k
    model.notes.append(f"made passive: scaled by {k:.4f} (it peaked at σ = {worst:.4f})")
    return k


def spice_subckt(model: RationalModel, name: str, port_names: Sequence[str] = (),
                 z0: float = 50.0, comment: str = "") -> str:
    """A SPICE subcircuit of the S-parameter model: ``.subckt name p1 … pP ref``.

    Each port is a Thevenin source 2·√Z0·b behind Z0, so that b = (V − Z0·I)/(2√Z0); its
    incident wave a = V/√Z0 − b is a node (1 Ω summing G sources), as is each b = D·a +
    Σ states; each state x = a_j/(s − p) is a capacitor with a conductance. Only R, C, G
    and E cards: any SPICE runs it."""
    p = model.ports
    rz = math.sqrt(z0)
    pins = [f"p{i + 1}" for i in range(p)]
    lines = [f"* {line}" for line in (comment or "KiPI rational macromodel").splitlines()]
    for i, nm in enumerate(port_names):
        lines.append(f"* p{i + 1}: {nm}")
    lines.append(f"* {len(model.poles)} poles; S-parameters, Z0 = {z0:g} ohm")
    lines.append(f".subckt {name} {' '.join(pins)} ref")

    def g(label, into, ctrl, gm):
        """A current gm·V(ctrl) into node ``into``."""
        if gm != 0.0:
            lines.append(f"G{label} ref {into} {ctrl} ref {gm:.12e}")

    for i in range(p):
        a, b, e = f"a{i + 1}", f"b{i + 1}", f"e{i + 1}"
        lines.append(f"R{e} {pins[i]} {e} {z0:.12g}")
        lines.append(f"E{e} {e} ref {b} ref {2 * rz:.12e}")
        lines.append(f"R{a} {a} ref 1")
        g(f"v_{a}", a, pins[i], 1.0 / rz)
        g(f"b_{a}", a, b, -1.0)
        lines.append(f"R{b} {b} ref 1")
        for j in range(p):
            g(f"d_{b}_{j + 1}", b, f"a{j + 1}", float(model.d[i, j]))
    n = len(model.poles)
    k = 0
    while k < n:
        pole = model.poles[k]
        real = abs(pole.imag) < 1e-12 * max(abs(pole), 1.0)
        c = 1.0 / abs(pole)
        for j in range(p):
            aj = f"a{j + 1}"
            if real:
                x = f"x{k}_{j + 1}"
                lines.append(f"C{x} {x} ref {c:.12e}")
                lines.append(f"R{x} {x} ref {1.0 / (-pole.real * c):.12e}")
                g(f"in_{x}", x, aj, c)
                for i in range(p):
                    g(f"o_{x}_{i + 1}", f"b{i + 1}", x, float(model.residues[k, i, j].real))
            else:
                # x = xr + j·xi = a_j/(s − p): s·xr = σ·xr − ω·xi + a_j, s·xi = ω·xr + σ·xi;
                # the pair adds 2·(Re r·xr − Im r·xi).
                sig, om = pole.real, pole.imag
                xr, xi = f"xr{k}_{j + 1}", f"xi{k}_{j + 1}"
                for node in (xr, xi):
                    lines.append(f"C{node} {node} ref {c:.12e}")
                    lines.append(f"R{node} {node} ref {1.0 / (-sig * c):.12e}")
                g(f"in_{xr}", xr, aj, c)
                g(f"w_{xr}", xr, xi, -om * c)
                g(f"w_{xi}", xi, xr, om * c)
                for i in range(p):
                    r = model.residues[k, i, j]
                    g(f"o_{xr}_{i + 1}", f"b{i + 1}", xr, 2 * float(r.real))
                    g(f"o_{xi}_{i + 1}", f"b{i + 1}", xi, -2 * float(r.imag))
        k += 1 if real else 2
    lines.append(f".ends {name}")
    return "\n".join(lines) + "\n"


def spice_sparams(netlist: str, freqs: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """The S-parameters (F, P, P) of a subcircuit :func:`spice_subckt` wrote, by modified
    nodal analysis of its R, C, G and E cards: a check that the netlist is the model."""
    from kipi.si.sparams import s_from_y
    cards = [ln.split() for ln in netlist.splitlines() if ln and ln[0] in "RCGE"]
    head = next(ln.split() for ln in netlist.splitlines() if ln.startswith(".subckt"))
    pins = head[2:-1]
    nodes = {nd: i for i, nd in enumerate(pins)}
    for c in cards:
        for nd in c[1:3] + (c[3:5] if c[0][0] in "GE" else []):
            if nd != "ref" and nd not in nodes:
                nodes[nd] = len(nodes)
    p, n = len(pins), len(nodes)
    es = [c for c in cards if c[0][0] == "E"]
    m = n + len(es) + p                    # nodes, E branch currents, port sources' currents
    g = np.zeros((m, m))                   # the system is g + s·cap
    cap = np.zeros((m, m))

    def at(nd):
        return None if nd == "ref" else nodes[nd]

    def add(mat, r, c, v):
        if r is not None and c is not None:
            mat[r, c] += v
    for c in cards:
        kind = c[0][0]
        n1, n2 = at(c[1]), at(c[2])
        if kind in "RC":
            mat, y = (g, 1.0 / float(c[3])) if kind == "R" else (cap, float(c[3]))
            add(mat, n1, n1, y)
            add(mat, n2, n2, y)
            add(mat, n1, n2, -y)
            add(mat, n2, n1, -y)
        elif kind == "G":                     # current n1 → n2 through it
            gm, c1, c2 = float(c[5]), at(c[3]), at(c[4])
            for nd, sg in ((n1, 1), (n2, -1)):
                add(g, nd, c1, sg * gm)
                add(g, nd, c2, -sg * gm)
    for k, c in enumerate(es):
        br = n + k
        n1, n2, c1, c2 = at(c[1]), at(c[2]), at(c[3]), at(c[4])
        add(g, n1, br, 1)
        add(g, n2, br, -1)
        add(g, br, n1, 1)
        add(g, br, n2, -1)
        add(g, br, c1, -float(c[5]))
        add(g, br, c2, float(c[5]))
    for i in range(p):                       # a 1 V source at each port in turn
        br = n + len(es) + i
        add(g, i, br, 1)
        add(g, br, i, 1)
    rhs = np.zeros((m, p))
    rhs[n + len(es):, :] = np.eye(p)
    out = np.zeros((len(freqs), p, p), complex)
    for fi, f in enumerate(freqs):
        sol = np.linalg.solve(g + 2j * math.pi * f * cap, rhs)
        # The sources' branch currents flow into the pins' nodes: the port admittance.
        out[fi] = -sol[n + len(es):, :]
    return s_from_y(out, z0)
