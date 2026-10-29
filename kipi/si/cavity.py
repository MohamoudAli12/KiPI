"""The impedance a via's return current sees crossing between two reference planes.

When a signal changes layers its return current has to change planes too. Where both
planes are the same net and **stitching vias** join them nearby, the return flows down
the nearest of them: the plane pair is then, below its first resonance, a 2D magnetic
problem whose Green's function between posts of radius r a distance ρ apart is
μ0·h/(2π)·(−ln ρ). The signal via's current I divides among the stitching posts, held at
the planes' common potential, with Σ I_s = −I (a bordered solve), which gives the loop's
inductance; with several stitching vias around, it falls as their number grows.

Where nothing joins the planes (different nets, or no stitching via near), the return is
the plane pair's displacement current: the rectangular cavity's impedance (Okoshi's
modal sum; Lei, Techentin and Gilbert 1999)

    Z(ω) = jωμ0·h/(a·b) · Σ_mn χm²χn²·f_mn² / (k_mn² − k²),  f_mn = cos(k_m x)cos(k_n y)
           ·sinc(k_m w/2)·sinc(k_n w/2),   k² = ω²μ0ε(1 − j(tan δ + δs/h)),

with the board's bounding box as the cavity. The (0, 0) term is the plane capacitance,
1/(jωC); a few hundred modes give the resonances, and the modes left out are put back as
the static log term μ0h/(2π)·ln(ρc/r) of a post in an infinite pair, whose cut-off ρc
(the size the kept modes resolve) was fitted against the full sum.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from kipi.ac.solver.physics import EPS0_F_PER_MM, MU0_H_PER_MM
from kipi.si.materials import RHO_CU_OHM_MM, Dielectric

MAX_MODES = 60                # per direction
RHO_C_FACTOR = 0.47           # ρc = factor·√(ab)/M; fitted against M = 1500 sums
STATIC_MODES = 1_000_000      # most modes in a plane pair's static sum
DYNAMIC_MODES = 40_000        # most in its frequency-dependent part


def stitched_impedance(f_hz: np.ndarray, h_mm: float, via_xy: Tuple[float, float],
                       via_r_mm: float, stitches: Sequence[Tuple[float, float, float]]
                       ) -> np.ndarray:
    """Z (F,) of a via between two planes joined by ``stitches`` ((x, y, radius) each):
    jω·L with L from the posts' 2D Green's function."""
    if not stitches:
        raise ValueError("no stitching via")
    pts = np.array([via_xy] + [(x, y) for x, y, _ in stitches], dtype=float)
    radii = np.array([via_r_mm] + [r for _, _, r in stitches], dtype=float)
    d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, radii)
    g = -np.log(np.maximum(d, 1e-6))
    s = len(stitches)
    # [Gss 1; 1ᵀ 0]·[Is; λ] = [−Gs1; −1]  (stitch potentials equal, currents sum to −1)
    m = np.zeros((s + 1, s + 1))
    m[:s, :s] = g[1:, 1:]
    m[:s, s] = -1.0
    m[s, :s] = 1.0
    rhs = np.concatenate([-g[1:, 0], [-1.0]])
    sol = np.linalg.solve(m, rhs)
    i_s, lam = sol[:s], sol[s]
    v1 = g[0, 0] + g[0, 1:] @ i_s - lam
    l_h = MU0_H_PER_MM * h_mm / (2 * math.pi) * v1
    f = np.maximum(np.asarray(f_hz, dtype=float), 1.0)
    # Copper loss by Wheeler's rule: the planes recede (h grows) and the barrel shrinks.
    rs = np.sqrt(math.pi * f * MU0_H_PER_MM * RHO_CU_OHM_MM)
    r = rs * (max(v1, 0.0) / math.pi + h_mm / (2 * math.pi * via_r_mm))
    return (1 + 1j) * r + 2j * math.pi * f * l_h


def cavity_impedance(f_hz: np.ndarray, a_mm: float, b_mm: float, h_mm: float,
                     xy: Tuple[float, float], via_r_mm: float, dielectric: Dielectric,
                     modes: int = MAX_MODES) -> np.ndarray:
    """Z (F,) at a via (x, y from the cavity's corner) in an a × b plane pair h apart."""
    f = np.maximum(np.asarray(f_hz, dtype=float), 1e3)
    w = 2 * math.pi * f
    eps = dielectric.eps(f)
    delta = np.sqrt(2 * RHO_CU_OHM_MM / (w * MU0_H_PER_MM))
    k2 = w ** 2 * MU0_H_PER_MM * EPS0_F_PER_MM * eps.real * (
        1 - 1j * (-eps.imag / eps.real + delta / h_mm))
    width = math.sqrt(math.pi) * via_r_mm             # a square port of the post's area
    x, y = xy
    m = np.arange(modes)
    km = m * math.pi / a_mm
    kn = m * math.pi / b_mm
    fm = np.cos(km * x) * np.sinc(km * width / 2 / math.pi)
    fn = np.cos(kn * y) * np.sinc(kn * width / 2 / math.pi)
    chi = np.where(m == 0, 1.0, 2.0)
    wm = chi * fm ** 2
    wn = chi * fn ** 2
    kmn2 = km[:, None] ** 2 + kn[None, :] ** 2            # (M, N)
    weight = wm[:, None] * wn[None, :]
    weight_flat = weight.ravel()
    kmn_flat = kmn2.ravel()
    # Σ weight/(kmn² − k²) for each frequency; the (0,0) term is the plane capacitance.
    nz = kmn_flat > 0
    terms = weight_flat[nz][None, :] / (kmn_flat[nz][None, :] - k2[:, None])
    s = terms.sum(axis=1) - weight_flat[0] / k2
    z = 1j * w * MU0_H_PER_MM * h_mm / (a_mm * b_mm) * s
    # The static part of the modes left out: a post in an infinite pair, cut off at ρc.
    rho_c = RHO_C_FACTOR * math.sqrt(a_mm * b_mm) / modes
    if rho_c > via_r_mm:
        z = z + 1j * w * MU0_H_PER_MM * h_mm / (2 * math.pi) * math.log(rho_c / via_r_mm)
    return z


# ---------------------------------------------------------------------------------------
# Several posts in one plane pair: the physics-based via model's parallel-plate block


def _post_modes(posts, a_mm, b_mm, step_mm, k_cut=None):
    """The cavity's modes on a grid ``step_mm`` apart in each direction (or up to
    wavenumber ``k_cut``): each post's modal weights Φ (P, K) and the modes' k² (K,)."""
    mx = max(int(math.ceil(a_mm / step_mm)), 2)
    my = max(int(math.ceil(b_mm / step_mm)), 2)
    km = np.arange(mx) * math.pi / a_mm
    kn = np.arange(my) * math.pi / b_mm
    k2 = (km[:, None] ** 2 + kn[None, :] ** 2).ravel()
    sel = k2 <= k_cut ** 2 if k_cut is not None else np.ones(k2.shape, bool)
    phis = []
    for x, y, r in posts:
        width = math.sqrt(math.pi) * r                # a square port of the post's area
        fx = np.sqrt(np.where(km == 0, 1.0, 2.0)) * np.cos(km * x) * np.sinc(km * width / 2 / math.pi)
        fy = np.sqrt(np.where(kn == 0, 1.0, 2.0)) * np.cos(kn * y) * np.sinc(kn * width / 2 / math.pi)
        phis.append((fx[:, None] * fy[None, :]).ravel()[sel])
    return np.array(phis), k2[sel]


def _wavenumber2(f: np.ndarray, h_mm: float, dielectric: Dielectric) -> np.ndarray:
    w = 2 * math.pi * f
    eps = dielectric.eps(f)
    delta = np.sqrt(2 * RHO_CU_OHM_MM / (w * MU0_H_PER_MM))
    return w ** 2 * MU0_H_PER_MM * EPS0_F_PER_MM * eps.real * (
        1 - 1j * (-eps.imag / eps.real + delta / h_mm))


def _distances(posts) -> np.ndarray:
    """Centre distances between posts, a post's radius on the diagonal."""
    xy = np.array([(x, y) for x, y, _ in posts], dtype=float)
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, [r for _, _, r in posts])
    return np.maximum(d, 1e-6)


def plane_pair_z(f_hz, h_mm: float, posts: Sequence[Tuple[float, float, float]],
                 dielectric: Dielectric, size: Optional[Tuple[float, float]] = None):
    """The parallel-plate impedance matrix between ``posts`` ((x, y, radius), mm) that cross
    a plane pair h apart: Z (F, P, P), with V_p the voltage between the planes at post p
    and I_q the current post q carries from one plane to the other.

    ``size`` (a, b): a rectangular cavity with open (magnetic-wall) edges, the posts at
    (x, y) from its corner, as the modal sum

        Z = jωμh/(ab)·[Σ_K Φ_p Φ_q/k_K² + Σ_K Φ_p Φ_q·k²/(k_K²(k_K² − k²)) − 1/k²],

    the static sum first, on a mode grid fine against the posts' radii and distances (the
    modes left out put back as the log term of :func:`cavity_impedance`), then the
    frequency-dependent rest, which converges as 1/k_K⁴; the last term is the pair's
    capacitance, returned on its own as ``c0`` (F,) so shorting posts can cancel it
    without round-off: ``(Z without it, c0)``.

    ``size`` None: planes going on for ever, Z_pq = (ωμh/4)·H0⁽²⁾(k·ρ_pq), ρ_pp the post's
    radius (no capacitance: ``c0`` None); the real part is the power radiated between the
    planes."""
    f = np.maximum(np.asarray(f_hz, dtype=float), 1e3)
    w = 2 * math.pi * f
    k2 = _wavenumber2(f, h_mm, dielectric)
    d = _distances(posts)
    if size is None:
        from scipy.special import hankel2
        k = np.sqrt(k2)
        k = np.where(k.imag > 0, -k, k)               # outgoing: Im k ≤ 0
        z = (w * MU0_H_PER_MM * h_mm / 4)[:, None, None] * hankel2(0, k[:, None, None] * d[None])
        return z, None
    a_mm, b_mm = size
    r_min = min(r for _, _, r in posts)
    step = max(min(0.1, r_min / 1.5, d.min() / 8), math.sqrt(a_mm * b_mm / STATIC_MODES))
    phi, kk = _post_modes(posts, a_mm, b_mm, step)
    nz = kk > 0
    static = (phi[:, nz] / kk[nz]) @ phi[:, nz].T
    rho_c = RHO_C_FACTOR * step
    static = static + a_mm * b_mm / (2 * math.pi) * np.log(np.maximum(rho_c / d, 1.0))
    k_max = math.sqrt(abs(k2[-1]))
    k_cut = max(8 * k_max, 30 * math.pi / min(a_mm, b_mm))
    k_cut = min(k_cut, math.sqrt(DYNAMIC_MODES * math.pi / (a_mm * b_mm)) * 2)
    phi, kk = _post_modes(posts, a_mm, b_mm, math.pi / k_cut / 2, k_cut)
    nz = kk > 0
    phi, kk = phi[:, nz], kk[nz]
    g = k2[:, None] / (kk[None, :] * (kk[None, :] - k2[:, None]))           # (F, K)
    dyn = np.einsum("fk,pk,qk->fpq", g, phi, phi, optimize=True)
    scale = (1j * w * MU0_H_PER_MM * h_mm / (a_mm * b_mm))[:, None, None]
    c0 = 1j * w * MU0_H_PER_MM * h_mm / (a_mm * b_mm) * (-1.0 / k2)
    return scale * (static[None] + dyn), c0


def shorted(z: np.ndarray, c0: Optional[np.ndarray], keep: int,
            loads: Optional[np.ndarray] = None) -> np.ndarray:
    """Z (F, keep, keep) at the first ``keep`` posts of :func:`plane_pair_z`'s
    ``(z, c0)`` with the rest shorted (V = 0: stitching vias joining the two planes), or,
    with ``loads`` (F, rest), each ended in its impedance (V = −Z_L·I: a decoupling
    capacitor joining planes of two nets). The common term c0 (every post alike) is
    carried by the total current T:
    [Z'ss + Z_L 1; 1ᵀ −1/c0]·[Is; c0·T] = [−Z'sp·Ip; −1ᵀ·Ip]."""
    n_f, n, _ = z.shape
    s = n - keep
    if s == 0:
        return z if c0 is None else z + c0[:, None, None]
    m = np.zeros((n_f, s + 1, s + 1), complex)
    m[:, :s, :s] = z[:, keep:, keep:]
    if loads is not None:
        idx = np.arange(s)
        m[:, idx, idx] += loads
    if c0 is not None:
        m[:, :s, s] = 1.0
        m[:, s, :s] = 1.0
        m[:, s, s] = -1.0 / c0
    else:
        m[:, s, s] = 1.0                       # no common term: u = 0
    rhs = np.zeros((n_f, s + 1, keep), complex)
    rhs[:, :s, :] = -z[:, keep:, :keep]
    if c0 is not None:
        rhs[:, s, :] = -1.0
    sol = np.linalg.solve(m, rhs)
    out = z[:, :keep, :keep] + z[:, :keep, keep:] @ sol[:, :s, :]
    if c0 is not None:
        out = out + sol[:, s:s + 1, :]          # + u on every row
    return out
