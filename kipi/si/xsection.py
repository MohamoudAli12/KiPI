"""Multiconductor 2D cross-section solver: frequency-dependent RLGC matrices of a cut.

The cut is a 2D problem, solved quasi-statically with P1 triangles (as
:mod:`kipi.noise.extract.xsection`, which solves one victim). Conductors are polygons
(rectangles, or trapezoids for etched traces); each signal conductor gets its own
potential, everything named :data:`GROUND` (planes, other nets, the box closing the
domain) is the reference.

* **C** (Maxwell capacitance matrix, F/mm): one solve per signal conductor, held at 1 V
  with the others at 0, the charge read off the stiffness matrix's residual.
* **L** = μ0ε0·C0⁻¹, from the same solves in vacuum (TEM).
* **G and dispersion.** C = Φᵀ·K·Φ is a sum over the dielectric bands, C = Σ_b ε_b·W_b,
  where W_b = Φᵀ·K_b·Φ is the field energy the potentials Φ put in band b. With each band's
  complex wideband-Debye ε_b(ω) (:mod:`kipi.si.materials`) this gives C(ω) and
  G(ω) = −ω·Im C(ω) to first order (the field barely moves as ε changes by a few %).
* **R** (skin effect, with the return current in the planes). At high frequency the
  current on a conductor's surface is the TEM mode's surface charge times the speed of
  light, so for a unit current on conductor i the surface current is the vacuum solve's
  charge distribution for charges e_i/c. Its loss Rs·Σ|J|²·ds gives the high-frequency
  resistance matrix, R_hf = Rs(f)·K(f)·Σ_n q_i(n)·q_j(n)/ds_n, current crowding at the
  edges and the return path in the planes included (Wheeler's incremental inductance
  rule says the same, with no derivative to take). The DC resistance ρ/A is joined to it
  as √(R_dc² + 2j·R_hf²) on the diagonal, which tends to R_dc at low frequency and to
  (1+j)·R_hf, internal inductance included, at high frequency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kipi.ac.solver.physics import EPS0_F_PER_MM, MU0_H_PER_MM
from kipi.si.materials import RHO_CU_OHM_MM, Dielectric, Roughness, surface_resistance

GROUND = ""
GRADE = 0.3
REFINE_PASSES = 2


@dataclass(frozen=True)
class Conductor:
    """A conductor's cross-section: polygon points (x across the cut, z up), mm."""
    points: Tuple[Tuple[float, float], ...]
    name: str = GROUND
    rho_ohm_mm: float = RHO_CU_OHM_MM

    @staticmethod
    def rect(x0: float, x1: float, z0: float, z1: float, name: str = GROUND,
             etch_mm: float = 0.0) -> "Conductor":
        """A rectangle; ``etch_mm`` narrows its top by that much on each side (a trapezoid
        trace etched from the top)."""
        e = min(max(etch_mm, 0.0), 0.45 * (x1 - x0))
        return Conductor(((x0, z0), (x1, z0), (x1 - e, z1), (x0 + e, z1)), name)

    @property
    def bounds(self):
        xs = [p[0] for p in self.points]
        zs = [p[1] for p in self.points]
        return min(xs), max(xs), min(zs), max(zs)

    @property
    def area_mm2(self) -> float:
        p = np.asarray(self.points)
        x, z = p[:, 0], p[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))

    @property
    def width_mm(self) -> float:
        x0, x1, _, _ = self.bounds
        return x1 - x0


@dataclass(frozen=True)
class Band:
    """A horizontal dielectric layer between z0 and z1."""
    z0: float
    z1: float
    material: Dielectric


@dataclass
class LineParams:
    """A cut's per-length line constants, for its signal conductors ``names``."""
    names: List[str]
    c: np.ndarray                     # (N, N) F/mm at the dielectrics' nominal εr
    c0: np.ndarray                    # (N, N) F/mm in vacuum
    weights: List[np.ndarray]         # per band, (N, N): C = Σ ε_b·W_b (+ air)
    air: np.ndarray                   # the air's share (ε = 1)
    bands: List[Band]
    r_hf: np.ndarray                  # (N, N) ohm/mm per ohm of surface resistance
    r_dc: np.ndarray                  # (N,) ohm/mm
    triangles: int = 0
    rho_ohm_mm: float = RHO_CU_OHM_MM
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def l(self) -> np.ndarray:  # noqa: E743 - the inductance matrix L, as in RLGC
        """External inductance matrix, H/mm."""
        return MU0_H_PER_MM * EPS0_F_PER_MM * np.linalg.inv(self.c0)

    def c_at(self, f_hz) -> np.ndarray:
        """Complex capacitance C(ω) = Σ ε_b(ω)·W_b (F, N, N): G = −ω·Im."""
        f = np.atleast_1d(np.asarray(f_hz, dtype=float))
        out = np.broadcast_to(self.air, (len(f), self.n, self.n)).astype(complex)
        for b, w in zip(self.bands, self.weights):
            out = out + b.material.eps(f)[:, None, None] * w
        return out

    def rlgc(self, f_hz, roughness: Optional[Roughness] = None):
        """Series impedance Z(ω) = R + jωL and shunt admittance Y(ω) = G + jωC per mm,
        (F, N, N) each."""
        f = np.atleast_1d(np.asarray(f_hz, dtype=float))
        w = 2 * math.pi * f
        rs = surface_resistance(f, self.rho_ohm_mm)
        if roughness is not None:
            rs = rs * roughness.factor(f, self.rho_ohm_mm)
        rac = rs[:, None, None] * self.r_hf
        zint = (1 + 1j) * rac
        d = np.arange(self.n)
        zint[:, d, d] = np.sqrt(self.r_dc[None, :] ** 2 + 2j * rac[:, d, d] ** 2)
        z = zint + 1j * w[:, None, None] * self.l[None]
        y = 1j * w[:, None, None] * self.c_at(f)
        return z, y

    # ------------------------------------------------------ single and pair numbers
    def z0(self, i: int = 0) -> float:
        """Conductor ``i``'s impedance alone (the others held at 0 V), lossless."""
        return math.sqrt(self.l[i, i] / self.c[i, i])

    def er_eff(self, i: int = 0) -> float:
        return float(self.c[i, i] / self.c0[i, i])

    def delay_s_per_mm(self, i: int = 0) -> float:
        return math.sqrt(self.l[i, i] * self.c[i, i])

    def pair(self, i: int = 0, j: int = 1) -> Dict[str, float]:
        """Odd and even mode numbers of conductors i and j (others held at 0 V): Zodd,
        Zeven, Zdiff = 2·Zodd, Zcm = Zeven/2, and each mode's delay per mm."""
        l, c = self.l, self.c
        ls, lm = 0.5 * (l[i, i] + l[j, j]), l[i, j]
        cs, cm = 0.5 * (c[i, i] + c[j, j]), c[i, j]        # cm < 0 (Maxwell)
        zodd = math.sqrt((ls - lm) / (cs - cm))
        zeven = math.sqrt((ls + lm) / (cs + cm))
        return {"zodd": zodd, "zeven": zeven, "zdiff": 2 * zodd, "zcm": zeven / 2,
                "td_odd": math.sqrt((ls - lm) * (cs - cm)),
                "td_even": math.sqrt((ls + lm) * (cs + cm))}

    def crosstalk(self, victim: int, aggressor: int) -> Dict[str, float]:
        """Backward (NEXT) coefficient Kb = ¼(Lm/L + Cm/C) and forward (FEXT) coefficient
        per mm, Kf = ½(Cm/C − Lm/L)·td: the far end's noise is Kf·length·dV/dt."""
        l, c = self.l, self.c
        lr = l[victim, aggressor] / math.sqrt(l[victim, victim] * l[aggressor, aggressor])
        cr = -c[victim, aggressor] / math.sqrt(c[victim, victim] * c[aggressor, aggressor])
        td = self.delay_s_per_mm(victim)
        return {"kb": 0.25 * (lr + cr), "kf_s_per_mm": 0.5 * (cr - lr) * td,
                "k_l": lr, "k_c": cr}

    def loss_db_per_mm(self, f_hz, i: int = 0, roughness: Optional[Roughness] = None):
        """Conductor ``i``'s attenuation alone, dB/mm."""
        z, y = self.rlgc(f_hz, roughness)
        g = np.sqrt(z[:, i, i] * y[:, i, i])
        return 20 / math.log(10) * g.real


# ------------------------------------------------------------------------- mesh
def _ring(points, step: Optional[float]):
    from shapely.geometry import LineString
    pts = list(points) + [points[0]]
    out = []
    for (ax, az), (bx, bz) in zip(pts, pts[1:]):
        n = max(1, int(np.ceil(math.hypot(bx - ax, bz - az) / step))) if step else 1
        for k in range(n):
            t = k / n
            out.append((ax + (bx - ax) * t, az + (bz - az) * t))
    out.append(out[0])
    return LineString(out)


def _distance(xy: np.ndarray, conductors: Sequence[Conductor]) -> np.ndarray:
    d = np.full(len(xy), np.inf)
    for c in conductors:
        x0, x1, z0, z1 = c.bounds
        dx = np.maximum(np.maximum(x0 - xy[:, 0], xy[:, 0] - x1), 0.0)
        dz = np.maximum(np.maximum(z0 - xy[:, 1], xy[:, 1] - z1), 0.0)
        d = np.minimum(d, np.hypot(dx, dz))
    return d


def mesh(conductors: Sequence[Conductor], bands: Sequence[Band], box, fine_mm: float,
         coarse_mm: float, grade: float = GRADE):
    import triangle
    from shapely.geometry import LineString, Polygon
    from kipi.core.mesh.mesher import _linework
    bx0, bx1, bz0, bz1 = box
    active = [c for c in conductors if c.name != GROUND]
    lines = [_ring(((bx0, bz0), (bx1, bz0), (bx1, bz1), (bx0, bz1)), coarse_mm)]
    for b in bands:
        for z in (b.z0, b.z1):
            if bz0 < z < bz1:
                lines.append(LineString([(bx0, z), (bx1, z)]))
    holes = []
    for c in conductors:
        pts = tuple((min(max(x, bx0), bx1), z) for x, z in c.points)
        lines.append(_ring(pts, fine_mm if c.name != GROUND else coarse_mm))
        rp = Polygon(pts).representative_point()
        holes.append((rp.x, rp.y))
    verts, segs = _linework(lines)
    data = {"vertices": verts, "segments": segs}
    if holes:
        data["holes"] = np.asarray(holes)
    out = triangle.triangulate(data, f"pq28a{coarse_mm ** 2 * 0.5:.6g}")
    for _ in range(REFINE_PASSES):
        cent = out["vertices"][out["triangles"]].mean(axis=1)
        size = np.clip(fine_mm + grade * _distance(cent, active), fine_mm, coarse_mm)
        out = triangle.triangulate(
            {**{k: out[k] for k in ("vertices", "triangles", "segments") if k in out},
             **({"holes": data["holes"]} if holes else {}),
             "triangle_max_area": 0.43 * size ** 2}, "rpq28a")
    return out["vertices"], out["triangles"]


def _band_stiffness(p, t):
    """Per-triangle unit-permittivity element matrices, and their scatter indices."""
    x, z = p[t, 0], p[t, 1]
    b = np.stack([z[:, 1] - z[:, 2], z[:, 2] - z[:, 0], z[:, 0] - z[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    area = 0.5 * np.abs(b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    ke = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * area)[:, None, None]
    rows = np.repeat(t, 3, axis=1).ravel()
    cols = np.tile(t, (1, 3)).ravel()
    return ke, rows, cols


def _assemble(ke, rows, cols, eps, n, mask=None):
    from scipy.sparse import coo_matrix
    k = ke * eps[:, None, None]
    if mask is not None:
        k = k * mask[:, None, None]
    return coo_matrix((k.ravel(), (rows, cols)), shape=(n, n)).tocsr()


class _Pattern:
    """A mesh's sparse matrix structure, found once: each assembly is then one weighted
    count into it, not a sort of every entry again."""

    def __init__(self, rows, cols, n):
        lin = rows.astype(np.int64) * n + cols
        uniq, self.inv = np.unique(lin, return_inverse=True)
        self.indices = (uniq % n).astype(np.int32)
        self.indptr = np.searchsorted(uniq // n, np.arange(n + 1)).astype(np.int32)
        self.n = n
        self.nnz = len(uniq)

    def matrix(self, ke, eps, mask=None):
        from scipy.sparse import csr_matrix
        w = eps if mask is None else eps * mask
        data = np.bincount(self.inv, weights=(ke * w[:, None, None]).ravel(), minlength=self.nnz)
        return csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))


def _inside(p, c: Conductor, tol: float) -> np.ndarray:
    """Nodes on or in conductor ``c`` (nodes are never strictly inside: it is a hole)."""
    import shapely
    from shapely.geometry import Polygon
    poly = Polygon(c.points).buffer(tol, join_style=2)
    return shapely.contains_xy(poly, p[:, 0], p[:, 1])


def solve(conductors: Sequence[Conductor], bands: Sequence[Band], box,
          fine_mm: Optional[float] = None, mesh_scale: float = 1.0) -> LineParams:
    """Line constants of every signal conductor (each distinct non-ground name, in order
    of first appearance) in a cut closed by a grounded box (x0, x1, z0, z1)."""
    from scipy.sparse.linalg import splu
    names = list(dict.fromkeys(c.name for c in conductors if c.name != GROUND))
    if not names:
        raise ValueError("cross-section has no signal conductor")
    sig = [c for c in conductors if c.name != GROUND]
    w = min(c.width_mm for c in sig)
    fine = fine_mm or float(np.clip(w / 30, 0.0005, 0.02)) * mesh_scale
    coarse = max((box[1] - box[0]) / 40 * math.sqrt(mesh_scale), 4 * fine)
    p, t = mesh(conductors, bands, box, fine, coarse)
    n = len(p)
    tol = 1e-6
    group = np.full(n, -1)
    bx0, bx1, bz0, bz1 = box
    on_box = ((np.abs(p[:, 0] - bx0) < tol) | (np.abs(p[:, 0] - bx1) < tol)
              | (np.abs(p[:, 1] - bz0) < tol) | (np.abs(p[:, 1] - bz1) < tol))
    group[on_box] = len(names)                     # the reference
    on_cond = np.zeros(n, bool)
    for c in conductors:
        m = _inside(p, c, tol)
        on_cond |= m
        group[m] = names.index(c.name) if c.name != GROUND else len(names)
    free = np.flatnonzero(group < 0)
    fixed = np.flatnonzero(group >= 0)
    ke, rows, cols = _band_stiffness(p, t)
    pat = _Pattern(rows, cols, n)
    cz = p[t, 1].mean(axis=1)
    in_band = [((cz > b.z0) & (cz < b.z1)).astype(float) for b in bands]
    covered = np.clip(sum(in_band) if in_band else np.zeros(len(t)), 0, 1)
    in_air = 1.0 - covered
    eps_real = np.ones(len(t))
    for b, m in zip(bands, in_band):
        er = float(b.material.eps(b.material.f_hz).real)
        eps_real = np.where(m > 0, er, eps_real)
    nn = len(names)
    phi_fixed = np.zeros((len(fixed), nn))
    for i in range(nn):
        phi_fixed[:, i] = (group[fixed] == i)
    out = {}
    for vacuum in (False, True):
        k = pat.matrix(ke, EPS0_F_PER_MM * (np.ones(len(t)) if vacuum else eps_real))
        kff = k[free][:, free].tocsc()
        lu = splu(kff)
        phi = np.zeros((n, nn))
        phi[fixed] = phi_fixed
        phi[free] = lu.solve(-(k[free][:, fixed] @ phi_fixed))
        q = k @ phi
        cmat = np.array([[q[group == i, j].sum() for j in range(nn)] for i in range(nn)])
        out[vacuum] = (0.5 * (cmat + cmat.T), phi, q)
    c, phi, _ = out[False]
    c0, _, q0 = out[True]
    weights = []
    for _b, m in zip(bands, in_band):
        kb = pat.matrix(ke, EPS0_F_PER_MM * np.ones(len(t)), m)
        weights.append(phi.T @ (kb @ phi))
    ka = pat.matrix(ke, EPS0_F_PER_MM * np.ones(len(t)), in_air)
    air = phi.T @ (ka @ phi)
    # Surface current: unit current on each conductor is charge e_i/c in vacuum.
    qn = q0 @ np.linalg.inv(c0)                        # node charge per unit conductor charge
    ds = _boundary_lengths(p, t, on_cond & ~on_box)
    surf = ds > 0
    r_hf = (qn[surf].T / ds[surf]) @ qn[surf]
    rho = {c.name: c.rho_ohm_mm for c in sig}
    area = {nm: sum(c.area_mm2 for c in sig if c.name == nm) for nm in names}
    r_dc = np.array([rho[nm] / area[nm] for nm in names])
    return LineParams(names, c, c0, weights, air, list(bands), 0.5 * (r_hf + r_hf.T), r_dc,
                      len(t), rho_ohm_mm=sig[0].rho_ohm_mm)


def _boundary_lengths(p, t, on_cond) -> np.ndarray:
    """Each conductor-surface node's share of the surface length: half of each boundary
    edge (an edge of one triangle only) between two conductor nodes."""
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e = np.sort(e, axis=1).astype(np.int64)
    n = int(e.max()) + 1
    lin, counts = np.unique(e[:, 0] * n + e[:, 1], return_counts=True)   # 1D: a fast sort
    uniq = np.stack([lin // n, lin % n], axis=1)
    b = uniq[(counts == 1) & on_cond[uniq[:, 0]] & on_cond[uniq[:, 1]]]
    ln = np.hypot(*(p[b[:, 0]] - p[b[:, 1]]).T)
    ds = np.zeros(len(p))
    np.add.at(ds, b[:, 0], ln / 2)
    np.add.at(ds, b[:, 1], ln / 2)
    return ds


class Cache:
    """Cuts already solved, keyed by their geometry: a conductor by its place in the cut
    (not its net's name), clipped to the box, its edges rounded to ``grid_mm`` (a ground
    copper's to ``ground_grid_mm``), and a cut and its mirror image alike. ``mesh_scale``
    above 1 meshes coarser (a quick scan's few % of accuracy for a fraction of the time)."""

    def __init__(self, grid_mm: float = 0.005, size: int = 4096,
                 ground_grid_mm: Optional[float] = None, mesh_scale: float = 1.0):
        self.grid = grid_mm
        self.ground_grid = ground_grid_mm or grid_mm
        self.size = size
        self.mesh_scale = mesh_scale
        self._d: Dict[tuple, LineParams] = {}
        self.hits = self.misses = 0

    def key(self, conductors, bands, box) -> tuple:
        names = list(dict.fromkeys(c.name for c in conductors if c.name != GROUND))
        bx0, bx1 = box[0], box[1]

        def side(sign):
            out = []
            for c in conductors:
                role = names.index(c.name) if c.name != GROUND else -1
                q = self.grid if role >= 0 else self.ground_grid
                xs = [min(max(sign * x, bx0), bx1) for x, _ in c.points]
                zs = [z for _, z in c.points]
                x0, x1 = min(xs), max(xs)
                if x1 - x0 < 1e-9:
                    continue                      # outside the box
                pts = tuple(sorted((round(x / q), round(z, 6)) for x, z in zip(xs, zs)))
                out.append((role, round(x0 / q), round(x1 / q), round(min(zs), 6),
                            round(max(zs), 6), len(pts) if role >= 0 else 0,
                            pts if role >= 0 else ()))
            return tuple(sorted(out))
        geo = min(side(1.0), side(-1.0)) if abs(bx0 + bx1) < 1e-9 else side(1.0)
        return (geo,
                tuple((round(b.z0, 6), round(b.z1, 6), round(b.material.er, 4),
                       round(b.material.tand, 5)) for b in bands),
                tuple(round(v / self.grid) for v in box), len(names))

    def get(self, conductors, bands, box) -> Optional[LineParams]:
        hit = self._d.get(self.key(conductors, bands, box))
        if hit is not None:
            self.hits += 1
            return _renamed(hit, conductors)
        return None

    def put(self, key, res: LineParams):
        if len(self._d) >= self.size:
            self._d.pop(next(iter(self._d)))
        self._d[key] = res

    def solve(self, conductors, bands, box) -> LineParams:
        k = self.key(conductors, bands, box)
        hit = self._d.get(k)
        if hit is not None:
            self.hits += 1
            return _renamed(hit, conductors)
        self.misses += 1
        res = solve(conductors, bands, box, mesh_scale=self.mesh_scale)
        self.put(k, res)
        return res


def _renamed(res: LineParams, conductors) -> LineParams:
    names = list(dict.fromkeys(c.name for c in conductors if c.name != GROUND))
    if names == res.names:
        return res
    import dataclasses
    return dataclasses.replace(res, names=names)
