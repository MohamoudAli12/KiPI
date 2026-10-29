"""A signal via as a circuit: its barrel's capacitance to each plane it passes, its pads,
and the return current's path between the planes (the physics-based via model of
Zhang, Fan, Drewniak et al., 2010).

* **Capacitances** come from an axisymmetric electrostatic solve (r, z) of the via: the
  barrel and its pads at 1 V, each plane (copper from its antipad outward) at 0 V, the
  stackup's dielectrics between. Each plane's charge is the barrel-to-plane capacitance
  at that plane, pads' fringing and the antipad's size included.
* **Series impedance.** Between two planes the barrel's current returns through the
  plane pair (:mod:`kipi.si.cavity`): the parallel-plate impedance matrix of every via of
  the channel crossing that pair, with the stitching vias near them shorted: a stitched
  pair is about an inductance, an unstitched one a cavity with resonances, and a pair's
  two vias are coupled through the planes they share. Each piece of barrel between two
  copper layers takes the share of the pair's impedance its height is of the pair's.
* **Stubs.** The barrel goes on past the layer the signal leaves on, to the via's last
  layer: that piece is the same chain, left open at its end, so its quarter-wave
  resonance comes out of the circuit by itself (a back-drilled via has no stub).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence

import numpy as np

from kipi.ac.solver.physics import EPS0_F_PER_MM, MU0_H_PER_MM
from kipi.si.xsection import Band, Conductor, _assemble, _band_stiffness, mesh


@dataclass
class ViaLayer:
    """One copper layer the barrel passes."""
    z_bottom: float
    z_top: float
    pad_mm: float = 0.0                 # pad diameter here, 0: none
    antipad_r_mm: Optional[float] = None    # a reference plane here: its clearance radius
    name: str = ""

    @property
    def is_plane(self) -> bool:
        return self.antipad_r_mm is not None


@dataclass
class PlanePair:
    """Two consecutive reference planes a via crosses."""
    layer_ids: tuple                    # (upper, lower) copper layer ids
    nets: tuple
    h_mm: float
    material: object                    # the dielectric between (kipi.si.materials)
    stitches: List[tuple] = field(default_factory=list)   # (x, y, r) shorting posts
    # Planes of two nets: the capacitors joining them, (x, y, r, C, ESR, ESL), each a post
    # ended in its own impedance.
    decaps: List[tuple] = field(default_factory=list)


@dataclass
class ViaStack:
    drill_mm: float
    layers: List[ViaLayer]              # top to bottom, the barrel's whole span
    bands: List[Band]
    # The return path's impedance (F,) between consecutive planes i < j (layer indices).
    cavity: Dict[tuple, Callable[[np.ndarray], np.ndarray]] = field(default_factory=dict)
    # What each such pair is, to couple vias crossing the same one (kipi.si.extract).
    pairs: Dict[tuple, "PlanePair"] = field(default_factory=dict)

    @property
    def r_mm(self) -> float:
        return self.drill_mm / 2

    def plane_indices(self) -> List[int]:
        return [i for i, l in enumerate(self.layers) if l.is_plane]


def capacitances(stack: ViaStack) -> np.ndarray:
    """Barrel-to-plane capacitance at each layer (F), from an axisymmetric solve: 0 where
    the layer is no plane."""
    from scipy.sparse.linalg import splu
    lay = stack.layers
    planes = stack.plane_indices()
    out = np.zeros(len(lay))
    if not planes:
        return out
    r = stack.r_mm
    z_top = max(l.z_top for l in lay)
    z_bot = min(l.z_bottom for l in lay)
    big = max([l.antipad_r_mm for l in lay if l.is_plane] + [l.pad_mm / 2 for l in lay])
    rmax = max(3 * big, big + 1.0)
    box = (0.0, rmax, z_bot - 1.0, z_top + 1.0)
    conds = [Conductor.rect(0.0, r, z_bot, z_top, "via")]
    for l in lay:
        if l.pad_mm > 0:
            conds.append(Conductor.rect(0.0, l.pad_mm / 2, l.z_bottom, l.z_top, "via"))
    for i in planes:
        l = lay[i]
        conds.append(Conductor.rect(min(l.antipad_r_mm, rmax * 0.9), rmax, l.z_bottom, l.z_top,
                                    f"plane{i}"))
    fine = max(min(r / 6, 0.02), 0.003)
    p, t = mesh(conds, stack.bands, box, fine, max(rmax / 25, 4 * fine))
    n = len(p)
    import shapely
    from shapely.geometry import Polygon
    group = np.full(n, -1)
    tol = 1e-6
    far = (np.abs(p[:, 0] - rmax) < tol) | (np.abs(p[:, 1] - box[2]) < tol) \
        | (np.abs(p[:, 1] - box[3]) < tol)
    group[far] = 0
    names = ["far", "via"] + [f"plane{i}" for i in planes]
    for c in conds:
        poly = Polygon(c.points).buffer(tol, join_style=2)
        group[shapely.contains_xy(poly, p[:, 0], p[:, 1])] = names.index(c.name)
    ke, rows, cols = _band_stiffness(p, t)
    cz = p[t, 1].mean(axis=1)
    cr = p[t, 0].mean(axis=1)
    eps = np.ones(len(t))
    for b in stack.bands:
        eps = np.where((cz > b.z0) & (cz < b.z1), b.material.er, eps)
    k = _assemble(ke, rows, cols, EPS0_F_PER_MM * eps * 2 * math.pi * cr, n)
    free = np.flatnonzero(group < 0)
    fixed = np.flatnonzero(group >= 0)
    phi = np.zeros(n)
    phi[fixed] = (group[fixed] == 1)
    phi[free] = splu(k[free][:, free].tocsc()).solve(-(k[free][:, fixed] @ phi[fixed]))
    q = k @ phi
    for j, i in enumerate(planes):
        out[i] = max(-q[group == 2 + j].sum(), 0.0)
    return out


def wire_inductance(h_mm: float, r_mm: float) -> float:
    """A barrel piece with no plane pair around it: a short post over its return,
    μ0·h/(2π)·ln(1 + 2h/r), henries."""
    return MU0_H_PER_MM * h_mm / (2 * math.pi) * math.log(1 + 2 * h_mm / max(r_mm, 1e-3))


def barrel_resistance(f: np.ndarray, h_mm: float, r_mm: float) -> np.ndarray:
    """The barrel's own skin-effect resistance over height h (Ω, (F,)), as (1 + j)·R."""
    from kipi.si.materials import RHO_CU_OHM_MM
    rs = np.sqrt(math.pi * np.maximum(f, 1.0) * MU0_H_PER_MM * RHO_CU_OHM_MM)
    return (1 + 1j) * rs * h_mm / (2 * math.pi * r_mm)


def segment_shares(stack: ViaStack, pa: int, pb: int) -> List[tuple]:
    """The barrel's pieces between planes ``pa`` and ``pb``: (i, share of the pair's
    height) for the piece from layer i to i + 1."""
    lay = stack.layers
    span = max(lay[pa].z_bottom - lay[pb].z_top, 1e-6)
    return [(i, max(lay[i].z_bottom - lay[i + 1].z_top, 1e-6) / span) for i in range(pa, pb)]


def add_coupled_segments(ckt, z: Callable[[np.ndarray], np.ndarray], share: float,
                         tops: Sequence[Hashable], bottoms: Sequence[Hashable],
                         label: str = "via barrels") -> None:
    """Pieces of N barrels in one plane pair, from ``tops`` to ``bottoms``, with their
    coupled impedance share·z(f) (F, N, N)."""
    n = len(tops)

    def y(f):
        yy = np.linalg.inv(share * z(f))
        out = np.empty((len(f), 2 * n, 2 * n), complex)
        out[:, :n, :n] = out[:, n:, n:] = yy
        out[:, :n, n:] = out[:, n:, :n] = -yy
        return out
    ckt.add_block([*tops, *bottoms], y, label)


def add_via_chain(ckt, stack: ViaStack, keys: Sequence[Hashable],
                  caps: Optional[np.ndarray] = None, skip: Sequence[tuple] = ()) -> None:
    """Put a via's barrel into circuit ``ckt``: ``keys`` are its nodes at each of its
    layers (top to bottom); a trace joins it at the node of its layer, and the barrel
    past the last layer a trace uses is a stub, open at its end. The plane pairs in
    ``skip`` are left out (put in by :func:`add_coupled_segments`)."""
    lay = stack.layers
    caps = capacitances(stack) if caps is None else caps
    planes = stack.plane_indices()
    for i in range(len(lay) - 1):
        gap = max(lay[i].z_bottom - lay[i + 1].z_top, 1e-6)
        above = [p for p in planes if p <= i]
        below = [p for p in planes if p >= i + 1]
        if above and below:
            pa, pb = above[-1], below[0]
            if (pa, pb) in skip:
                continue
            z_pair = stack.cavity.get((pa, pb))
            span = max(lay[pa].z_bottom - lay[pb].z_top, 1e-6)
            if z_pair is not None:
                ckt.add_z(keys[i], keys[i + 1],
                          lambda f, z=z_pair, s=gap / span: 1e-4 + s * z(f), "via barrel")
                continue
        ckt.add_l(keys[i], keys[i + 1], wire_inductance(gap, stack.r_mm), 1e-4, "via barrel")
    for i in range(len(lay)):
        if caps[i] > 0:
            ckt.add_c(keys[i], None, float(caps[i]), "via to plane")


def add_via(ckt, stack: ViaStack, enter: int, leave: int, node_in: Hashable,
            node_out: Hashable, prefix: Hashable, caps: Optional[np.ndarray] = None) -> None:
    """A via the signal enters at layer index ``enter`` (node ``node_in``) and leaves at
    ``leave`` (``node_out``); its other layers are stubs."""
    keys = [node_in if i == enter else node_out if i == leave else (prefix, i)
            for i in range(len(stack.layers))]
    if enter == leave:
        ckt.add_r(node_in, node_out, 1e-6, "via")
    add_via_chain(ckt, stack, keys, caps)
