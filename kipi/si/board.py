"""The board as signal integrity sees it: every net's copper by layer (indexed, to cut
cross-sections anywhere), the dielectrics as frequency-dependent materials, and each
trace's reference planes.

A cut is a line across a trace at its middle, ``window`` mm to each side. Every copper
layer's copper is intersected with it; the pieces (by net) at their layers' heights are
the cross-section's conductors: the trace's own, the nets the caller names (its partner
in a pair, a bus's other lines) as their own conductors, and everything else as the
reference (a quiet neighbour is held at 0 V, as field solvers report it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kipi.core.model.board import BoardSnapshot
from kipi.core.model.stackup import combined_epsilon_r
from kipi.si.materials import DEFAULT_ER, Dielectric, default_tand, laminate
from kipi.si.xsection import GROUND, Band, Cache, Conductor

MASK = Dielectric(3.5, 0.025)


TILE_MM = 4.0                  # the copper's tiles for cutting (SIBoard.pieces)


@dataclass
class Piece:
    """Where a cut crosses one net's copper: x0..x1 along the cut (0 at its middle)."""
    layer: int
    x0: float
    x1: float
    net: str


@dataclass
class Reference:
    layer: int
    net: str
    distance_mm: float                  # between the trace's and the plane's facing faces


def dielectric_bands(stack, default_tand_value: Optional[float] = None,
                     er_override: Optional[float] = None) -> List[Band]:
    """The stackup's dielectrics as wideband-Debye bands, the inner coppers' heights
    filled with the dielectric beside them and the solder masks outside the outer
    coppers. A dielectric's tan δ is the stackup's, else its material's typical one,
    else ``default_tand_value``."""
    bands = []
    cu = stack.copper

    def material(subs: list) -> Dielectric:
        er = er_override or combined_epsilon_r([(d.thickness_mm, d.epsilon_r) for d in subs])
        tds = [(d.thickness_mm, d.loss_tangent) for d in subs if d.loss_tangent > 0]
        name = subs[0].material if subs else ""
        if tds:
            td = sum(t * x for t, x in tds) / sum(t for t, _ in tds)
        elif laminate(name) is not None or default_tand_value is None:
            td = default_tand(name)
        else:
            td = default_tand_value
        if not er:
            lam = laminate(name)
            er = lam[0] if lam else DEFAULT_ER
        return Dielectric(float(er), float(td))

    mats = []
    for upper, lower in zip(cu, cu[1:]):
        subs = [d for d in stack.dielectrics
                if d.above_layer == upper.id and d.below_layer == lower.id]
        m = material(subs)
        mats.append(m)
        bands.append(Band(lower.z_top_mm, upper.z_bottom_mm, m))
    for i, c in enumerate(cu):
        if 0 < i < len(cu) - 1:
            bands.append(Band(c.z_bottom_mm, c.z_top_mm, mats[i]))
    for side in ("top", "bottom"):
        mask = next((c for c in stack.coatings_on(side) if c.kind == "mask"), None)
        if mask is None or not cu:
            continue
        outer = cu[0] if side == "top" else cu[-1]
        if side == "top":
            bands.append(Band(outer.z_bottom_mm, outer.z_top_mm + mask.thickness_mm, MASK))
        else:
            bands.append(Band(outer.z_bottom_mm - mask.thickness_mm, outer.z_top_mm, MASK))
    return bands


class SIBoard:
    """A board's copper indexed for cuts. Built once per board (a few seconds on a large
    one); every cut after that is a spatial query."""

    def __init__(self, snap: BoardSnapshot, default_tand_value: Optional[float] = None,
                 etch_mm: float = 0.0, check_cancel=None):
        import shapely
        from shapely.strtree import STRtree
        from kipi.core.model.nets import classify_nets
        self.snap = snap
        self.stack = snap.stackup
        self.kinds = classify_nets(snap)
        self.etch_mm = etch_mm
        self.layer_ids = [c.id for c in self.stack.copper]
        self.geoms: Dict[int, List[Tuple[str, object]]] = {}
        self.trees: Dict[int, object] = {}
        for lay in self.stack.copper:
            items = []
            for n in snap.net_names():
                if check_cancel:
                    check_cancel()
                g = line_copper(snap, n, lay.id)
                if g is not None and not g.is_empty:
                    shapely.prepare(g)
                    items.append((n, g))
            self.geoms[lay.id] = items
            self.trees[lay.id] = STRtree([g for _, g in items]) if items else None
        self.bands = dielectric_bands(self.stack, default_tand_value)
        zs = [c.z_bottom_mm for c in self.stack.copper] + [c.z_top_mm for c in self.stack.copper]
        self.z_range = (min(zs), max(zs))
        self.cache = Cache()
        self.outline_box = self._outline_box()
        self._tiles = None

    def _outline_box(self):
        pts = [p for poly in self.snap.outline for p in poly.outline]
        if not pts:
            pts = [p for t in self.snap.tracks for p in (t.start, t.end)] or [(0, 0), (1, 1)]
        a = np.asarray(pts)
        return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())

    def is_reference(self, net: str) -> bool:
        return self.kinds.get(net) in ("ground", "power")

    def layer_index(self, layer_id: int) -> int:
        return self.layer_ids.index(layer_id)

    # ------------------------------------------------------------------ cuts
    def _tile_index(self):
        """The copper cut into squares of :data:`TILE_MM` (a cut then meets small polygons,
        not a whole pour), per layer: (items, STRtree)."""
        if self._tiles is not None:
            return self._tiles
        import shapely
        from shapely.strtree import STRtree
        self._tiles = {}
        self._tile_geoms = {}
        for lid, items in self.geoms.items():
            out = []
            for net, g in items:
                x0, y0, x1, y1 = g.bounds
                if max(x1 - x0, y1 - y0) <= TILE_MM:
                    out.append((net, g))
                    continue
                xs = np.arange(math.floor(x0 / TILE_MM), math.ceil(x1 / TILE_MM)) * TILE_MM
                ys = np.arange(math.floor(y0 / TILE_MM), math.ceil(y1 / TILE_MM)) * TILE_MM
                boxes = shapely.box(*(np.meshgrid(xs, ys)[k].ravel() + d
                                      for k, d in ((0, 0), (1, 0), (0, TILE_MM), (1, TILE_MM))))
                hit = STRtree(boxes).query(g, predicate="intersects")
                for part in shapely.intersection(g, boxes[hit]):
                    if not part.is_empty and part.area > 0:
                        out.append((net, part))
            self._tiles[lid] = (out, STRtree([g for _, g in out]) if out else None)
            self._tile_geoms[lid] = np.array([g for _, g in out], dtype=object)
        return self._tiles

    def pieces(self, mid, normal, window: float) -> List[Piece]:
        """Every net's copper crossing the cut through ``mid`` along ``normal``."""
        return self.pieces_many([mid], [normal], window)[0]

    def pieces_many(self, mids, normals, window: float) -> List[List[Piece]]:
        """:meth:`pieces` of many cuts at once (one tree query and one vectorised
        intersection per layer for them all)."""
        import shapely
        m = np.asarray(mids, float).reshape(-1, 2)
        nv = np.asarray(normals, float).reshape(-1, 2)
        ends = np.stack([m - window * nv, m + window * nv], axis=1)
        cuts = shapely.linestrings(ends)
        out: List[List[Piece]] = [[] for _ in range(len(m))]
        for lid, (items, tree) in self._tile_index().items():
            if tree is None:
                continue
            ci, ti = tree.query(cuts, predicate="intersects")
            if not len(ci):
                continue
            geoms = self._tile_geoms[lid]
            inter = shapely.intersection(cuts[ci], geoms[ti])
            parts, src = shapely.get_parts(inter, return_index=True)
            line = (shapely.get_type_id(parts) == 1) & (shapely.length(parts) >= 1e-6)
            parts, src = parts[line], src[line]
            if not len(parts):
                continue
            xy, owner = shapely.get_coordinates(parts, return_index=True)
            cut_of = ci[src][owner]
            sv = ((xy[:, 0] - m[cut_of, 0]) * nv[cut_of, 0]
                  + (xy[:, 1] - m[cut_of, 1]) * nv[cut_of, 1])
            starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
            lo = np.minimum.reduceat(sv, starts)
            hi = np.maximum.reduceat(sv, starts)
            spans: Dict[tuple, list] = {}
            for j, k in enumerate(owner[starts]):
                c = int(ci[src[k]])
                spans.setdefault((c, items[int(ti[src[k]])][0]), []).append(
                    (float(lo[j]), float(hi[j])))
            for (c, net), sp in spans.items():
                # Pieces of one net the tiles' edges split are one piece again.
                sp.sort()
                a, b = sp[0]
                for x0, x1 in sp[1:]:
                    if x0 <= b + 1e-6:
                        b = max(b, x1)
                    else:
                        out[c].append(Piece(lid, a, b, net))
                        a, b = x0, x1
                out[c].append(Piece(lid, a, b, net))
        return out

    def section(self, mid, normal, layer: int, width_mm: float, net: str, window: float,
                partners: Optional[Dict[str, Tuple[int, float]]] = None,
                pieces: Optional[List[Piece]] = None):
        """The cross-section at a trace's middle: (conductors, box, partner names found).
        ``partners`` names nets to give their own conductor: net -> (layer, x expected);
        the piece of that net nearest there is it, its other pieces are reference."""
        pieces = self.pieces(mid, normal, window) if pieces is None else pieces
        lay = self.stack.layer(layer)
        conds = [Conductor.rect(-width_mm / 2, width_mm / 2, lay.z_bottom_mm, lay.z_top_mm, net,
                                self.etch_mm)]
        chosen: Dict[str, Piece] = {}
        for pnet, (play, px) in (partners or {}).items():
            cands = [p for p in pieces if p.net == pnet]
            if not cands:
                continue
            best = min(cands, key=lambda p: (p.layer != play, abs(0.5 * (p.x0 + p.x1) - px)))
            chosen[pnet] = best
        keep = (-width_mm / 2, width_mm / 2)
        for p in pieces:
            if p.net == net and p.layer == layer and p.x0 <= keep[1] and p.x1 >= keep[0]:
                continue                                    # the trace itself
            pl = self.stack.layer(p.layer)
            x0, x1 = p.x0, p.x1
            if p.layer == layer:
                x0, x1 = _clip_out(x0, x1, keep)
                if x1 - x0 < 1e-4:
                    continue
            name = p.net if chosen.get(p.net) is p else GROUND
            if name != GROUND:
                conds.append(Conductor.rect(x0, x1, pl.z_bottom_mm, pl.z_top_mm, name,
                                            self.etch_mm))
            else:
                conds.append(Conductor.rect(x0, x1, pl.z_bottom_mm, pl.z_top_mm))
        margin = max(2.0, window)
        box = (-window, window, self.z_range[0] - margin, self.z_range[1] + margin)
        return conds, box, list(chosen)

    def solve(self, conds, box):
        return self.cache.solve(conds, self.bands, box)

    def references(self, pieces: Sequence[Piece], layer: int, width_mm: float
                   ) -> Tuple[Optional[Reference], Optional[Reference], bool]:
        """The trace's nearest reference above and below (a plane or pour of a power or
        ground net covering the trace's width), and whether a nearer reference copper
        only partly covers it (the trace runs over a gap or a plane's edge)."""
        i = self.layer_index(layer)
        lay = self.stack.copper[i]
        half = width_mm / 2
        partial = False
        found = []
        for direction in (-1, 1):
            ref = None
            j = i + direction
            while 0 <= j < len(self.stack.copper):
                lid = self.layer_ids[j]
                over = [p for p in pieces if p.layer == lid and self.is_reference(p.net)
                        and p.x0 < half and p.x1 > -half]
                full = [p for p in over if p.x0 <= -half - 1e-3 and p.x1 >= half + 1e-3]
                if full:
                    other = self.stack.copper[j]
                    d = (other.z_bottom_mm - lay.z_top_mm if direction < 0
                         else lay.z_bottom_mm - other.z_top_mm)
                    ref = Reference(lid, full[0].net, d)
                    break
                if over:
                    partial = True
                j += direction
            found.append(ref)
        return found[0], found[1], partial


def line_copper(snap: BoardSnapshot, net: str, layer: int):
    """A net's tracks and pours on a layer: the copper a transmission line's cross-section
    is made of. Pads and via rings are left out; they are discontinuities along a line,
    which the channel models as such."""
    import shapely
    from kipi.core.geometry.copper import _arc_points, poly_to_shapely
    from shapely.geometry import LineString
    tracks, _, _, zones = snap.items_on_net(net)
    parts = []
    straight = [t for t in tracks if t.layer == layer and not t.is_arc]
    if straight:
        lines = shapely.linestrings([[t.start, t.end] for t in straight])
        widths = np.array([t.width_mm for t in straight])
        parts.extend(shapely.buffer(lines, widths / 2.0, quad_segs=4, cap_style="flat"))
    for t in tracks:
        if t.layer == layer and t.is_arc:
            pts = [t.start] + _arc_points(t.start, t.mid, t.end)
            parts.append(LineString(pts).buffer(t.width_mm / 2.0, quad_segs=4, cap_style="flat"))
    for z in zones:
        parts += [poly_to_shapely(p) for p in z.filled.get(layer, [])]
    parts = [p for p in parts if p is not None and not p.is_empty]
    return shapely.union_all(parts) if parts else None


def _clip_out(x0, x1, keep):
    k0, k1 = keep
    if x1 <= k0 or x0 >= k1:
        return x0, x1
    if x0 < k0:
        return x0, k0 - 1e-3
    if x1 > k1:
        return k1 + 1e-3, x1
    return 0.0, 0.0


def normal_of(a, b) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """(middle, unit normal, length) of segment a→b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    ln = math.hypot(dx, dy)
    if ln < 1e-12:
        return ((a[0], a[1]), (0.0, 1.0), 0.0)
    return ((0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])), (-dy / ln, dx / ln), ln)
