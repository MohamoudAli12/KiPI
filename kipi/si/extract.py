"""A channel's nets as a circuit (:class:`kipi.si.network.Circuit`).

**Routes.** Each net's tracks are cut into straight segments of at most ``segment_mm``
(arcs as chords). Their ends are joined where they meet (within a few µm, on a layer),
where they reach a pad (inside its copper) or a via (within its pad), and a track that
ends on the middle of another splits it there (a T).

**Coupled sections.** The nets are coupled to the first one (the reference: a pair's
P line, a victim). For each of its segments, the cut across its middle finds the other
nets' parallel segments; the reference segment's ends projected onto each gives the
matching stretch of that segment, which is split out of it. The reference segment and
those stretches are one N-conductor line section, their RLGC matrices from the cut; what
is left of the other nets' segments are single lines. So a pair's coupled run, its
uncoupled breakouts and the extra length of a serpentine all come out of one circuit, and
so the skew does.

**Vias** are barrels through the stack (:mod:`kipi.si.via`): a node at each copper layer
they span, the barrel's capacitance to each plane, and the return path's impedance
between planes: stitching vias of the plane's net nearby make it an inductance, without
them it is the plane pair's cavity. A via's unused layers are stubs.

**Series parts.** A two-pin resistor, capacitor, inductor or ferrite whose pins are on
two of the channel's nets (an AC-coupling capacitor, a series terminator) joins them,
with the value read from its footprint's value field.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from kipi.core.model.board import BoardSnapshot, Pad, Via
from kipi.si.board import SIBoard, normal_of
from kipi.si.cavity import plane_pair_z, shorted
from kipi.si.materials import Roughness
from kipi.si.network import Circuit
from kipi.si.via import (
    PlanePair, ViaLayer, ViaStack, add_coupled_segments, add_via_chain, barrel_resistance,
    capacitances, segment_shares,
)

SNAP_MM = 0.005
STITCH_RADIUS_MM = 5.0
MAX_STITCHES = 8
MAX_POSTS = 24                 # stitching vias in one plane pair's block
# Decoupling capacitors joining a plane pair of two nets (GND and a supply): the nearest
# this many, each as a post at the part with its ESR and its inductance with its mounting
# (pads, vias, the loop to the planes) and a via's radius.
MAX_DECAPS = 16
DECAP_ESR_OHM = 0.01
DECAP_ESL_H = 1.0e-9
DECAP_POST_R_MM = 0.15
PARALLEL_COS = 0.97


@dataclass
class Seg:
    net: str
    layer: int
    a: Tuple[float, float]
    b: Tuple[float, float]
    width_mm: float
    na: Hashable = None                 # node keys
    nb: Hashable = None

    @property
    def length_mm(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])

    def unit(self):
        ln = self.length_mm or 1.0
        return ((self.b[0] - self.a[0]) / ln, (self.b[1] - self.a[1]) / ln)

    def project(self, p) -> float:
        """Where ``p`` falls along the segment, 0 at a and 1 at b."""
        ux, uy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        l2 = ux * ux + uy * uy or 1.0
        return ((p[0] - self.a[0]) * ux + (p[1] - self.a[1]) * uy) / l2

    def at(self, t: float):
        return (self.a[0] + (self.b[0] - self.a[0]) * t, self.a[1] + (self.b[1] - self.a[1]) * t)


@dataclass
class SectionInfo:
    """What the circuit's line sections were, for the impedance profile and the 3D view."""
    nets: List[str]
    segs: List[Seg]
    z0: List[float]                    # each conductor's own Z0 (others at 0 V)
    zdiff: Optional[float] = None       # a coupled pair's
    length_mm: float = 0.0
    delay_s: float = 0.0


@dataclass
class Channel:
    """A built channel circuit and what was found building it."""
    circuit: Circuit
    nets: List[str]
    sections: List[SectionInfo] = field(default_factory=list)
    vias: List[str] = field(default_factory=list)
    pads: Dict[str, Hashable] = field(default_factory=dict)        # pad id -> node key
    series: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def net_length(self, net: str) -> float:
        return sum(s.length_mm for sec in self.sections for s in sec.segs if s.net == net)

    def net_delay(self, net: str) -> float:
        out = 0.0
        for sec in self.sections:
            for s in sec.segs:
                if s.net == net:
                    out += sec.delay_s * (s.length_mm / max(sec.length_mm, 1e-9))
        return out


# ------------------------------------------------------------------- routes
def clip_to_box(p, q, box):
    """The part of segment p–q inside ``box`` (x0, y0, x1, y1), or None (Liang–Barsky)."""
    x0, y0, x1, y1 = box
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for d, lo, hi, a in ((dx, x0, x1, p[0]), (dy, y0, y1, p[1])):
        if abs(d) < 1e-12:
            if a < lo or a > hi:
                return None
            continue
        ta, tb = (lo - a) / d, (hi - a) / d
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 >= t1:
            return None
    return (p[0] + dx * t0, p[1] + dy * t0), (p[0] + dx * t1, p[1] + dy * t1)


# Chords to a full turn of an arc in a channel: 30° each. Each chord is a section with its
# own field solve, and a bend's few tenths of a mm need no more at any frequency swept; the
# chords are 1.1 % shorter than the arc.
SI_ARC_SEGMENTS_PER_CIRCLE = 12


# A coupled section's conductor this much (a share) longer than the shortest goes on alone
# for the rest; less is taken as the same length.
TAIL_MIN_SHARE = 0.05


def net_segments(snap: BoardSnapshot, net: str, segment_mm: float,
                 region: Optional[Tuple[float, float, float, float]] = None) -> List[Seg]:
    """``net``'s tracks cut into segments at most ``segment_mm`` long; with ``region``,
    only their parts inside it."""
    from kipi.core.geometry.arcs import arc_points
    tracks, _, _, _ = snap.items_on_net(net)
    out = []
    for t in tracks:
        pts = [t.start] + (arc_points(t.start, t.mid, t.end,
                                      segments_per_circle=SI_ARC_SEGMENTS_PER_CIRCLE)
                           if t.is_arc else [t.end])
        for p, q in zip(pts, pts[1:]):
            if region is not None:
                inside = clip_to_box(p, q, region)
                if inside is None:
                    continue
                p, q = inside
            length = math.hypot(q[0] - p[0], q[1] - p[1])
            if length < 1e-6:
                continue
            n = max(1, int(math.ceil(length / segment_mm)))
            for k in range(n):
                a = (p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n)
                b = (p[0] + (q[0] - p[0]) * (k + 1) / n, p[1] + (q[1] - p[1]) * (k + 1) / n)
                out.append(Seg(net, t.layer, a, b, t.width_mm))
    return _split_tees(out)


def _split_tees(segs: List[Seg]) -> List[Seg]:
    """Split a segment where another one's end lands on its middle."""
    keys = {}
    for s in segs:
        for p in (s.a, s.b):
            k = (s.layer, round(p[0] / SNAP_MM), round(p[1] / SNAP_MM))
            keys[k] = keys.get(k, 0) + 1
    dangling = [(s.layer, p) for s in segs for p in (s.a, s.b)
                if keys[(s.layer, round(p[0] / SNAP_MM), round(p[1] / SNAP_MM))] == 1]
    if not dangling:
        return segs
    out = []
    for s in segs:
        cuts = []
        for lay, p in dangling:
            if lay != s.layer:
                continue
            t = s.project(p)
            if 1e-3 < t < 1 - 1e-3:
                q = s.at(t)
                if math.hypot(q[0] - p[0], q[1] - p[1]) <= s.width_mm / 2 + SNAP_MM:
                    cuts.append(t)
        if not cuts:
            out.append(s)
            continue
        ts = [0.0] + sorted(set(cuts)) + [1.0]
        for t0, t1 in zip(ts, ts[1:]):
            out.append(Seg(s.net, s.layer, s.at(t0), s.at(t1), s.width_mm))
    return out


class _Nodes:
    """Node keys for the segments' ends: a pad's, a via layer's, or a shared point's."""

    def __init__(self, snap: BoardSnapshot, nets: Sequence[str]):
        self.pads: List[Pad] = []
        self.vias: List[Via] = []
        for n in nets:
            _, vias, pads, _ = snap.items_on_net(n)
            self.pads += [p for p in pads if p.pad_type != "npth"]
            self.vias += vias
        self._pad_geo = {}
        self.used_vias: Dict[str, set] = {}
        self.used_pads: Dict[str, Pad] = {}

    def _pad_contains(self, pad: Pad, layer: int, p) -> bool:
        import shapely
        from kipi.core.geometry.copper import poly_to_shapely
        key = (pad.id, layer)
        g = self._pad_geo.get(key)
        if g is None:
            polys = [poly_to_shapely(x) for x in pad.shapes.get(layer, [])]
            polys = [x for x in polys if x is not None and not x.is_empty]
            g = shapely.union_all(polys).buffer(SNAP_MM) if polys else False
            self._pad_geo[key] = g
        if g is False:
            return False
        from shapely.geometry import Point
        return g.contains(Point(p))

    def key(self, net: str, layer: int, p) -> Hashable:
        for v in self.vias:
            if v.net == net and layer in v.layers:
                r = max(v.pad_diameter(layer), v.drill_mm) / 2 + SNAP_MM
                if math.hypot(p[0] - v.position[0], p[1] - v.position[1]) <= r:
                    self.used_vias.setdefault(v.id, set()).add(layer)
                    return ("via", v.id, layer)
        for pad in self.pads:
            if pad.net == net and layer in pad.shapes and self._pad_contains(pad, layer, p):
                self.used_pads[pad.id] = pad
                return ("pad", pad.id)
        return ("pt", net, layer, round(p[0] / SNAP_MM), round(p[1] / SNAP_MM))


# ---------------------------------------------------------------- coupling
# A piece of a reference segment this much (a share of it) or shorter, beside other
# segments than its neighbour, is taken into the neighbour.
ABSORB_SHARE = 0.25
# Another net's segments beside a reference segment more than this many times as far
# off as its nearest one are passed over.
FAR_SHARE = 1.5
# A stretch of another net's segment beside nothing, this much of a segment's length or
# shorter, goes in with the stretch beside the reference next to it (as its tail).
ABSORB_MM_SHARE = 0.5


def _runs(segs: List[Seg], plain: Optional[Callable[[str, int, tuple], bool]]) -> List[Seg]:
    """``segs`` with straight runs joined back into one segment: where two of the same
    width meet end to end in line, at a point ``plain`` says joins nothing else."""
    if plain is None:
        return list(segs)
    ends: Dict[Hashable, List[Tuple[Seg, int]]] = {}
    for s in segs:
        for e, p in ((0, s.a), (1, s.b)):
            ends.setdefault((s.layer, round(p[0] / SNAP_MM), round(p[1] / SNAP_MM)),
                            []).append((s, e))

    def joinable(k):
        (s, es), (t, et) = ends[k]
        if s is t or s.width_mm != t.width_mm:
            return False
        su, tu = s.unit(), t.unit()
        sign = 1 if es != et else -1          # b meets a: same way round
        if sign * (su[0] * tu[0] + su[1] * tu[1]) < 0.9999:
            return False
        return plain(s.net, s.layer, s.b if es else s.a)

    joints = {k for k, v in ends.items() if len(v) == 2 and joinable(k)}
    done, out = set(), []
    for s in segs:
        if id(s) in done:
            continue
        chain = [(s, 0)]                     # (segment, reversed?)
        done.add(id(s))
        for forward in (True, False):        # grow from b, then from a
            cur, rev = s, 0
            while True:
                end = (cur.a if rev else cur.b) if forward else (cur.b if rev else cur.a)
                k = (cur.layer, round(end[0] / SNAP_MM), round(end[1] / SNAP_MM))
                if k not in joints:
                    break
                nxt = [(t, e) for t, e in ends[k] if t is not cur]
                t, e = nxt[0]
                if id(t) in done:
                    break
                done.add(id(t))
                # Leaving the joint from t's end e: t runs on reversed if e is its b.
                rev = int(e == 1) if forward else int(e == 0)
                chain.insert(len(chain) if forward else 0, (t, rev))
                cur = t
        first, r0 = chain[0]
        last, r1 = chain[-1]
        a = first.b if r0 else first.a
        b = last.a if r1 else last.b
        out.append(Seg(s.net, s.layer, a, b, s.width_mm))
    return out


def _chunks(s: Seg, segment_mm: float) -> List[Seg]:
    n = max(1, int(math.ceil(s.length_mm / segment_mm - 1e-9)))
    return [Seg(s.net, s.layer, s.at(k / n), s.at((k + 1) / n), s.width_mm) for k in range(n)]


def _pair_up(ref: List[Seg], others: Dict[str, List[Seg]], window: float, board: SIBoard,
             segment_mm: float = 1.0,
             plain: Optional[Callable[[str, int, tuple], bool]] = None
             ) -> Tuple[List[Tuple[Seg, Dict[str, Seg]]], Dict[str, List[Seg]]]:
    """The reference segments, cut where the nets beside them change, each piece with the
    stretch of each other net's segment it runs beside; and the other nets' segments that
    are left (cut no longer than ``segment_mm``). The other nets' straight runs are joined
    first (where ``plain`` says a joint joins nothing else), so they are cut where the
    reference's segments are and not where their own were."""
    other_list = [s for segs in others.values() for s in _runs(segs, plain)]
    pieces: List[Tuple[Seg, float, float, Dict[str, int]]] = []    # r, u0, u1, net -> s
    for r in ref:
        mid, nrm, ln = normal_of(r.a, r.b)
        if ln <= 0:
            continue
        ux, uy = r.unit()
        beside = []                          # (lo, hi, d, net, s index)
        for k, s in enumerate(other_list):
            sx, sy = s.unit()
            if abs(ux * sx + uy * sy) < PARALLEL_COS:
                continue
            lo, hi = sorted((r.project(s.a), r.project(s.b)))
            lo, hi = max(lo, 0.0), min(hi, 1.0)
            if (hi - lo) * ln < 1e-3:
                continue
            m = r.at(0.5 * (lo + hi))
            q = s.at(min(max(s.project(m), 0.0), 1.0))
            d = abs((q[0] - m[0]) * nrm[0] + (q[1] - m[1]) * nrm[1])
            if d <= window:
                beside.append((lo, hi, d, s.net, k))
        # A net's segments much farther off than its nearest are the far side of a
        # meander, not the partner running beside it.
        nearest: Dict[str, float] = {}
        for b in beside:
            nearest[b[3]] = min(nearest.get(b[3], b[2]), b[2])
        beside = [b for b in beside if b[2] <= FAR_SHARE * nearest[b[3]] + 0.02]
        us = sorted({0.0, 1.0, *(b[0] for b in beside), *(b[1] for b in beside)})
        cut = []
        for u0, u1 in zip(us, us[1:]):
            if u1 - u0 <= 1e-9:
                continue
            um = 0.5 * (u0 + u1)
            near: Dict[str, Tuple[float, int]] = {}
            for lo, hi, d, net, k in beside:
                if lo <= um <= hi and (net not in near or d < near[net][0]):
                    near[net] = (d, k)
            cut.append([u0, u1, {n: k for n, (_, k) in near.items()}])
        # A stretch with nothing beside it goes in with its coupled neighbour: past the
        # shortest conductor's end a section's longer ones go on alone (a tail) anyway.
        if any(c[2] for c in cut):
            k = 0
            while k < len(cut):
                if cut[k][2]:
                    k += 1
                    continue
                if k > 0:
                    cut[k - 1][1] = cut[k][1]
                else:
                    cut[k + 1][0] = cut[k][0]
                del cut[k]
        while len(cut) > 1:
            j = min(range(len(cut)), key=lambda j: cut[j][1] - cut[j][0])
            if (cut[j][1] - cut[j][0]) > ABSORB_SHARE:
                break
            nb = [x for x in (j - 1, j + 1) if 0 <= x < len(cut)]
            into = max(nb, key=lambda x: (len(cut[x][2]), cut[x][1] - cut[x][0]))
            cut[into][0] = min(cut[into][0], cut[j][0])
            cut[into][1] = max(cut[into][1], cut[j][1])
            del cut[j]
        # Neighbours beside the same segments are one piece.
        merged = [cut[0]]
        for c in cut[1:]:
            if c[2] == merged[-1][2]:
                merged[-1][1] = c[1]
            else:
                merged.append(c)
        pieces += [(r, u0, u1, partners) for u0, u1, partners in merged]
    claims: Dict[int, List[tuple]] = {}      # s index -> (-parallel, t0, t1, piece)
    for i, (r, u0, u1, partners) in enumerate(pieces):
        for k in partners.values():
            s = other_list[k]
            t0, t1 = sorted(min(max(s.project(r.at(u)), 0.0), 1.0) for u in (u0, u1))
            sx, sy = s.unit()
            rx, ry = r.unit()
            claims.setdefault(k, []).append((-abs(rx * sx + ry * sy), t0, t1, i))
    found: Dict[int, Dict[str, Seg]] = {}
    rest: Dict[str, List[Seg]] = {n: [] for n in others}
    for k, s in enumerate(other_list):
        # The segments most nearly parallel to it first: a chord of a bend, not the
        # straight it turns from, has the stretch of the chord beside it. Each claim
        # keeps the longest part of its stretch the ones before it left.
        kept = []
        for _, t0, t1, i in sorted(claims.get(k, [])):
            free = [(t0, t1)]
            for a, b, _ in kept:
                free = [x for f0, f1 in free
                        for x in ((f0, min(f1, a)), (max(f0, b), f1)) if x[1] > x[0]]
            if free:
                t0, t1 = max(free, key=lambda x: x[1] - x[0])
                if (t1 - t0) * s.length_mm >= 1e-3:
                    kept.append([t0, t1, i])
        kept.sort()
        # A short stretch beside nothing goes in with the stretch before it (or after, at
        # the start): the outer chord of a tight bend, past the inner's ends, is its tail.
        gap = ABSORB_MM_SHARE * segment_mm / max(s.length_mm, 1e-9)
        if kept and kept[0][0] <= gap:
            kept[0][0] = 0.0
        for c, d in zip(kept, kept[1:]):
            if d[0] - c[1] <= gap:
                c[1] = d[0]
        if kept and 1.0 - kept[-1][1] <= gap:
            kept[-1][1] = 1.0
        prev = 0.0
        for t0, t1, i in kept + [[1.0, 1.0, None]]:
            if t0 - prev > 1e-9:
                rest[s.net] += _chunks(Seg(s.net, s.layer, s.at(prev), s.at(t0), s.width_mm),
                                       segment_mm)
            if i is not None:
                r = pieces[i][0]
                piece = Seg(s.net, s.layer, s.at(t0), s.at(t1), s.width_mm)
                # Orient the stretch as the reference segment runs.
                if s.project(r.a) > s.project(r.b):
                    piece = Seg(s.net, s.layer, piece.b, piece.a, s.width_mm)
                found.setdefault(i, {})[s.net] = piece
            prev = max(prev, t1)
    out = []
    i = 0
    while i < len(pieces):
        r = pieces[i][0]
        mine = []                            # [u0, u1, partners] along r
        while i < len(pieces) and pieces[i][0] is r:
            mine.append([pieces[i][1], pieces[i][2], found.get(i, {})])
            i += 1
        # A piece whose stretch beside it went to another (or came to nothing) goes in
        # with a neighbour that kept its own.
        k = 0
        while len(mine) > 1 and k < len(mine):
            if mine[k][2]:
                k += 1
                continue
            if k > 0:
                mine[k - 1][1] = mine[k][1]
            else:
                mine[k + 1][0] = mine[k][0]
            del mine[k]
            if not any(m[2] for m in mine):
                break
        for u0, u1, partners in mine:
            seg = r if (u0, u1) == (0.0, 1.0) else \
                Seg(r.net, r.layer, r.at(u0), r.at(u1), r.width_mm)
            out.append((seg, partners))
    return out, rest


# ------------------------------------------------------------- series parts
_VALUE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([pnuµmkKMR]?)([0-9]*)", re.I)
_SCALE = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3,
          "M": 1e6, "R": 1.0, "": 1.0}


def parse_value(text: str) -> Optional[float]:
    """A part value as a number: "100n", "4k7", "0R1", "33", "10uF" (None: unreadable)."""
    m = _VALUE.match(text or "")
    if not m:
        return None
    whole, unit, frac = m.groups()
    unit = unit if unit in _SCALE else unit.lower()
    v = float(whole + ("." + frac if frac else ""))
    return v * _SCALE.get(unit, 1.0)


def series_parts(snap: BoardSnapshot, nets: Sequence[str],
                 not_fitted: Sequence[str] = ()) -> List[Tuple[str, str, str, str]]:
    """Two-pin R, C, L or ferrite parts joining two different nets of ``nets``:
    (reference, kind, value text, "netA|netB"); none of ``not_fitted``."""
    by_fp: Dict[str, List[Pad]] = {}
    want = set(nets)
    for p in snap.pads:
        if p.footprint and p.net in want:
            by_fp.setdefault(p.footprint, []).append(p)
    out = []
    for fp in snap.footprints:
        if fp.reference in not_fitted:
            continue
        pads = by_fp.get(fp.reference, [])
        nets_here = {p.net for p in pads}
        all_pads = [p for p in snap.pads if p.footprint == fp.reference and p.net]
        if len(all_pads) != 2 or len(nets_here) != 2:
            continue
        kind = re.match(r"[A-Za-z]+", fp.reference)
        kind = kind.group(0).upper() if kind else ""
        if kind in ("R", "RN"):
            k = "R"
        elif kind == "C":
            k = "C"
        elif kind in ("L", "FB", "FL"):
            k = "L"
        else:
            continue
        a, b = sorted(nets_here)
        out.append((fp.reference, k, fp.value, f"{a}|{b}"))
    return out


def follow_series(snap: BoardSnapshot, net: str, limit: int = 4,
                  not_fitted: Sequence[str] = ()) -> List[str]:
    """``net`` and the nets reached from it through series parts (an AC-coupling
    capacitor, a series resistor), up to ``limit`` nets; not through ``not_fitted`` (an
    assembly variant's empty footprints)."""
    out = [net]
    frontier = [net]
    while frontier and len(out) < limit:
        n = frontier.pop()
        _, _, pads, _ = snap.items_on_net(n)
        for fp in {p.footprint for p in pads if p.footprint}:
            fpads = [p for p in snap.pads if p.footprint == fp and p.net]
            if len(fpads) != 2 or not re.match(r"(R|C|L|FB)\d", fp) or fp in not_fitted:
                continue
            other = next((p.net for p in fpads if p.net != n), None)
            if other and other not in out:
                from kipi.core.model.nets import is_ground_name, is_power_name
                if is_ground_name(other) or is_power_name(other):
                    continue
                out.append(other)
                frontier.append(other)
    return out


# -------------------------------------------------------------------- build
@dataclass
class BuildOptions:
    segment_mm: float = 1.0
    window_mm: float = 1.5
    roughness: Roughness = field(default_factory=Roughness)
    via_models: bool = True
    stitch_radius_mm: float = STITCH_RADIUS_MM
    region: Optional[Tuple[float, float, float, float]] = None   # only the copper in it
    # The plane pairs' edges: "board" (open edges at the board's outline: its cavity
    # resonances) or "open" (planes going on for ever, as in a 3D model's region).
    planes: str = "board"
    # Parts not fitted (an assembly variant's): their pads stay copper, but nothing joins
    # them.
    not_fitted: Tuple[str, ...] = ()


def build_channel(board: SIBoard, groups: Sequence[Sequence[str]], opts: BuildOptions,
                  check_cancel=None,
                  pairs: Optional[Sequence[Tuple[str, str]]] = None,
                  on_progress: Optional[Callable[[int, int], None]] = None,
                  workers: int = 0) -> Channel:
    """The circuit of ``groups``: each group is a list of nets that run side by side (a
    pair's P and N, or a victim and its aggressors), coupled into multiconductor
    sections; the groups' nets are joined by any series parts between them. A section
    holding both nets of one of ``pairs`` gets their Zdiff, with any others in it held
    at 0 V; ``pairs`` defaults to the groups of exactly two nets. The sections' distinct
    cross-sections are field-solved together, on ``workers`` processes (0: one fewer than
    the cores) when there are enough of them; ``on_progress(done, total)`` is called as
    they are."""
    snap = board.snap
    if pairs is None:
        pairs = [(g[0], g[1]) for g in groups if len(g) == 2]
    nets = [n for g in groups for n in g]
    ckt = Circuit()
    ch = Channel(ckt, list(nets))
    nodes = _Nodes(snap, nets)
    probe = _Nodes(snap, nets)                  # its keys mark nothing as used

    def plain(net, layer, p):
        return probe.key(net, layer, p)[0] == "pt"
    work: List[List[Seg]] = []                  # each section's segments, reference first
    for g in groups:
        segs = {n: net_segments(snap, n, opts.segment_mm, opts.region) for n in g}
        ref, others = g[0], {n: segs[n] for n in g[1:]}
        paired, rest = _pair_up(segs[ref], others, opts.window_mm, board, opts.segment_mm,
                                plain) if others else ([(s, {}) for s in segs[ref]], {})
        work += [[r, *partners.values()] for r, partners in paired]
        work += [[s] for lst in rest.values() for s in lst]
    cuts = []
    for sec in work:
        if check_cancel:
            check_cancel()
        c = _cut(board, sec, opts)
        cuts.append(c)
        if c is not None and len(c.segs) < len(sec):
            # A partner the reference's cut does not cross (it turns into a pad there,
            # say) goes on alone, or its net would be broken at it.
            kept = {id(s) for s in c.segs}
            cuts += [_cut(board, [s], opts) for s in sec[1:] if id(s) not in kept]
    _solve_cuts(board, [c for c in cuts if c is not None], workers, check_cancel, on_progress)
    for c in cuts:
        if c is not None:
            _add_section(ch, board, nodes, c, opts, pairs)
    _add_vias(ch, board, [next(v for v in nodes.vias if v.id == vid) for vid in nodes.used_vias],
              opts)
    for ref, kind, value, pair in (series_parts(snap, nets, opts.not_fitted) if opts.region is None
                                     else []):
        a, b = pair.split("|")
        pads = [p for p in snap.pads if p.footprint == ref and p.net]
        pa = next(p for p in pads if p.net == a)
        pb = next(p for p in pads if p.net == b)
        v = parse_value(value)
        if v is None:
            ch.notes.append(f"{ref}: cannot read the value {value!r}; taken as a short")
            ckt.add_r(("pad", pa.id), ("pad", pb.id), 1e-3, ref)
        elif kind == "R":
            ckt.add_r(("pad", pa.id), ("pad", pb.id), max(v, 1e-3), ref)
        elif kind == "C":
            esl = 0.5e-9
            ckt.add_z(("pad", pa.id), ("pad", pb.id),
                      lambda f, c=v, l=esl: 0.02 + 2j * math.pi * f * l + 1 / (2j * math.pi * f * c),
                      ref)
        else:
            ckt.add_l(("pad", pa.id), ("pad", pb.id), v, 0.05, ref)
        nodes.used_pads[pa.id] = pa
        nodes.used_pads[pb.id] = pb
        ch.series.append(ref)
    ch.pads = {pid: ("pad", pid) for pid in nodes.used_pads}
    return ch


@dataclass
class _SectionCut:
    segs: List[Seg]                     # the reference first, then the partners found
    conds: list
    box: tuple
    key: tuple


def _cut(board: SIBoard, segs: List[Seg], opts: BuildOptions) -> Optional[_SectionCut]:
    """The cross-section of a section's segments (None for one of no length)."""
    r = segs[0]
    mid, nrm, ln = normal_of(r.a, r.b)
    if ln <= 1e-6:
        return None
    partners = {}
    for s in segs[1:]:
        q = s.at(0.5)
        partners[s.net] = (s.layer, (q[0] - mid[0]) * nrm[0] + (q[1] - mid[1]) * nrm[1])
    conds, box, found = board.section(mid, nrm, r.layer, r.width_mm, r.net, opts.window_mm,
                                      partners)
    segs = [r] + [s for s in segs[1:] if s.net in found]
    return _SectionCut(segs, conds, box, board.cache.key(conds, board.bands, box))


def _solve_cuts(board: SIBoard, cuts: List[_SectionCut], workers: int, check_cancel,
                on_progress):
    """Field-solve the cuts not in the board's cache yet, each distinct one once."""
    from types import SimpleNamespace
    from kipi.core.progress import ProgressReporter
    from kipi.si.scan import _solve_all
    todo = {}
    for c in cuts:
        if c.key in board.cache._d or c.key in todo:
            board.cache.hits += 1
        else:
            todo[c.key] = c

    class _Cancel(ProgressReporter):
        def check_cancel(self):
            if check_cancel:
                check_cancel()
    _solve_all(todo, board, board.cache,
               SimpleNamespace(workers=workers, mesh_scale=board.cache.mesh_scale),
               _Cancel(), on_progress)


def _add_section(ch: Channel, board: SIBoard, nodes: _Nodes, cut: _SectionCut,
                 opts: BuildOptions, pairs: Sequence[Tuple[str, str]] = ()):
    from kipi.si.xsection import _renamed
    segs = cut.segs
    hit = board.cache._d.get(cut.key)
    lp = _renamed(hit, cut.conds) if hit is not None else board.solve(cut.conds, cut.box)
    order = [lp.names.index(s.net) for s in segs]
    idx = np.array(order)

    def rlgc(f, lp=lp, idx=idx):
        z, y = lp.rlgc(f, opts.roughness)
        return z[:, idx][:, :, idx], y[:, idx][:, :, idx]
    for s in segs:
        s.na = nodes.key(s.net, s.layer, s.a)
        s.nb = nodes.key(s.net, s.layer, s.b)
    length = float(np.mean([s.length_mm for s in segs]))
    # Coupled over the shortest conductor's length; a longer one (the outer leg of a
    # bend) goes on alone for the rest of its own, so each keeps its length (and delay).
    common = min(s.length_mm for s in segs)
    ends = []
    for k, s in enumerate(segs):
        extra = s.length_mm - common
        if len(segs) == 1 or extra <= TAIL_MIN_SHARE * common:
            ends.append(s.nb)
            continue
        m = s.at(common / s.length_mm)
        mid_key = ("tail", s.net, s.layer, round(m[0] / SNAP_MM), round(m[1] / SNAP_MM))
        ends.append(mid_key)
        i = order[k]

        def self_rlgc(f, lp=lp, i=i):
            z, y = lp.rlgc(f, opts.roughness)
            return z[:, [i]][:, :, [i]], y[:, [i]][:, :, [i]]
        ch.circuit.add_line([mid_key], [s.nb], extra, self_rlgc, f"{s.net} (alone)")
    ch.circuit.add_line([s.na for s in segs], ends,
                        common if len(segs) > 1 else length, rlgc,
                        "+".join(s.net for s in segs))
    info = SectionInfo([s.net for s in segs], segs, [lp.z0(i) for i in order],
                       length_mm=length, delay_s=lp.delay_s_per_mm(order[0]) * length)
    at = {s.net: i for s, i in zip(segs, order)}
    for p, n in pairs:
        if p in at and n in at:
            info.zdiff = lp.pair(at[p], at[n])["zdiff"]
            break
    ch.sections.append(info)


def decaps_between(board: SIBoard, na: str, nb: str, at, not_fitted: Sequence[str] = ()
                   ) -> List[tuple]:
    """The capacitors joining nets ``na`` and ``nb`` (a supply's decoupling to ground),
    nearest ``at`` first, up to MAX_DECAPS: (x, y, r, C, ESR, ESL). None of
    ``not_fitted``, nor one whose value cannot be read."""
    cache = board.__dict__.setdefault("_decaps", {})
    key = tuple(sorted((na, nb)))
    if key not in cache:
        by_fp: Dict[str, List[Pad]] = {}
        for p in board.snap.pads:
            if p.footprint and p.net in key:
                by_fp.setdefault(p.footprint, []).append(p)
        values = {f.reference: f.value for f in board.snap.footprints}
        found = []
        for ref, pads in by_fp.items():
            if not re.match(r"C\d", ref) or len(pads) != 2 or {p.net for p in pads} != set(key):
                continue
            if len([p for p in board.snap.pads if p.footprint == ref]) != 2:
                continue
            c = parse_value(values.get(ref, ""))
            if c is None or c <= 0:
                continue
            x = sum(p.position[0] for p in pads) / 2
            y = sum(p.position[1] for p in pads) / 2
            found.append((ref, (x, y, DECAP_POST_R_MM, c, DECAP_ESR_OHM, DECAP_ESL_H)))
        cache[key] = found
    near = sorted((d for ref, d in cache[key] if ref not in not_fitted),
                  key=lambda d: math.hypot(d[0] - at[0], d[1] - at[1]))
    return near[:MAX_DECAPS]


def via_stack(board: SIBoard, via: Via, stitch_radius_mm: float = STITCH_RADIUS_MM,
              not_fitted: Sequence[str] = ()) -> ViaStack:
    """A via's barrel through the board: its layers, pads, the planes around it (their
    antipads) and the return path between each pair of planes: stitching vias between
    planes of one net, decoupling capacitors between planes of two."""
    from shapely.geometry import Point
    stack = board.stack
    layers = [l for l in stack.layers_between(via.layers[0], via.layers[-1])]
    vl, plane_net = [], {}
    pt = Point(via.position)
    for lid in layers:
        cu = stack.layer(lid)
        antipad = None
        best = None
        for net, g in board.geoms.get(lid, []):
            if net == via.net or not board.is_reference(net):
                continue
            d = g.distance(pt)
            if d < 2.0 and (best is None or d < best[0]):
                # A plane surrounds the via: copper on most of a ring just outside it.
                ring = pt.buffer(d + 0.3, quad_segs=8).exterior
                cover = ring.intersection(g).length / max(ring.length, 1e-9)
                if cover > 0.6:
                    best = (d, net)
        if best is not None:
            antipad = max(best[0], via.drill_mm / 2 + 0.01)
            plane_net[len(vl)] = best[1]
        vl.append(ViaLayer(cu.z_bottom_mm, cu.z_top_mm, via.pad_diameter(lid), antipad,
                           cu.name))
    vs = ViaStack(via.drill_mm, vl, board.bands)
    planes = vs.plane_indices()
    for pa, pb in zip(planes, planes[1:]):
        h = vl[pa].z_bottom - vl[pb].z_top
        na, nb = plane_net[pa], plane_net[pb]
        band = next((b for b in board.bands if b.z0 <= vl[pb].z_top + 1e-6
                     and b.z1 >= vl[pb].z_top + 1e-6), board.bands[0])
        stitches = []
        if na == nb:
            _, gvias, gpads, _ = board.snap.items_on_net(na)
            la, lb = layers[pa], layers[pb]
            cands = [(v.position, v.drill_mm / 2) for v in gvias
                     if la in v.layers and lb in v.layers]
            cands += [(p.position, max(p.drill_mm) / 2) for p in gpads
                      if p.pad_type == "pth" and la in p.shapes and lb in p.shapes]
            cands = [(math.hypot(q[0] - via.position[0], q[1] - via.position[1]), q, r)
                     for q, r in cands]
            cands = sorted(c for c in cands if 1e-3 < c[0] <= stitch_radius_mm)[:MAX_STITCHES]
            stitches = [(q[0], q[1], max(r, 0.05)) for _, q, r in cands]
        decaps = [] if na == nb else decaps_between(board, na, nb, via.position, not_fitted)
        pair = PlanePair((layers[pa], layers[pb]), (na, nb), h, band.material, stitches,
                         decaps)
        vs.pairs[(pa, pb)] = pair
        vs.cavity[(pa, pb)] = (lambda f, pair=pair: pair_impedance(
            board, pair, [(*via.position, via.drill_mm / 2)])(f)[:, 0, 0])
    return vs


def pair_impedance(board: SIBoard, pair: PlanePair, posts: Sequence[tuple], planes: str = "board"):
    """z(f) (F, N, N) of the signal vias ``posts`` ((x, y, r), board mm) crossing ``pair``,
    its stitching vias shorted and each barrel's own resistance added; the last result is
    kept for the next call at the same frequencies."""
    x0, y0, x1, y1 = board.outline_box
    size = None if planes == "open" else (max(x1 - x0, 1.0), max(y1 - y0, 1.0))
    origin = (0.0, 0.0) if size is None else (x0, y0)
    own = {(round(x, 4), round(y, 4)) for x, y, _ in posts}
    centre = np.mean([(x, y) for x, y, _ in posts], axis=0)
    shorts = sorted({s for s in pair.stitches if (round(s[0], 4), round(s[1], 4)) not in own},
                    key=lambda s: math.hypot(s[0] - centre[0], s[1] - centre[1]))[:MAX_POSTS]
    decaps = list(getattr(pair, "decaps", []))
    every = [(x - origin[0], y - origin[1], r) for x, y, r in
             list(posts) + shorts + [d[:3] for d in decaps]]
    n = len(posts)
    cache = {}

    def loads(f):
        """Shorts, then each capacitor's ESR + jωESL + 1/(jωC)."""
        if not decaps:
            return None
        w = 2 * math.pi * np.maximum(np.asarray(f, float), 1e3)
        out = np.zeros((len(f), len(shorts) + len(decaps)), complex)
        for k, (_, _, _, c, esr, esl) in enumerate(decaps):
            out[:, len(shorts) + k] = esr + 1j * w * esl + 1 / (1j * w * c)
        return out

    def z(f):
        key = (len(f), float(f[0]), float(f[-1]))
        if key not in cache:
            zz, c0 = plane_pair_z(f, pair.h_mm, every, pair.material, size)
            out = shorted(zz, c0, n, loads(f))
            for k, (_, _, r) in enumerate(posts):
                out[:, k, k] += barrel_resistance(f, pair.h_mm, r)
            cache.clear()
            cache[key] = out
        return cache[key]
    return z


def _add_vias(ch: Channel, board: SIBoard, vias: Sequence[Via], opts: BuildOptions):
    """The channel's vias: each barrel's chain, with the pieces in a plane pair coupled to
    every other via of the channel crossing the same pair."""
    stacks = {}
    groups: Dict[tuple, List[tuple]] = {}
    for via in vias:
        layers = board.stack.layers_between(via.layers[0], via.layers[-1])
        keys = [("via", via.id, lid) for lid in layers]
        if not opts.via_models:
            for k in keys[1:]:
                ch.circuit.add_r(keys[0], k, 1e-4, "via")
            ch.vias.append(via.id)
            continue
        vs = via_stack(board, via, opts.stitch_radius_mm, opts.not_fitted)
        stacks[via.id] = (via, vs, keys)
        for (pa, pb), pair in vs.pairs.items():
            groups.setdefault((pair.layer_ids, pair.nets), []).append((via, vs, keys, pa, pb))
    done: Dict[str, set] = {vid: set() for vid in stacks}
    for _, members in groups.items():
        pair = members[0][1].pairs[(members[0][3], members[0][4])]
        stitches = {s for m in members for s in m[1].pairs[(m[3], m[4])].stitches}
        decaps = {d for m in members for d in m[1].pairs[(m[3], m[4])].decaps}
        centre = np.mean([m[0].position for m in members], axis=0)
        decaps = sorted(decaps, key=lambda d: math.hypot(d[0] - centre[0],
                                                         d[1] - centre[1]))[:MAX_DECAPS]
        pair = PlanePair(pair.layer_ids, pair.nets, pair.h_mm, pair.material, sorted(stitches),
                         decaps)
        posts = [(*m[0].position, m[0].drill_mm / 2) for m in members]
        z = pair_impedance(board, pair, posts, opts.planes)
        _, vs0, _, pa0, pb0 = members[0]
        for k, (_, share) in enumerate(segment_shares(vs0, pa0, pb0)):
            tops = [m[2][m[3] + k] for m in members]
            bottoms = [m[2][m[3] + k + 1] for m in members]
            add_coupled_segments(ch.circuit, z, share, tops, bottoms)
        for m in members:
            done[m[0].id].add((m[3], m[4]))
    for vid, (via, vs, keys) in stacks.items():
        add_via_chain(ch.circuit, vs, keys, capacitances(vs), skip=done[vid])
        ch.vias.append(via.id)
