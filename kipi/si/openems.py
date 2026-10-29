"""3D full-wave extraction of a board region with openEMS (FDTD; Liebig, GPL-3).

KiPI writes openEMS's XML input itself and runs its ``openEMS`` program, so neither its
Python bindings nor Octave are needed: only the program, found on the PATH, in
``KIPI_OPENEMS`` (the program or its folder) or in the usual install places.

**The model** of a rectangular region of the board:

* every dielectric of the stackup (and the solder masks) as a box, its εr and its loss
  tangent at the band's centre as a conductivity κ = ω·ε0·εr·tan δ;
* every copper layer's copper in the region as zero-thickness PEC sheets at the layer's
  height (the usual FDTD board model: a sheet is meshed far cheaper than a 35 µm slab),
  plane clearances (antipads) cut out of them with higher-priority dielectric patches;
* vias as PEC cylinders through their span;
* a lumped port (50 Ω) where each chosen net's trace leaves the region, between the
  trace's end and its nearest reference sheet;
* air above and below, the region padded by ``pad_mm`` all round, and PML on every side;
  a plane pair (two planes and the dielectric between) runs on through the PML, as the
  board's go on past the region, so it has no edges to resonate between.

**The mesh** is rectilinear: lines at every feature (via centres and barrels, antipads,
trace edges, the ports, each copper and dielectric height), filled between them with
cells that grow by at most ``grade`` from cell to cell up to λ/``cells_per_lambda`` at
f_max in the densest dielectric.

**The run** excites each port in turn with a Gaussian pulse covering 0..f_max until the
energy has fallen by ``end_db``. Each port's voltage and current probes are Fourier
transformed and split into incident and reflected waves at 50 Ω, which gives a column
of S per run, as openEMS's own port code does.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kipi.ac.solver.physics import EPS0_F_PER_MM
from kipi.core.model.board import BoardSnapshot
from kipi.core.progress import NULL, ProgressReporter
from kipi.si.board import SIBoard

C0_MM_S = 299792458e3
_ENERGY = re.compile(r"Energy:\s*~?\s*[-+0-9.eE]+\s*\(\s*([-+0-9.]+)\s*dB\)")


FAR_MM = 200.0                 # the planes and dielectrics run this far past the region


class OpenEMSError(RuntimeError):
    pass


def find_openems() -> Optional[str]:
    """The openEMS program: ``KIPI_OPENEMS`` (the program or its folder), the PATH, or a
    usual install place; None if there is none."""
    env = os.environ.get("KIPI_OPENEMS")
    cands = []
    if env:
        p = Path(env).expanduser()
        cands += [p, p / "openEMS", p / "openEMS.exe", p / "bin" / "openEMS"]
    which = shutil.which("openEMS") or shutil.which("openEMS.exe")
    if which:
        cands.append(Path(which))
    home = Path.home()
    cands += [home / "opt" / "openEMS" / "bin" / "openEMS", Path("/usr/local/bin/openEMS"),
              Path("/opt/openEMS/bin/openEMS"), Path("/usr/bin/openEMS"),
              Path("C:/openEMS/openEMS.exe"), Path("C:/Program Files/openEMS/openEMS.exe"),
              home / "openEMS" / "openEMS.exe"]
    for c in cands:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


# The excitation's centre and 20 dB half-width, and the lowest frequency reported, as
# shares of f_max.
F0_SHARE, FC_SHARE, F_LOW_SHARE = 0.6, 0.4, 0.1


@dataclass
class EMSettings:
    f_max_hz: float = 20e9
    cells_per_lambda: int = 20
    fine_mm: float = 0.05             # cells at features (via barrels, trace edges)
    grade: float = 1.4
    pad_mm: float = 1.0               # dielectric and planes beyond the region
    air_mm: float = 2.0               # above and below the board
    pml_cells: int = 8
    end_db: float = -40.0
    max_timesteps: int = 300000
    threads: int = 0                  # 0: openEMS's choice
    n_freqs: int = 201
    z0_ohm: float = 50.0


@dataclass
class EMPort:
    name: str
    net: str
    start: Tuple[float, float, float]
    stop: Tuple[float, float, float]
    direction: int                    # 0 x, 1 y, 2 z: the excitation's
    xy: Tuple[float, float] = (0.0, 0.0)
    layer: int = 0                    # the trace's copper layer


@dataclass
class EMModel:
    region: Tuple[float, float, float, float]       # x0, y0, x1, y1 (board mm)
    ports: List[EMPort]
    mesh: Dict[str, np.ndarray]
    properties_xml: str
    settings: EMSettings
    notes: List[str] = field(default_factory=list)

    @property
    def cells(self) -> int:
        return int(np.prod([len(v) - 1 for v in self.mesh.values()]))

    def xml(self, excite: int) -> str:
        """The openEMS input with port ``excite`` driven."""
        s = self.settings
        # A band-pass pulse, f_max·(0.2–1) within 20 dB, its DC ~45 dB down: DC charge
        # left on a closed loop of PEC (planes and stitching vias) would hold its energy
        # up for ever and the run would never meet its end criterion.
        f0, fc = F0_SHARE * s.f_max_hz, FC_SHARE * s.f_max_hz
        props = [self.properties_xml]
        for k, p in enumerate(self.ports):
            props.append(_port_xml(p, k + 1, s.z0_ohm, k == excite))
        bc = " ".join(f'{b}="PML_{s.pml_cells}"' for b in ("xmin", "xmax", "ymin", "ymax",
                                                          "zmin", "zmax"))
        m = self.mesh
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n<openEMS>\n'
            f'  <FDTD NumberOfTimesteps="{s.max_timesteps}" endCriteria="{10 ** (s.end_db / 10):.3g}" '
            f'f_max="{s.f_max_hz:.6g}">\n'
            f'    <Excitation Type="0" f0="{f0:.6g}" fc="{fc:.6g}">\n    </Excitation>\n'
            f'    <BoundaryCond {bc}>\n    </BoundaryCond>\n  </FDTD>\n'
            '  <ContinuousStructure CoordSystem="0">\n    <Properties>\n'
            + "".join(props) +
            '    </Properties>\n    <RectilinearGrid DeltaUnit="0.001" CoordSystem="0">\n'
            f'      <XLines>{_lines(m["x"])}</XLines>\n      <YLines>{_lines(m["y"])}</YLines>\n'
            f'      <ZLines>{_lines(m["z"])}</ZLines>\n    </RectilinearGrid>\n'
            '  </ContinuousStructure>\n</openEMS>\n')


def _lines(v) -> str:
    return ",".join(f"{x:.6g}" for x in v)


def _pt(tag, x, y, z) -> str:
    return f'<{tag} X="{x:.6g}" Y="{y:.6g}" Z="{z:.6g}"></{tag}>'


def _box(p1, p2, prio) -> str:
    return f'<Box Priority="{prio}">{_pt("P1", *p1)}{_pt("P2", *p2)}</Box>'


def _poly(ring, z, prio) -> str:
    v = "".join(f'<Vertex X1="{x:.6g}" X2="{y:.6g}"></Vertex>' for x, y in ring)
    return f'<Polygon Priority="{prio}" Elevation="{z:.6g}" NormDir="2">{v}</Polygon>'


def _port_xml(p: EMPort, n: int, z0: float, excite: bool) -> str:
    d = p.direction
    sign = 1 if p.stop[d] > p.start[d] else -1
    out = (f'<LumpedElement Name="port_resist_{n}" Direction="{d}" Caps="1" R="{z0:g}">'
           f'<Primitives>{_box(p.start, p.stop, 5)}</Primitives></LumpedElement>\n')
    if excite:
        ex = ["0", "0", "0"]
        ex[d] = str(-sign)
        out += (f'<Excitation Name="port_excite_{n}" Type="0" Excite="{",".join(ex)}">'
                f'<Primitives>{_box(p.start, p.stop, 5)}</Primitives></Excitation>\n')
    mid = [0.5 * (a + b) for a, b in zip(p.start, p.stop)]
    u1, u2 = list(mid), list(mid)
    u1[d], u2[d] = p.start[d], p.stop[d]
    out += (f'<ProbeBox Name="port_ut{n}" Type="0" Weight="-1">'
            f'<Primitives>{_box(u1, u2, 5)}</Primitives></ProbeBox>\n')
    i1, i2 = list(p.start), list(p.stop)
    i1[d] = i2[d] = 0.5 * (p.start[d] + p.stop[d])
    out += (f'<ProbeBox Name="port_it{n}" Type="1" Weight="{sign}" NormDir="{d}">'
            f'<Primitives>{_box(i1, i2, 5)}</Primitives></ProbeBox>\n')
    return out


# ------------------------------------------------------------------ mesh
def graded_lines(fixed: Sequence[float], fine: Sequence[float], h_max: float,
                 grade: float) -> np.ndarray:
    """Mesh lines through every ``fixed`` line, cells of ``fine`` (each fixed line's own)
    at them, growing by ``grade`` per cell away from them up to ``h_max``."""
    f = np.asarray(sorted(set(round(x, 6) for x in fixed)))
    if len(f) < 2:
        return f
    fine_at = dict(zip([round(x, 6) for x in fixed], fine))
    hs = np.array([fine_at.get(round(x, 6), h_max) for x in f])

    def size(x):
        # local size: min over fixed lines of h_i·grade^(distance in cells) ~ linear growth
        d = np.abs(x - f)
        return float(np.min(np.minimum(h_max, hs + (grade - 1) * d)))
    out = [f[0]]
    for a, b in zip(f, f[1:]):
        # Equidistribute: lines where ∫ dx / size(x) from a reaches a whole number of
        # cells (rounded to fit), so neighbouring cells differ by about the grade at most.
        xs = np.linspace(a, b, 257)
        w = 1.0 / np.array([size(x) for x in xs])
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(xs))])
        n = max(1, int(math.ceil(cum[-1] - 0.25)))
        out += list(np.interp(np.linspace(0, cum[-1], n + 1)[1:-1], cum, xs))
        out.append(b)
    out = np.unique(np.round(out, 6))
    # No cell much smaller than the finest asked for: lines that close are one.
    min_gap = 0.5 * float(np.min(hs))
    keep = [out[0]]
    fixed_set = set(np.round(f, 6))
    for x in out[1:]:
        if x - keep[-1] >= min_gap:
            keep.append(x)
        elif round(x, 6) in fixed_set and round(keep[-1], 6) not in fixed_set:
            keep[-1] = x
    return np.asarray(keep)


# ----------------------------------------------------------------- model
def build_model(board: SIBoard, region: Tuple[float, float, float, float], nets: Sequence[str],
                settings: Optional[EMSettings] = None) -> EMModel:
    """The openEMS model of ``region`` (x0, y0, x1, y1, board mm), with ports where the
    traces of ``nets`` leave it."""
    import shapely
    from shapely.geometry import box as sbox
    from kipi.core.geometry.copper import net_layer_copper
    s = settings or EMSettings()
    snap = board.snap
    stack = board.stack
    x0, y0, x1, y1 = region
    pad = s.pad_mm
    dom = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    clip = sbox(*dom)
    inner = sbox(*region)
    # A plane pair (its two planes and the dielectric between) goes on through the
    # absorbing walls, so it is infinite: no edges a region's size away to resonate between.
    far = (dom[0] - FAR_MM, dom[1] - FAR_MM, dom[2] + FAR_MM, dom[3] + FAR_MM)
    outside = sbox(*far).difference(clip)
    edge = clip.exterior
    # KiCad's y grows down; openEMS gets y up (y' = −y), so the model is not mirrored.
    def ty(y):
        return -y
    props = []
    zb, zt = board.z_range
    z_lines = [zb - s.air_mm, zt + s.air_mm]
    fine_z = [s.air_mm / 4, s.air_mm / 4]
    eps_max = 1.0
    f_c = s.f_max_hz / 2
    bands = sheet_bands(board)
    # Which sheets are planes across the model (a reference net's copper on most of its
    # edge): those go on past it, and so does a dielectric between two of them. An outer
    # dielectric stops at the model's edge: carried on, it would carry the surface wave a
    # lumped port launches off into the absorbing walls, a loss the board has not.
    copper = {}
    planes = set()
    for i, lay in enumerate(stack.copper):
        for net in snap.net_names():
            g = net_layer_copper(snap, net, lay.id, cut_drills=False, quad_segs=4)
            copper[(net, lay.id)] = g
            if g is not None and not g.is_empty and board.is_reference(net) and \
                    edge.intersection(g).length > 0.5 * edge.length:
                planes.add(i)
    pairs = {i for i in planes if i - 1 in planes or i + 1 in planes}
    for k, (za, zb_, mat) in enumerate(bands):
        ext = far if (k in pairs and k + 1 in pairs) else dom
        e = mat.eps(f_c)
        er = float(e.real)
        eps_max = max(eps_max, er)
        kappa = 2 * math.pi * f_c * EPS0_F_PER_MM * 1e3 * float(-e.imag)   # S/m
        props.append(
            f'<Material Name="diel{k}"><Property Epsilon="{er:.5g}" Kappa="{kappa:.5g}">'
            f'</Property><Primitives>'
            + _box((ext[0], ty(ext[3]), za), (ext[2], ty(ext[1]), zb_), 1 + k)
            + '</Primitives></Material>\n')
    xs, ys = [dom[0], dom[2], x0, x1], [dom[1], dom[3], y0, y1]
    fx, fy = [s.fine_mm] * 4, [s.fine_mm] * 4
    metal, holes = [], []
    for i, lay in enumerate(stack.copper):
        z = lay.z_mid_mm
        z_lines.append(z)
        fine_z.append(s.fine_mm / 2)
        for net in snap.net_names():
            g = copper[(net, lay.id)]
            if g is None or g.is_empty:
                continue
            ref = board.is_reference(net)
            g = g.intersection(clip if ref else inner)
            if g.is_empty:
                continue
            if ref and i in pairs and edge.intersection(g).length > 0.5 * edge.length:
                g = g.union(outside)          # a plane: on past the model's edge
            g = g.simplify(0.005)
            for poly in shapely.get_parts(g):
                if poly.geom_type != "Polygon" or poly.area < 1e-6:
                    continue
                ring = [(x, ty(y)) for x, y in list(poly.exterior.coords)[:-1]]
                metal.append(_poly(ring, z, 50))
                for h in poly.interiors:
                    hr = [(x, ty(y)) for x, y in list(h.coords)[:-1]]
                    holes.append((hr, z))
    props.append('<Metal Name="copper"><Primitives>' + "".join(metal) + '</Primitives></Metal>\n')
    if holes:
        # A clearance is the dielectric beside it, above the copper's priority.
        def band_at(z):
            return next((k for k, b in enumerate(bands) if b[0] <= z <= b[1]), 0)
        by = {}
        for ring, z in holes:
            by.setdefault(band_at(z), []).append(_poly(ring, z, 60))
        for k, polys in by.items():
            e = bands[k][2].eps(f_c)
            props.append(f'<Material Name="clear{k}"><Property Epsilon="{float(e.real):.5g}">'
                         f'</Property><Primitives>' + "".join(polys) + '</Primitives></Material>\n')
    cyl = []
    for v in snap.vias:
        px, py = v.position
        if not (dom[0] < px < dom[2] and dom[1] < py < dom[3]):
            continue
        za = stack.layer(v.layers[0]).z_mid_mm
        zbm = stack.layer(v.layers[-1]).z_mid_mm
        r = v.drill_mm / 2
        cyl.append(f'<Cylinder Priority="70" Radius="{r:.6g}">{_pt("P1", px, ty(py), za)}'
                   f'{_pt("P2", px, ty(py), zbm)}</Cylinder>')
        xs += [px - r, px, px + r]
        ys += [py - r, py, py + r]
        fx += [s.fine_mm / 2] * 3
        fy += [s.fine_mm / 2] * 3
        for lid in v.layers:
            ap = _antipad(board, v, lid)
            if ap:
                xs += [px - ap, px + ap]
                ys += [py - ap, py + ap]
                fx += [s.fine_mm] * 2
                fy += [s.fine_mm] * 2
    for p in snap.pads:
        if p.pad_type == "pth" and dom[0] < p.position[0] < dom[2] and dom[1] < p.position[1] < dom[3]:
            ls = sorted(p.shapes)
            if ls:
                za, zbm = stack.layer(ls[0]).z_mid_mm, stack.layer(ls[-1]).z_mid_mm
                r = max(p.drill_mm) / 2
                cyl.append(f'<Cylinder Priority="70" Radius="{r:.6g}">'
                           f'{_pt("P1", p.position[0], ty(p.position[1]), za)}'
                           f'{_pt("P2", p.position[0], ty(p.position[1]), zbm)}</Cylinder>')
    if cyl:
        props.append('<Metal Name="vias"><Primitives>' + "".join(cyl) + '</Primitives></Metal>\n')
    ports = _ports(board, region, nets)
    model_notes = []
    for p in ports:
        if p.direction == 2:
            if abs(p.start[0] - p.stop[0]) < 1e-9:      # a port across y at an x edge
                xs.append(p.start[0])
                ys += [-p.start[1], -p.stop[1]]
            else:
                ys.append(-p.start[1])
                xs += [p.start[0], p.stop[0]]
            fx += [s.fine_mm] * (len(xs) - len(fx))
            fy += [s.fine_mm] * (len(ys) - len(fy))
    for t in snap.tracks:
        if t.net in nets:
            for (a, b) in ((t.start, t.end),):
                if abs(a[1] - b[1]) < 1e-6:                     # along x: edges in y
                    ys += [a[1] - t.width_mm / 2, a[1] + t.width_mm / 2]
                    fy += [s.fine_mm / 2] * 2
                elif abs(a[0] - b[0]) < 1e-6:
                    xs += [a[0] - t.width_mm / 2, a[0] + t.width_mm / 2]
                    fx += [s.fine_mm / 2] * 2
    xs, fx = zip(*[(x, f) for x, f in zip(xs, fx) if dom[0] <= x <= dom[2]])
    ys, fy = zip(*[(y, f) for y, f in zip(ys, fy) if dom[1] <= y <= dom[3]])
    h_max = C0_MM_S / (s.f_max_hz * math.sqrt(eps_max)) / s.cells_per_lambda
    # The absorbing layers take the outermost cells: grow the mesh past the model by that
    # many (and a few), so the ports and copper stay out of them.
    extra = s.pml_cells + 4
    mesh = {"x": pml_padded(graded_lines(xs, fx, h_max, s.grade), extra, h_max, s.grade),
            "y": np.sort(-pml_padded(graded_lines(ys, fy, h_max, s.grade), extra, h_max,
                                     s.grade)),
            "z": pml_padded(graded_lines(z_lines, fine_z, min(h_max, s.air_mm / 2), s.grade),
                            extra, min(h_max, s.air_mm / 2), s.grade)}
    if not ports:
        model_notes.append("No trace of the chosen nets leaves the region: no ports")
    return EMModel(region, ports, mesh, "".join(props), s, model_notes)


def pml_padded(lines: np.ndarray, n: int, h_max: float, grade: float) -> np.ndarray:
    """``lines`` with ``n`` more cells beyond each end, growing by ``grade`` from the end
    cell up to ``h_max``."""
    lines = np.asarray(lines, float)
    if len(lines) < 2:
        return lines
    lo, hi = [lines[0]], [lines[-1]]
    h_lo, h_hi = lines[1] - lines[0], lines[-1] - lines[-2]
    for _ in range(n):
        h_lo, h_hi = min(h_lo * grade, h_max), min(h_hi * grade, h_max)
        lo.append(lo[-1] - h_lo)
        hi.append(hi[-1] + h_hi)
    return np.concatenate([lo[:0:-1], lines, hi[1:]])


def sheet_bands(board: SIBoard):
    """The dielectrics between the copper sheets' mid-planes, (z0, z1, material) each: in
    the 3D model copper is a sheet at its layer's middle, so each dielectric reaches from
    sheet to sheet, and the outer ones from the outer sheets to the board's faces. (A
    solder mask, 10–20 µm, is left out: its cells would set the time step.)"""
    cu = board.stack.copper
    out = []
    for upper, lower in zip(cu, cu[1:]):
        mid = 0.5 * (lower.z_top_mm + upper.z_bottom_mm)
        mat = next((b.material for b in board.bands if b.z0 <= mid <= b.z1),
                   board.bands[0].material)
        out.append((lower.z_mid_mm, upper.z_mid_mm, mat))
    return out


def _antipad(board: SIBoard, via, lid) -> Optional[float]:
    from shapely.geometry import Point
    pt = Point(via.position)
    best = None
    for net, g in board.geoms.get(lid, []):
        if net == via.net or not board.is_reference(net):
            continue
        d = g.distance(pt)
        if d < 2.0 and (best is None or d < best):
            best = d
    return best


def _ports(board: SIBoard, region, nets) -> List[EMPort]:
    """A lumped port where each track of ``nets`` crosses the region's edge: from the
    trace down (or up) to the nearest reference copper under it."""
    from shapely.geometry import LineString, box as sbox
    x0, y0, x1, y1 = region
    edge = sbox(*region).exterior
    stack = board.stack
    out = []
    for t in board.snap.tracks:
        if t.net not in nets:
            continue
        seg = LineString([t.start, t.end])
        inter = seg.intersection(edge)
        pts = [inter] if inter.geom_type == "Point" else list(getattr(inter, "geoms", []))
        for p in pts:
            px, py = p.x, p.y
            lay = stack.layer(t.layer)
            ref = _nearest_plane(board, t.layer, (px, py))
            if ref is None:
                continue
            zr = stack.layer(ref).z_mid_mm
            dx, dy = t.end[0] - t.start[0], t.end[1] - t.start[1]
            ln = math.hypot(dx, dy) or 1.0
            on_x_edge = min(abs(px - x0), abs(px - x1)) < 1e-6
            if on_x_edge:
                half = t.width_mm / 2 / max(abs(dx) / ln, 0.2)
                start = (px, -(py - half), zr)
                stop = (px, -(py + half), lay.z_mid_mm)
            else:
                half = t.width_mm / 2 / max(abs(dy) / ln, 0.2)
                start = (px - half, -py, zr)
                stop = (px + half, -py, lay.z_mid_mm)
            out.append(EMPort(f"{t.net}@{px:.2f},{py:.2f}", t.net, start, stop, 2, (px, py),
                              t.layer))
    return out


def _nearest_plane(board: SIBoard, layer: int, xy) -> Optional[int]:
    from shapely.geometry import Point
    pt = Point(xy)
    i = board.layer_index(layer)
    best = None
    for j, lid in enumerate(board.layer_ids):
        if j == i:
            continue
        for net, g in board.geoms.get(lid, []):
            if board.is_reference(net) and g.contains(pt):
                if best is None or abs(j - i) < abs(best[0] - i):
                    best = (j, lid)
                break
    return best[1] if best else None


def via_region(snap: BoardSnapshot, via_ids: Sequence[str], margin_mm: float = 2.0):
    """A square region around vias (a pair's, or one), ``margin_mm`` beyond them."""
    pts = [v.position for v in snap.vias if v.id in via_ids]
    if not pts:
        raise OpenEMSError("no such via")
    a = np.asarray(pts)
    return (float(a[:, 0].min() - margin_mm), float(a[:, 1].min() - margin_mm),
            float(a[:, 0].max() + margin_mm), float(a[:, 1].max() + margin_mm))


# ------------------------------------------------------------------- run
@dataclass
class EMResult:
    freqs_hz: np.ndarray
    s: np.ndarray                     # (F, P, P)
    port_names: List[str]
    cells: int
    seconds: float
    workdir: str
    notes: List[str] = field(default_factory=list)


def read_probe(path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="%")
    return data[:, 0], data[:, 1]


def dft(t: np.ndarray, v: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    dt = t[1] - t[0]
    out = np.empty(len(freqs), complex)
    for k in range(0, len(freqs), 32):
        f = freqs[k:k + 32]
        out[k:k + 32] = 2 * dt * (np.exp(-2j * np.pi * f[:, None] * t[None, :]) @ v)
    return out


def run(model: EMModel, workdir, progress: ProgressReporter = NULL,
        program: Optional[str] = None) -> EMResult:
    """Run every port's excitation and return the S-parameters."""
    exe = program or find_openems()
    if exe is None:
        raise OpenEMSError("openEMS was not found: install it, or set KIPI_OPENEMS to the "
                           "openEMS program or its folder")
    if not model.ports:
        raise OpenEMSError("the region has no ports")
    s = model.settings
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    freqs = np.linspace(F_LOW_SHARE * s.f_max_hz, s.f_max_hz, s.n_freqs)
    p = len(model.ports)
    smat = np.zeros((len(freqs), p, p), complex)
    t0 = time.time()
    progress.info(f"openEMS: {p} port(s), {model.cells / 1e6:.2f} M cells "
                  f"({' × '.join(str(len(v) - 1) for v in model.mesh.values())}), up to "
                  f"{s.f_max_hz / 1e9:g} GHz")
    for j in range(p):
        d = work / f"port{j + 1}"
        d.mkdir(exist_ok=True)
        (d / "model.xml").write_text(model.xml(j), encoding="utf-8")
        cmd = [exe, "model.xml"]
        if s.threads:
            cmd.append(f"--numThreads={s.threads}")
        progress.info(f"openEMS: exciting port {j + 1} of {p} ({model.ports[j].name})")
        _run_one(cmd, d, progress, j, p, s.end_db)
        inc = {}
        for k in range(p):
            tu, u = read_probe(d / f"port_ut{k + 1}")
            ti, i = read_probe(d / f"port_it{k + 1}")
            uf, i_f = dft(tu, u, freqs), dft(ti, i, freqs)
            z = s.z0_ohm
            u_inc = 0.5 * (uf + i_f * z)
            u_ref = uf - u_inc
            inc[k] = (u_inc, u_ref)
        a = inc[j][0]
        for k in range(p):
            smat[:, k, j] = inc[k][1] / a
    progress.fraction(1.0)
    return EMResult(freqs, smat, [x.name for x in model.ports], model.cells, time.time() - t0,
                    str(work), list(model.notes))


def _run_one(cmd, cwd: Path, progress: ProgressReporter, j: int, p: int, end_db: float):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ""               # as openEMS.sh does
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env)
    log = []
    try:
        for line in proc.stdout:
            log.append(line)
            m = _ENERGY.search(line)
            if m:
                db = float(m.group(1))
                frac = min(max(db / end_db, 0.0), 1.0)
                progress.fraction((j + frac) / p)
            try:
                progress.check_cancel()
            except Exception:
                proc.kill()
                raise
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    (cwd / "openems.log").write_text("".join(log), encoding="utf-8")
    if proc.returncode != 0:
        raise OpenEMSError(f"openEMS failed (exit {proc.returncode}): "
                           + "".join(log[-5:]).strip())
