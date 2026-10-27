"""Build PyVista meshes from board geometry. This module has no Qt dependency.

Copper is drawn as slabs at the true stackup Z. Vias are tubes. The viewer
applies any Z exaggeration as a plotter scale, so the meshes here stay in
real millimeters.

KiCad's y axis points down. The meshes flip it (y -> -y) so the board reads
correctly when viewed from +Z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import shapely
from shapely.geometry.base import BaseGeometry

from kipi.core.geometry.copper import drill_holes, iter_polygons, poly_to_shapely
from kipi.core.model.board import BoardSnapshot, CopperLayer
from kipi.core.viz.layer_colors import DEFAULT_COPPER, coating_rgb, layer_key
from kipi.core.viz.polydata import triangles

# Copper layer colors, keyed like a KiCad color theme ("f", "in1".., "b"): KiCad's default
# theme until :func:`set_layer_colors` gives the PCB editor's.
_layer_colors: Dict[str, Tuple[int, int, int]] = dict(DEFAULT_COPPER)


def set_layer_colors(colors: Dict[str, Tuple[int, int, int]]):
    """Use these copper colors (0..255 RGB by theme key) for nets built from now on."""
    global _layer_colors
    _layer_colors = {**DEFAULT_COPPER, **colors}


def layer_color(layer: CopperLayer, n_layers: int) -> Tuple[float, float, float]:
    rgb = _layer_colors.get(layer_key(layer.id, layer.index, n_layers), (184, 115, 51))
    return tuple(v / 255.0 for v in rgb)


# GEOS triangulates a polygon in time that grows much faster than its size: a ground
# plane with thousands of antipads took seconds per layer. Polygons with more holes or
# vertices than these are cut into tiles first, which triangulate in a fraction of it.
TILE_MAX_HOLES = 8
TILE_MAX_VERTICES = 1000
_TILE_BATCH = 256          # tiles per triangulation call handed to a worker thread


_CUT_CLEARANCE = 1e-4      # mm a cut keeps from every vertex


def _cuts(lo: float, hi: float, n: int, taken: np.ndarray) -> np.ndarray:
    """``n + 1`` evenly spaced cut positions from ``lo`` to ``hi``, the inner ones moved
    off the vertex coordinates ``taken`` (sorted). A cut through a vertex leaves it on one
    tile's edge and not the other's: a T-junction, and an open seam in the slab."""
    cuts = np.linspace(lo, hi, n + 1)
    step = (hi - lo) / n
    for i in range(1, n):
        for k in range(200):
            c = cuts[i] + (k + 1) // 2 * (1 if k % 2 else -1) * 3 * _CUT_CLEARANCE
            j = np.searchsorted(taken, c)
            near = taken[max(0, j - 1):j + 1]
            if not len(near) or np.abs(near - c).min() > _CUT_CLEARANCE:
                if abs(c - cuts[i]) < step / 2:
                    cuts[i] = c
                break
    return cuts


def _tiles(poly) -> List[BaseGeometry]:
    """``poly`` cut by a grid into pieces of about :data:`TILE_MAX_HOLES` holes and
    :data:`TILE_MAX_VERTICES` vertices. Every cut runs right across, clear of the
    vertices, so neighboring tiles get the same vertices along it and their triangles
    join without T-junctions."""
    n = max(len(poly.interiors) / TILE_MAX_HOLES,
            shapely.get_num_coordinates(poly) / TILE_MAX_VERTICES)
    if n <= 1:
        return [poly]
    x0, y0, x1, y1 = poly.bounds
    w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    nx = max(1, round(math.sqrt(n * w / h)))
    ny = max(1, math.ceil(n / nx))
    xy = shapely.get_coordinates(poly)
    xs = _cuts(x0, x1, nx, np.sort(xy[:, 0]))
    ys = _cuts(y0, y1, ny, np.sort(xy[:, 1]))
    out = []
    for ya, yb in zip(ys[:-1], ys[1:]):          # strips first: each cell clips less
        for strip in iter_polygons(shapely.clip_by_rect(poly, x0, ya, x1, yb)):
            for xa, xb in zip(xs[:-1], xs[1:]):
                out.extend(iter_polygons(shapely.clip_by_rect(strip, xa, ya, xb, yb)))
    return out


def _cdt(polys: np.ndarray) -> np.ndarray:
    """The triangles of each polygon, as an (M, 3, 2) array of their corners."""
    tris = shapely.get_parts(shapely.constrained_delaunay_triangles(polys))
    return shapely.get_coordinates(tris).reshape(-1, 4, 2)[:, :3, :]


def triangulate(geom: BaseGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """Constrained Delaunay triangulation of a (multi)polygon.

    Returns ``(points[N,2], triangles[M,3])`` with shared vertices merged.
    """
    from kipi.core.geometry.parallel import pmap
    pieces = np.array([t for poly in iter_polygons(geom) for t in _tiles(poly)], dtype=object)
    if len(pieces) == 0:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    bad = ~shapely.is_valid(pieces)
    if bad.any():            # clipping is quick but not guaranteed clean
        fixed = pmap(shapely.make_valid, np.array_split(pieces[bad], min(bad.sum(), 32)))
        pieces = np.array([*pieces[~bad], *(p for gs in fixed for g in gs
                                            for p in iter_polygons(g))], dtype=object)
    batches = [pieces[i:i + _TILE_BATCH] for i in range(0, len(pieces), _TILE_BATCH)]
    coords = np.concatenate(pmap(_cdt, batches)).reshape(-1, 2)
    if len(coords) == 0:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    # Tiles meet along their cuts; merging their vertices joins them up again. Each
    # (x, y) is taken as one complex number: unique over those sorts in a fraction of
    # the time unique over rows takes, to the same result.
    xy = np.ascontiguousarray(np.round(coords, 7)).view(np.complex128).ravel()
    uniq, inverse = np.unique(xy, return_inverse=True)
    return np.column_stack([uniq.real, uniq.imag]), inverse.reshape(-1, 3)


def sheet(geom: BaseGeometry, z: float) -> Optional[pv.PolyData]:
    pts2, tris = triangulate(geom)
    if len(tris) == 0:
        return None
    pts = np.column_stack([pts2[:, 0], -pts2[:, 1], np.full(len(pts2), z)])
    return triangles(pts, tris)


def slab_from_triangles(pts2: np.ndarray, tris: np.ndarray, z_bottom: float,
                        thickness: float) -> Tuple[pv.PolyData, np.ndarray]:
    """Closed slab from a 2D triangulation in KiCad coordinates (y is flipped here).

    Bottom and top faces reuse the triangles; side walls are quads on the boundary
    edges (edges used by exactly one triangle). Winding is made consistent so all
    normals point outward, which keeps shading and volume correct.

    Returns the mesh and, for each output face, the index of the source triangle, so
    per-triangle data can be carried onto the slab. Points are the 2D points twice
    (bottom copy then top copy), so per-point data maps with ``np.tile(data, 2)``.
    """
    # Mesh coordinates flip y, so make triangles counter-clockwise in (x, -y).
    xy = np.column_stack([pts2[:, 0], -pts2[:, 1]])
    a, b, c = xy[tris[:, 0]], xy[tris[:, 1]], xy[tris[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    tris = tris.copy()
    cw = cross < 0
    tris[cw] = tris[cw][:, ::-1]
    n = len(xy)
    m = len(tris)
    pts = np.vstack([np.column_stack([xy, np.full(n, z_bottom)]),
                     np.column_stack([xy, np.full(n, z_bottom + thickness)])])
    bottom = tris[:, ::-1]                       # faces down
    top = tris + n                               # faces up
    # Directed edges of CCW triangles; boundary edges have no reversed twin.
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e_tri = np.tile(np.arange(m), 3)
    key = e[:, 0].astype(np.int64) * n + e[:, 1]
    rkey = e[:, 1].astype(np.int64) * n + e[:, 0]
    # A sort and a binary search find the twins several times faster than np.isin.
    key.sort()
    at = np.searchsorted(key, rkey)
    is_boundary = at == len(key)
    inside = ~is_boundary
    is_boundary[inside] = key[at[inside]] != rkey[inside]
    boundary = e[is_boundary]
    wall_tri = e_tri[is_boundary]
    i, j = boundary[:, 0], boundary[:, 1]
    # Outward wall for a CCW boundary edge i->j: (i, j, j+n) and (i, j+n, i+n).
    walls = np.concatenate([np.column_stack([i, j, j + n]), np.column_stack([i, j + n, i + n])])
    allf = np.concatenate([bottom, top, walls])
    source = np.concatenate([np.arange(m), np.arange(m), wall_tri, wall_tri])
    return triangles(pts, allf), source


def slab(geom: BaseGeometry, z_bottom: float, thickness: float) -> Optional[pv.PolyData]:
    """Closed copper slab for a shapely (multi)polygon."""
    pts2, tris = triangulate(geom)
    if len(tris) == 0:
        return None
    if thickness <= 0:
        return sheet(geom, z_bottom)
    return slab_from_triangles(pts2, tris, z_bottom, thickness)[0]


@dataclass
class LayerMesh:
    net: str
    layer_id: int
    mesh: pv.PolyData
    color: Tuple[float, float, float]
    area_mm2: float


@dataclass
class NetScene:
    """Meshes for one net, so the viewer can add or remove nets independently."""
    net: str
    copper: List[LayerMesh] = field(default_factory=list)
    vias: Optional[pv.PolyData] = None
    pads: Dict[int, pv.PolyData] = field(default_factory=dict)   # layer id -> pad tops


@dataclass
class SceneData:
    """Board body plus any number of nets."""
    nets: Dict[str, NetScene] = field(default_factory=dict)
    board: Optional[pv.PolyData] = None
    outline_lines: Optional[pv.PolyData] = None


def via_tubes(snap: BoardSnapshot, nets: List[str], n_sides: int = 12) -> Optional[pv.PolyData]:
    """Via barrels of the given nets as one mesh.

    A unit cylinder is scaled and translated per via with numpy, which is much
    faster than building and merging one cylinder per via.
    """
    st = snap.stackup
    rows = []   # x, y, z_bottom, z_top, radius
    for v in (v for n in nets for v in snap.items_on_net(n)[1]):
        layers = [st.layer(l) for l in v.layers if st.layer(l) is not None]
        if not layers:
            continue
        z_top = max(l.z_top_mm for l in layers)
        z_bot = min(l.z_bottom_mm for l in layers)
        rows.append((v.position[0], -v.position[1], z_bot, max(z_top, z_bot + 1e-3),
                     max(v.drill_mm, 0.05) / 2.0))
    if not rows:
        return None
    a = np.asarray(rows)
    unit = pv.Cylinder(center=(0, 0, 0.5), direction=(0, 0, 1), radius=1.0, height=1.0,
                       resolution=n_sides, capping=True).triangulate()
    up = np.asarray(unit.points)                                   # (P, 3)
    uf = unit.faces.reshape(-1, 4)[:, 1:]                          # (F, 3) triangles
    n, P = len(a), len(up)
    pts = np.empty((n, P, 3))
    pts[:, :, 0] = a[:, None, 0] + up[None, :, 0] * a[:, None, 4]
    pts[:, :, 1] = a[:, None, 1] + up[None, :, 1] * a[:, None, 4]
    pts[:, :, 2] = a[:, None, 2] + up[None, :, 2] * (a[:, None, 3] - a[:, None, 2])
    tris = uf[None, :, :] + (np.arange(n) * P)[:, None, None]      # (n, F, 3)
    return triangles(pts, tris)


def pad_tops(snap: BoardSnapshot, nets: List[str]) -> Dict[int, pv.PolyData]:
    """Pad shapes as sheets sitting just above their copper layer (for highlighting)."""
    st = snap.stackup
    out: Dict[int, pv.PolyData] = {}
    for cu in st.copper:
        polys = []
        for p in (p for n in nets for p in snap.items_on_net(n)[2]):
            for poly in p.shapes.get(cu.id, []):
                g = poly_to_shapely(poly)
                if g is not None and not g.is_empty:
                    polys.append(g)
        if not polys:
            continue
        geom = shapely.union_all(polys)
        holes = [h for h in (drill_holes(snap, n, cu.id) for n in nets) if h is not None]
        if holes:   # each net's holes are already one geometry
            geom = geom.difference(holes[0] if len(holes) == 1 else shapely.union_all(holes))
        lift = 0.002
        z = cu.z_top_mm + lift if cu.index == 0 else (
            cu.z_bottom_mm - lift if cu.index == len(st.copper) - 1 else cu.z_top_mm + lift)
        m = sheet(geom, z)
        if m is not None:
            out[cu.id] = m
    return out


BODY_GAP_FRACTION = 0.01    # body inset from the outer copper, as a share of its span


def _body_span(stackup) -> Tuple[float, float]:
    """``(z_bottom, thickness)`` of the laminate body.

    The body sits inside the outer copper with a small gap: a face coplanar with copper
    z-fights with it (speckle that shifts as the view rotates). Falls back to the full
    board thickness when the stackup leaves no room between the outer copper.
    """
    t = stackup.board_thickness_mm
    tol = 1e-6
    lo = max([c.z_top_mm for c in stackup.copper if c.z_bottom_mm <= tol], default=0.0)
    hi = min([c.z_bottom_mm for c in stackup.copper if c.z_top_mm >= t - tol], default=t)
    if hi - lo <= 0:
        return 0.0, t
    gap = BODY_GAP_FRACTION * (hi - lo)
    return lo + gap, hi - lo - 2 * gap


def board_body(snap: BoardSnapshot) -> Tuple[Optional[pv.PolyData], Optional[pv.PolyData]]:
    """Translucent laminate body and its outline edges."""
    if not snap.outline:
        return None, None
    geoms = [poly_to_shapely(p) for p in snap.outline]
    geom = shapely.union_all([g for g in geoms if g is not None])
    t = snap.stackup.board_thickness_mm
    body = slab(geom, *_body_span(snap.stackup))
    lines = []
    for poly in iter_polygons(geom):
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            for z in (0.0, t):
                pts = np.column_stack([xy[:, 0], -xy[:, 1], np.full(len(xy), z)])
                lines.append(pv.lines_from_points(pts))
    return body, (pv.merge(lines) if lines else None)


COATING_GAP_MM = 0.002     # coatings off the copper, so their faces do not z-fight


def board_coatings(snap: BoardSnapshot) -> List[Tuple[str, pv.PolyData, Tuple[float, ...]]]:
    """The solder masks and silkscreens as slabs over the board (#82): ``(kind, mesh,
    color)`` for each. A mask is open over the pads of the copper under it; a silkscreen
    is what it prints (see :func:`~kipi.core.geometry.silkscreen.printed_silk`)."""
    from kipi.core.geometry.silkscreen import mask_openings, printed_silk
    st = snap.stackup
    if not snap.outline or not st.coatings or not st.copper:
        return []
    from kipi.core.geometry.parallel import pmap
    board = shapely.union_all([g for g in (poly_to_shapely(p) for p in snap.outline)
                               if g is not None])

    def coat(side):
        out = []
        cu = st.copper[0] if side == "top" else st.copper[-1]
        z = cu.z_top_mm if side == "top" else cu.z_bottom_mm
        openings = mask_openings(snap, side)
        z += COATING_GAP_MM if side == "top" else -COATING_GAP_MM
        for c in st.coatings_on(side):
            if c.kind == "mask":
                geom = board if openings is None else board.difference(openings)
            else:
                geom = printed_silk(snap, side, board, openings)
            t = c.thickness_mm
            m = slab(geom, z if side == "top" else z - t, t) if geom is not None else None
            if m is not None:
                out.append((c.kind, m, coating_rgb(c)))
            z += t if side == "top" else -t
        return out

    return [c for side in pmap(coat, ("top", "bottom")) for c in side]


def build_net_scene(snap: BoardSnapshot, net: str, layers: Dict[int, BaseGeometry],
                    progress=None) -> NetScene:
    """Build the meshes for one net from its per-layer copper geometry."""
    from kipi.core.geometry.parallel import pmap
    st = snap.stackup
    n = len(st.copper)
    ns = NetScene(net=net)

    def layer(item):
        lid, geom = item
        cu = st.layer(lid)
        if cu is None:
            return None
        m = slab(geom, cu.z_bottom_mm, cu.thickness_mm)
        if m is None:
            return None
        if progress is not None:
            progress.debug(f"  mesh {net} / {cu.name}: {m.n_cells} cells")
            progress.check_cancel()
        return LayerMesh(net=net, layer_id=lid, mesh=m, color=layer_color(cu, n),
                         area_mm2=geom.area)

    ns.copper = [lm for lm in pmap(layer, layers.items()) if lm is not None]
    ns.vias = via_tubes(snap, [net])
    ns.pads = pad_tops(snap, [net])
    return ns


# Nets with at least this share of the board's items (the ground, a big supply) are
# built one at a time on a thread of their own, each spreading its layers and tiles over
# the cores; the many small ones are built side by side meanwhile.
BIG_NET_SHARE = 0.02


def build_net_scenes(snap: BoardSnapshot, nets: List[str],
                     copper: Dict[str, Dict[int, BaseGeometry]], progress=None):
    """Yield the :func:`build_net_scene` of each of ``nets``, as each is ready. Copper not
    yet in ``copper`` (``net -> layer -> geometry``) is worked out and kept there."""
    import queue
    import threading

    from kipi.core.geometry.copper import net_copper
    from kipi.core.geometry.parallel import WORKERS, pmap
    from kipi.core.progress import NULL

    def one(net):
        if progress is not None:
            progress.check_cancel()
        if net not in copper:
            copper[net] = net_copper(snap, net, progress or NULL)
        return build_net_scene(snap, net, copper[net], progress)

    size = {n: sum(len(items) for items in snap.items_on_net(n)) for n in nets}
    total = max(1, sum(size.values()))
    order = sorted(nets, key=lambda n: -size[n])
    big = [n for n in order if size[n] >= BIG_NET_SHARE * total]
    small = order[len(big):]

    # A small net takes a few milliseconds, mostly Python holding the GIL; the ground
    # takes seconds, mostly GEOS without it. Built one after the other, the cores idle
    # through both.
    ready: "queue.SimpleQueue" = queue.SimpleQueue()
    stop = threading.Event()
    done = object()

    def build_big():
        try:
            for net in big:
                if stop.is_set():
                    break
                ready.put(one(net))
        except BaseException as e:  # noqa: BLE001 - raised again in the caller's thread
            ready.put(e)
        finally:
            ready.put(done)

    finished = False

    def take(block: bool):
        nonlocal finished
        while not finished:
            try:
                got = ready.get(block=block)
            except queue.Empty:
                return
            if got is done:
                finished = True
            elif isinstance(got, BaseException):
                raise got
            else:
                yield got

    worker = threading.Thread(target=build_big, name="kipi-big-nets", daemon=True)
    worker.start()
    try:
        step = 4 * WORKERS
        for i in range(0, len(small), step):
            yield from pmap(one, small[i:i + step])
            yield from take(block=False)
        yield from take(block=True)
    finally:
        stop.set()
        worker.join()


def build_scene(snap: BoardSnapshot, copper: Dict[str, Dict[int, BaseGeometry]],
                progress=None) -> SceneData:
    """Build the board body and every net in ``copper`` (``net -> layer -> geometry``)."""
    scene = SceneData()
    for net, layers in copper.items():
        scene.nets[net] = build_net_scene(snap, net, layers, progress)
    scene.board, scene.outline_lines = board_body(snap)
    return scene
