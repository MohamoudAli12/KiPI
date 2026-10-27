import pytest

pv = pytest.importorskip("pyvista")

from kipi.core.geometry.copper import net_copper
from kipi.core.viz.meshes import build_scene, triangulate
from synthetic import F, IN1, make_board


def test_triangulation_preserves_area():
    snap = make_board()
    g = net_copper(snap, "GND")[IN1]
    pts, tris = triangulate(g)
    a = pts[tris]
    area = 0.5 * abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1])
                     - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1])).sum()
    assert area == pytest.approx(g.area, rel=1e-9)


def test_a_plane_with_many_holes_is_tiled_into_one_closed_slab():
    # A ground plane full of antipads is cut into tiles to triangulate quickly; the
    # tiles' vertices are merged again, so the slab has no walls along the cuts.
    import shapely
    from kipi.core.viz import meshes
    holes = [shapely.Point(x, y).buffer(0.3).exterior.coords
             for x in range(2, 60, 2) for y in range(2, 40, 2)]
    plane = shapely.Polygon([(0, 0), (61, 0), (61, 41), (0, 41)], holes)
    assert len(plane.interiors) > 10 * meshes.TILE_MAX_HOLES
    assert len(meshes._tiles(plane)) > 1
    m = meshes.slab(plane, 0.0, 0.035)
    assert m.volume == pytest.approx(plane.area * 0.035, rel=1e-6)
    edges = m.extract_feature_edges(boundary_edges=True, non_manifold_edges=True,
                                    feature_edges=False, manifold_edges=False)
    assert edges.n_cells == 0


def test_segment_strokes_match_stroking_each_segment():
    import shapely
    from kipi.core.geometry.shapes import segment_strokes, stroke
    segs = [(0, 0, 1, 0, 0.2), (1, 0, 1, 1, 0.2), (1, 1, 0, 1, 0.2),   # one path
            (3, 0, 4, 0, 0.1), (4, 0, 4, 1, 0.3),                      # width changes
            (6, 0, 6, 0, 0.2), (7, 0, 8, 0, 0.0)]                      # a dot; no ink
    got = shapely.union_all(segment_strokes(segs))
    want = shapely.union_all([g for g in (stroke([s[:2], s[2:4]], s[4]) for s in segs)
                              if g is not None] + [shapely.Point(6, 0).buffer(0.1)])
    assert len(segment_strokes(segs)) == 4
    assert got.symmetric_difference(want).area < 1e-3 * want.area


def test_scene_builds_closed_slabs():
    snap = make_board()
    copper = {n: net_copper(snap, n) for n in ("+3V3", "GND")}
    scene = build_scene(snap, copper)
    assert {n: len(ns.copper) for n, ns in scene.nets.items()} == {"+3V3": 4, "GND": 2}
    cu = next(lm for lm in scene.nets["+3V3"].copper if lm.layer_id == F)
    layer = snap.stackup.layer(F)
    assert cu.mesh.volume == pytest.approx(cu.area_mm2 * layer.thickness_mm, rel=1e-6)
    assert cu.mesh.bounds[4] == pytest.approx(layer.z_bottom_mm)
    assert scene.nets["+3V3"].vias is not None and scene.nets["+3V3"].vias.n_cells > 0
    assert scene.nets["GND"].vias is None         # GND has no vias
    assert set(scene.nets["+3V3"].pads) == {F}
    assert scene.board is not None


def test_board_body_clears_every_copper_face():
    """A body face coplanar with copper z-fights with it: speckle that moves as the view
    rotates (issue #36)."""
    from kipi.core.viz.meshes import board_body
    snap = make_board()
    body, _ = board_body(snap)
    z_lo, z_hi = body.bounds[4], body.bounds[5]
    faces = [z for c in snap.stackup.copper for z in (c.z_bottom_mm, c.z_top_mm)]
    assert min(abs(z - zf) for zf in faces for z in (z_lo, z_hi)) > 1e-3
    outer = sorted(snap.stackup.copper, key=lambda c: c.z_bottom_mm)
    assert outer[0].z_top_mm < z_lo and z_hi < outer[-1].z_bottom_mm
