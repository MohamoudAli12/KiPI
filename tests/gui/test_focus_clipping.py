"""Picking out nets (an SI channel's, say) keeps the whole board in view: the layer the
nets are drawn on again shares the board's camera, so it must not fit the camera's
clipping range to those nets alone."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

from kipi.gui.viewer3d import Viewer3D  # noqa: E402


def test_picked_out_nets_do_not_clip_the_board():
    pl = pv.Plotter(off_screen=True)
    board = pv.Plane(center=(0, 0, 0), i_size=100, j_size=100)
    pl.add_mesh(board)
    pl.camera.position = (0, -80, 25)            # low and close: the near edge is near
    pl.camera.focal_point = (0, 0, 0)
    pl.renderer.ResetCameraClippingRange()
    net = pv.Cube(center=(0, 0, 0.5), x_length=2, y_length=2, z_length=1)
    v = NS(plotter=pl, _focus_renderer=None, _focus_actors={}, _focus_nets=frozenset({"N"}),
           _hide_non_included=False, _layer_opacity=1.0, _zscale=1.0, _result=None,
           _coupling_actors=[], _batch=NS(meshes_of=lambda nets: {("cu", 0): net}))
    v._dimming = lambda: True
    v._apply_actor_visibility = lambda cat, actor: None
    for name in ("_focus_layer", "_clear_focus_actors", "_coupling_to_focus"):
        setattr(v, name, getattr(Viewer3D, name).__get__(v))
    Viewer3D._draw_focus(v)
    cam = pl.camera
    pos = np.array(cam.position)
    view = np.array(cam.focal_point) - pos
    view /= np.linalg.norm(view)
    depths = [float(np.dot(p - pos, view)) for p in board.points]
    near, far = cam.clipping_range
    assert near <= min(depths) and far >= max(depths)
    pl.close()


def test_an_overlay_is_drawn_over_the_picked_out_nets():
    """The picked-out nets are drawn again over everything: the impedance overlay goes on
    their layer too, or their copper hides it, and comes off it with them."""
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Plane(i_size=10, j_size=10))
    overlay = pl.add_mesh(pv.Line((0, 0, 0.1), (1, 0, 0.1)))
    net = pv.Cube(center=(0, 0, 0.05), x_length=1, y_length=0.2, z_length=0.05)
    v = NS(plotter=pl, _focus_renderer=None, _focus_actors={}, _focus_nets=frozenset({"N"}),
           _hide_non_included=False, _layer_opacity=1.0, _zscale=1.0, _result=None,
           _coupling_actors=[overlay], _batch=NS(meshes_of=lambda nets: {("cu", 0): net}))
    v._dimming = lambda: bool(v._focus_nets)
    v._apply_actor_visibility = lambda cat, actor: None
    for name in ("_focus_layer", "_clear_focus_actors", "_coupling_to_focus"):
        setattr(v, name, getattr(Viewer3D, name).__get__(v))
    Viewer3D._draw_focus(v)
    assert v._focus_renderer.GetActors().IsItemPresent(overlay)
    v._focus_nets = None
    Viewer3D._draw_focus(v)
    assert not v._focus_renderer.GetActors().IsItemPresent(overlay)
    pl.close()
