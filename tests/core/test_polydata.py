import gc
import pickle

import numpy as np
import pytest

pytest.importorskip("pyvista")

from kipi.core.viz import polydata  # noqa: E402


def test_a_mesh_owns_its_points_whatever_their_memory_order():
    # Column-major points flatten to a copy; VTK must not be left pointing at it.
    def make():
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])[:, [0, 2, 1]]
        assert not pts.flags.c_contiguous
        m = polydata.triangles(pts * 2.0, [[0, 1, 2]])
        polydata.set_point_array(m, "n", np.eye(3)[:, [2, 1, 0]])
        return m
    m = make()
    gc.collect()
    np.zeros((1000, 3))                      # reuse any memory freed meanwhile
    assert m.points.tolist() == [[2, 6, 4], [8, 12, 10], [14, 18, 16]]
    assert m.point_data["n"].tolist() == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]


def test_arrays_are_the_meshs_own_and_it_pickles():
    m = polydata.triangles(np.random.default_rng(0).random((4, 3)), [[0, 1, 2], [1, 2, 3]])
    polydata.set_cell_array(m, "rgb", np.array([[1, 2, 3], [4, 5, 6]], np.uint8))
    pts, tris = polydata.arrays(m)
    assert np.array_equal(pts, m.points) and np.array_equal(tris, m.regular_faces)
    back = pickle.loads(pickle.dumps(m))
    assert np.array_equal(back.points, m.points)
    assert back.cell_data["rgb"].tolist() == [[1, 2, 3], [4, 5, 6]]


def test_what_a_mesh_holds_goes_with_it():
    m = polydata.triangles(np.zeros((3, 3)), [[0, 1, 2]])
    key = id(m)
    assert key in polydata._KEPT
    del m
    gc.collect()
    assert key not in polydata._KEPT


def test_a_copy_vtk_keeps_outlives_the_mesh():
    # An actor's mapper holds the VTK data, not the Python mesh: the memory is VTK's.
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    import pyvista as pv
    m = polydata.triangles(np.full((300, 3), 7.0), [[0, 1, 2]] * 100)
    polydata.set_cell_array(m, "rgb", np.full((100, 3), 9, np.uint8))
    held = vtkPolyData()
    held.ShallowCopy(m)
    del m
    gc.collect()
    junk = [np.full((300, 3), 1.0) for _ in range(100)]   # noqa: F841 - reuse freed memory
    c = pv.wrap(held)
    assert (c.points == 7.0).all()
    assert (c.cell_data["rgb"] == 9).all()
    assert (c.regular_faces == [0, 1, 2]).all()
