"""Triangle meshes that do not slow VTK down as they pile up.

VTK 9.4 and later give each data array a Python subclass. When Python lets go of the
wrapper of a VTK object that is still alive and carries Python state, VTK keeps a
stand-in for it, and then searches all its stand-ins each time an array is made or
fetched. ``numpy_to_vtk``, pyvista's ``wrap`` and ``cell_data[...] =`` each leave such
stand-ins behind for every mesh; with the thousands of net, pad and part meshes KiPI
holds, each array cost about a millisecond. Meshes made here keep every wrapper they
create alive alongside them, so they leave none, and keep numpy views of their points
and triangles, which :func:`arrays` returns without asking VTK.

Each array's memory is VTK's own, never numpy's: VTK hands arrays on to copies, filters
and actors that can outlive the mesh, and memory freed under them crashes the renderer.
"""

from __future__ import annotations

import weakref
from typing import Dict, Tuple

import numpy as np
import pyvista as pv
from vtkmodules.util.numpy_support import create_vtk_array, get_vtk_array_type
from vtkmodules.vtkCommonCore import vtkIdTypeArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

_ID = np.int64 if vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32


# id(mesh) -> what it holds; kept off the mesh itself so that it still pickles.
_KEPT: Dict[int, tuple] = {}


def _keep(mesh: pv.PolyData, *objs):
    """Hold ``objs`` for as long as ``mesh`` lives."""
    key = id(mesh)
    if key not in _KEPT:
        weakref.finalize(mesh, _KEPT.pop, key, None)
    _KEPT[key] = _KEPT.get(key, ()) + objs


def _vtk_array(values, dtype=None):
    """A copy of ``values`` (1-D or 2-D) in a VTK array, and a numpy view of it."""
    values = np.asarray(values, dtype=dtype)
    a = create_vtk_array(get_vtk_array_type(values.dtype))
    a.SetNumberOfComponents(values.shape[1] if values.ndim == 2 else 1)
    a.SetNumberOfTuples(values.shape[0] if values.ndim else 1)
    view = np.frombuffer(a, dtype=values.dtype).reshape(values.shape)
    view[...] = values
    return a, view


def triangles(points, tris) -> pv.PolyData:
    """A mesh of ``points`` (N, 3) and the triangles ``tris`` (M, 3) indexing them,
    on its own copies of both."""
    points = np.asarray(points)
    vtk_pts, points = _vtk_array(points.reshape(-1, 3), points.dtype
                                 if points.dtype in (np.float32, np.float64) else np.float64)
    vtk_conn, conn = _vtk_array(np.asarray(tris).reshape(-1), _ID)
    vtk_off, offsets = _vtk_array(np.arange(0, len(conn) + 1, 3), _ID)
    pts = vtkPoints()
    pts.SetData(vtk_pts)
    cells = vtkCellArray()
    cells.SetData(vtk_off, vtk_conn)
    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    mesh = pv.PolyData()
    mesh.ShallowCopy(poly)
    _keep(mesh, points, conn, offsets, vtk_pts, vtk_off, vtk_conn, pts, cells, poly)
    mesh._kipi_arrays = (points, conn.reshape(-1, 3))
    return mesh


def set_cell_array(mesh: pv.PolyData, name: str, values) -> None:
    """``mesh.cell_data[name] = values``, on a copy of ``values``."""
    _set(mesh, mesh.GetCellData(), name, values)


def set_point_array(mesh: pv.PolyData, name: str, values) -> None:
    """``mesh.point_data[name] = values``, on a copy of ``values``."""
    _set(mesh, mesh.GetPointData(), name, values)


def _set(mesh, data, name: str, values):
    a, values = _vtk_array(values)
    a.SetName(name)
    data.AddArray(a)
    _keep(mesh, values, a, data)


def arrays(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    """``(points (N, 3), triangles (M, 3))`` of a mesh made by :func:`triangles`, or of
    any other mesh (more slowly), triangulating it first if it needs to be."""
    kept = getattr(mesh, "_kipi_arrays", None)
    if kept is not None and len(kept[0]) == mesh.n_points and len(kept[1]) == mesh.n_cells:
        return kept
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    return np.asarray(mesh.points), mesh.regular_faces
