"""DC solver entry points: mesh -> solve -> post-process for one net."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional

from kipi.core.model.board import BoardSnapshot, Pad
from kipi.core.progress import NULL, ProgressReporter
from kipi.dc.solver.setup import CurrentLoad, NetProblem, SeriesLink, SolveSettings, VoltageSource

if TYPE_CHECKING:
    from kipi.core.mesh.mesher import NetMesh
    from kipi.dc.solver.fem import Solution
    from kipi.dc.solver.post import Fields

# The mesher, solver and post-processing load SciPy; they are imported when first used, so
# that the setup's dataclasses (kipi.dc.solver.setup, which the setup panels read) load at
# KiPI's start without them.
_LAZY = {"NetMesh": "kipi.core.mesh.mesher", "mesh_nets": "kipi.core.mesh.mesher",
         "Solution": "kipi.dc.solver.fem", "SolveError": "kipi.dc.solver.fem",
         "solve": "kipi.dc.solver.fem", "Fields": "kipi.dc.solver.post",
         "compute_fields": "kipi.dc.solver.post"}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value

__all__ = ["NetResult", "run", "pads_for", "NetProblem", "VoltageSource", "CurrentLoad",
           "SeriesLink", "SolveSettings", "SolveError"]


@dataclass
class NetResult:
    mesh: NetMesh
    solution: Solution
    fields: Fields


def run(snap: BoardSnapshot, problem: NetProblem,
        progress: ProgressReporter = NULL) -> NetResult:
    from kipi.core.mesh.mesher import mesh_nets
    from kipi.dc.solver.fem import solve
    from kipi.dc.solver.post import compute_fields
    s = problem.settings
    with progress.timed(f"Meshing {problem.net}"):
        mesh = mesh_nets(snap, problem.nets, problem.terminal_pad_ids(), name=problem.net,
                         max_edge_mm=s.max_edge_mm,
                        min_angle_deg=s.min_angle_deg, simplify_mm=s.simplify_mm,
                        circle_quad_segs=s.circle_quad_segs, min_edge_mm=s.min_edge_mm,
                        grading=s.grading, progress=progress)
    sol = solve(snap, mesh, problem, progress)
    with progress.timed("Post-processing"):
        fields = compute_fields(sol)
        sm = fields.summary
        boost = f", max boost {sm['max_boost'] * 1e3:.3f} mV" if sm["max_boost"] > 0 else ""
        progress.info(f"Max drop {sm['max_drop'] * 1e3:.3f} mV{boost}, peak current density "
                      f"{sm['j_max']:.2f} A/mm^2, copper loss {sm['power_loss'] * 1e3:.3f} mW "
                      f"(energy balance error {sm['energy_balance_error']:.1e})")
    return NetResult(mesh, sol, fields)


def pads_for(snap: BoardSnapshot, net: str, reference: str,
             numbers: Optional[Iterable[str]] = None) -> List[Pad]:
    """Pads of footprint ``reference`` on ``net`` (optionally only the given pad numbers)."""
    _, _, pads, _ = snap.items_on_net(net)
    wanted = set(numbers) if numbers is not None else None
    out = [p for p in pads if p.footprint == reference and (wanted is None or p.number in wanted)]
    # A pin may be several pads (a microphone's ground ring in four parts). One with no
    # copper shape at all carries nothing; it is left out while another pad of the pin
    # has copper, so the mesh is not asked for copper that is not there.
    shaped = {p.number for p in out if p.shapes}
    out = [p for p in out if p.shapes or p.number not in shaped]
    if not out:
        which = f" pins {sorted(wanted)}" if wanted else ""
        from kipi.dc.solver.fem import SolveError
        raise SolveError(f"{reference}{which} has no pads on net {net}")
    return sorted(out, key=lambda p: (len(p.number), p.number))
