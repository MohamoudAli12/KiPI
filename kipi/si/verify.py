"""A 3D check of KiPI's channel model: openEMS (:mod:`kipi.si.openems`) on the region
around a via of a channel, beside the 2.5D model (:func:`kipi.si.extract.build_channel`)
of the same copper cut at the same edges, both seen from the same ports where the traces
leave the region (to the nearest plane under them, 50 Ω)."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from kipi.core.progress import NULL, ProgressReporter
from kipi.si import sparams

PAIR_VIA_MM = 2.0          # a pair's other via within this of the one picked is in the region


@dataclass
class ViaCheck:
    channel: str
    via_id: str
    region: Tuple[float, float, float, float]
    port_names: List[str]
    freqs: np.ndarray
    s_em: np.ndarray                     # (F, P, P) openEMS
    s_kipi: np.ndarray                   # (F, P, P) KiPI's model
    cells: int = 0
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def worst_db(self, i: int, j: int) -> float:
        """The largest |S_ij| difference in dB where both are above −40 dB (a null in
        one, a few % of a small reflection, would otherwise read as tens of dB)."""
        a, b = sparams.db(self.s_em[:, i, j]), sparams.db(self.s_kipi[:, i, j])
        sel = np.minimum(a, b) > -40
        return float(np.max(np.abs(a - b)[sel])) if sel.any() else 0.0

    def through(self) -> Optional[Tuple[int, int]]:
        """The first port pair along one net (a through path), if any."""
        nets = [n.split("@")[0] for n in self.port_names]
        for i in range(len(nets)):
            for j in range(i + 1, len(nets)):
                if nets[i] == nets[j]:
                    return j, i
        return None

    def summary_text(self) -> str:
        bits = [f"openEMS check of {self.channel} around {self.via_id}: "
                f"{self.cells / 1e6:.2f} M cells, {self.seconds:.0f} s"]
        th = self.through()
        if th is not None:
            bits.append(f"through |ΔS| ≤ {self.worst_db(*th):.2f} dB")
        bits.append(f"reflection |ΔS| ≤ {self.worst_db(0, 0):.2f} dB")
        return "; ".join(bits)

    def summary_html(self) -> str:
        th = self.through()
        rows = []
        if th is not None:
            d = self.worst_db(*th)
            rows.append(("Through (S{0}{1})".format(th[0] + 1, th[1] + 1), d, 1.0))
        rows.append(("Reflection (S11)", self.worst_db(0, 0), 3.0))
        out = [f"<b>openEMS</b>: {self.cells / 1e6:.2f} M cells, {self.seconds:.0f} s, to "
               f"{self.freqs[-1] / 1e9:.3g} GHz"]
        for label, d, lim in rows:
            col = "#60d060" if d <= lim else "#e0c040" if d <= 2 * lim else "#ff6060"
            out.append(f'{label}: KiPI within <span style="color:{col}">{d:.2f} dB</span> '
                       f"of openEMS")
        out.append("A small reflection (below −10 dB) is hard to compare: openEMS's lumped ports "
                   "add their own, a few dB on a plain trace, so compare its shape more than its "
                   "level.")
        out += [html.escape(n) for n in self.notes]
        return "<br>".join(out)


def _region(snap, via_id: str, nets, margin: float):
    from kipi.si.openems import via_region
    via = next(v for v in snap.vias if v.id == via_id)
    ids = [via_id]
    for v in snap.vias:
        if v.id != via_id and v.net in nets and v.net != via.net and \
                math.dist(v.position, via.position) <= PAIR_VIA_MM:
            ids.append(v.id)
    return via_region(snap, ids, margin)


def kipi_model(board, nets: List[str], region, ports, freqs, opts) -> np.ndarray:
    """KiPI's S (F, P, P) of ``nets`` inside ``region``, at the openEMS ``ports``."""
    from kipi.si.extract import SNAP_MM, build_channel
    groups = [list(nets)]
    chan = build_channel(board, groups, _with_region(opts, region))
    ckt = chan.circuit
    for p in ports:
        best, key = None, None
        for k in ckt.nodes:
            if isinstance(k, tuple) and k and k[0] == "pt" and k[1] == p.net and k[2] == p.layer:
                d = math.hypot(k[3] * SNAP_MM - p.xy[0], k[4] * SNAP_MM - p.xy[1])
                if best is None or d < best:
                    best, key = d, k
        if key is None or best > 0.05:
            raise ValueError(f"KiPI's model has no node at the port {p.name}")
        ckt.add_port(p.name, key)
    return sparams.s_from_y(ckt.solve_y(freqs), 50.0)


def _with_region(opts, region):
    import copy
    o = copy.copy(opts)
    o.region = region
    o.planes = "open"           # the 3D model's planes run on into its absorbing walls
    return o


def verify_via(snap, setup, channel: str, via_id: str, workdir, progress: ProgressReporter = NULL,
               board=None) -> ViaCheck:
    from kipi.si import openems
    from kipi.si.board import SIBoard
    from kipi.si.extract import BuildOptions
    from kipi.si.solver import roughness
    ch = next((c for c in setup.si_channels if c.name == channel), None)
    if ch is None:
        raise ValueError(f"no channel {channel!r}")
    s = setup.si
    board = board or SIBoard(snap, setup.ac.loss_tangent, s.etch_mm)
    nets = [n for n in (ch.net, ch.partner) if n]
    via = next((v for v in snap.vias if v.id == via_id), None)
    if via is None:
        raise ValueError("the via is no longer on the board")
    if via.net not in nets:
        nets = [via.net]
    region = _region(snap, via_id, nets, s.em_margin_mm)
    settings = openems.EMSettings(f_max_hz=s.em_f_max_hz, fine_mm=s.em_fine_mm,
                                  end_db=s.em_end_db)
    with progress.timed("Building the openEMS model"):
        model = openems.build_model(board, region, nets, settings)
    if len(model.ports) < 2:
        raise ValueError("fewer than two traces leave the region: widen the region margin")
    em = openems.run(model, workdir, progress)
    with progress.timed("KiPI's model of the same region"):
        opts = BuildOptions(min(s.segment_mm, 0.5), s.window_mm, roughness(s), s.via_models)
        s_k = kipi_model(board, nets, region, model.ports, em.freqs_hz, opts)
    return ViaCheck(channel, f"{via.net} via at ({via.position[0]:.2f}, {via.position[1]:.2f})",
                    region, [p.name for p in model.ports], em.freqs_hz, em.s, s_k, em.cells,
                    em.seconds, list(em.notes) + list(model.notes if hasattr(model, "notes")
                                                      else []))
