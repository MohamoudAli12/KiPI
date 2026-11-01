"""What the 3D view draws for Signal Integrity: each line section of the Quick Scan or of
a solved channel coloured by how far its impedance is from the target, drawn with the
Coupled Noise overlay (:class:`kipi.noise.viz.CouplingOverlay`, values in Ω here)."""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

from kipi.noise.viz import CouplingOverlay

LIMIT_OHM = 15.0           # the colour scale: ± this off the target


def _z(snap, layer) -> float:
    lay = snap.stackup.layer(layer)
    return lay.z_mid_mm if lay is not None else 0.0


def _span(snap, layer) -> tuple:
    lay = snap.stackup.layer(layer)
    return (lay.z_bottom_mm, lay.z_top_mm) if lay is not None else (0.0, 0.0)


def _overlay(title, segs, vals, widths, spans, limit, note) -> CouplingOverlay:
    return CouplingOverlay(title, np.array(segs, float), np.array(vals, float),
                           np.zeros((0, 3)), np.zeros(0), limit, note,
                           widths=np.array(widths, float), z_spans=np.array(spans, float))


def scan_overlay(snap, scan, nets: Optional[Sequence[str]] = None) -> Optional[CouplingOverlay]:
    """The Quick Scan's segments of ``nets`` (all when None): Z0 (or a pair's Zdiff)
    less its target."""
    if scan is None:
        return None
    by = scan.by_net()
    segs, vals, widths, spans = [], [], [], []
    for s in scan.segments:
        if nets is not None and s.net not in nets:
            continue
        ns = by.get(s.net)
        if ns is None:
            continue
        z = s.zdiff if (ns.partner and s.zdiff is not None) else s.z0
        target = ns.target or 50.0
        if not z or not np.isfinite(z):
            continue
        zz = _z(snap, s.layer)
        segs.append([(s.a[0], s.a[1], zz), (s.b[0], s.b[1], zz)])
        vals.append(z - target)
        widths.append(s.width_mm)
        spans.append(_span(snap, s.layer))
    if not segs:
        return None
    what = "the scanned nets" if nets is None else ", ".join(nets)
    return _overlay("Z − target (Ω)", segs, vals, widths, spans, LIMIT_OHM,
                    f"Impedance of {what}: blue under the target, red over it")


def channel_overlay(snap, result) -> Optional[CouplingOverlay]:
    """A solved channel's line sections: each conductor's Z0 (a pair: its Zdiff) less the
    interface's target. A pair's leg where it runs on its own (no Zdiff) is grey."""
    if result is None or not result.sections:
        return None
    ch = result.channel
    target = ch.proto().z_target_ohm
    pair = bool(ch.partner)
    segs, vals, widths, spans = [], [], [], []
    for sec in result.sections:
        for k, seg in enumerate(sec.segs):
            if seg.net in ch.aggressors:
                continue
            if pair:
                z = sec.zdiff if sec.zdiff is not None else np.nan
            else:
                z = sec.z0[min(k, len(sec.z0) - 1)]
            zz = _z(snap, seg.layer)
            segs.append([(seg.a[0], seg.a[1], zz), (seg.b[0], seg.b[1], zz)])
            vals.append(z - target)
            widths.append(seg.width_mm)
            spans.append(_span(snap, seg.layer))
    if not segs:
        return None
    note = f"{ch.name}: blue under the {target:g} Ω target, red over it"
    if pair:
        note += "; grey where a leg runs without its partner (no Zdiff)"
    return _overlay(f"{'Zdiff' if pair else 'Z0'} − {target:g} Ω", segs, vals, widths, spans,
                    LIMIT_OHM, note)


def end_labels(snap, si, ch) -> list:
    """Where a channel's Transmitter and Receiver are, for the 3D view: ``("TX U1", xyz)``
    and ``("RX U2", xyz)`` over the middle of each part's pads on the channel's nets, on the
    side the part sits. Empty when the ends can't be told (a net with no parts)."""
    from kipi.dc.viz.markers import _pad_z, part_sides
    from kipi.si.extract import follow_series
    from kipi.si.model import not_fitted
    from kipi.si.solver import ends
    if snap is None or ch is None or not ch.net:
        return []
    skip = not_fitted(snap, si, ch)
    nets = [n for x in (ch.net, ch.partner) if x
            for n in (follow_series(snap, x, not_fitted=skip) if ch.follow_series else [x])]
    try:
        refs = ends(snap, dataclasses.replace(ch, not_fitted=skip), nets)
    except Exception:  # noqa: BLE001 (no ends to show; Solve says why)
        return []
    sides = part_sides(snap)
    out = []
    for text, ref in zip(("TX", "RX"), refs):
        pads = [p for n in nets for p in snap.items_on_net(n)[2] if p.footprint == ref]
        if not pads:
            continue
        side = sides.get(ref, "top")
        zs = [_pad_z(snap, p, side) for p in pads]
        xy = np.mean([p.position for p in pads], axis=0)
        out.append((f"{text} {ref}", np.array([xy[0], xy[1], min(zs) if side == "bottom" else max(zs)])))
    return out
