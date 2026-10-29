"""The Signal Integrity example: a small board with a clock net and a USB pair.

    python examples/signal_integrity/make_board.py

writes ``signal_integrity.kipi.json.gz`` and ``signal_integrity.kipi-setup.yml`` next to
this file, with a channel for each. Open it without KiCad:

    python -m kipi --snapshot examples/signal_integrity/signal_integrity.kipi.json.gz

then switch to **SI** mode: **Scan** the board, or pick a channel and **Solve**. The tests
build the same board from here (``tests/helpers/si_boards.py``).

A 60 × 30 mm 4-layer board: F.Cu signals, In1.Cu and In2.Cu solid GND, B.Cu signals,
FR4 (εr 4.2, tan δ 0.02).

* ``CLK``: U1.1 → R1 (33 Ω) → ``CLK_R`` → U2.1, a 0.36 mm microstrip over In1 (~50 Ω with
  the mask).
* ``USB_P``/``USB_N``: a 0.2 mm / 0.15 mm gap pair from J1 on F.Cu, each leg jogging out
  to its own via (at x = 30 and 31, GND stitching vias beside them) to B.Cu, on to U3. The N line has a jog,
  so it is about 0.4 mm longer than P (skew).
"""

import sys
from pathlib import Path

from kipi.core.model.board import (
    BoardSnapshot, Coating, CopperLayer, DielectricLayer, Footprint, Net, Pad, Poly, Stackup,
    Track, Via, Zone,
)

F, IN1, IN2, B = 3, 4, 5, 34
IDS = [F, IN1, IN2, B]


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def circle(cx, cy, r, n=24):
    import math
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def stackup(er=4.2, tand=0.02):
    cu_t = [0.035, 0.035, 0.035, 0.035]
    diel_t = [0.2, 1.0, 0.2]
    names = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    total = sum(cu_t) + sum(diel_t)
    z = total
    copper, diels = [], []
    for i in range(4):
        z -= cu_t[i]
        copper.append(CopperLayer(IDS[i], names[i], i, cu_t[i], z))
        if i < 3:
            z -= diel_t[i]
            diels.append(DielectricLayer(diel_t[i], "FR4", er, IDS[i], IDS[i + 1],
                                         loss_tangent=tand))
    return Stackup(copper, diels, total, coatings=[
        Coating("mask", "top", 0.015), Coating("mask", "bottom", 0.015)])


def _pad(pid, net, ref, num, x, y, layer=F, w=0.6, h=0.4):
    return Pad(pid, net, ref, num, (x, y), "smd",
               shapes={layer: [Poly(rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2))]})


def make_board(stitch=True) -> BoardSnapshot:
    tracks = [
        Track("CLK", F, 0.36, (5.0, 5.0), (20.0, 5.0)),
        Track("CLK_R", F, 0.36, (22.0, 5.0), (55.0, 5.0)),
        # The pair: P at y = 15, N at y = 15.35 (0.2 wide, 0.15 gap).
        # Each leg jogs out to its own via, clear of the other leg (and its pad).
        Track("USB_P", F, 0.2, (5.0, 15.0), (29.6, 15.0)),
        Track("USB_P", F, 0.2, (29.6, 15.0), (30.0, 14.6)),
        Track("USB_N", F, 0.2, (5.0, 15.35), (12.0, 15.35)),
        Track("USB_N", F, 0.2, (12.0, 15.35), (12.5, 15.85)),       # a 1 mm jog
        Track("USB_N", F, 0.2, (12.5, 15.85), (13.0, 15.35)),
        Track("USB_N", F, 0.2, (13.0, 15.35), (30.6, 15.35)),
        Track("USB_N", F, 0.2, (30.6, 15.35), (31.0, 15.75)),
        Track("USB_P", B, 0.2, (30.0, 14.6), (30.4, 15.0)),
        Track("USB_P", B, 0.2, (30.4, 15.0), (55.0, 15.0)),
        Track("USB_N", B, 0.2, (31.0, 15.75), (31.4, 15.35)),
        Track("USB_N", B, 0.2, (31.4, 15.35), (55.0, 15.35)),
    ]
    vias = [Via("vp", "USB_P", (30.0, 14.6), 0.25, 0.5, IDS),
            Via("vn", "USB_N", (31.0, 15.75), 0.25, 0.5, IDS)]
    if stitch:
        vias += [Via("g1", "GND", (30.0, 13.8), 0.3, 0.6, IDS),
                 Via("g2", "GND", (31.0, 16.55), 0.3, 0.6, IDS)]
    pads = [
        _pad("u1", "CLK", "U1", "1", 5.0, 5.0), _pad("r1a", "CLK", "R1", "1", 20.0, 5.0),
        _pad("r1b", "CLK_R", "R1", "2", 22.0, 5.0), _pad("u2", "CLK_R", "U2", "1", 55.0, 5.0),
        _pad("jp", "USB_P", "J1", "1", 5.0, 15.0, w=0.4, h=0.2),
        _pad("jn", "USB_N", "J1", "2", 5.0, 15.35, w=0.4, h=0.2),
        _pad("u3p", "USB_P", "U3", "1", 55.0, 15.0, layer=B, w=0.4, h=0.2),
        _pad("u3n", "USB_N", "U3", "2", 55.0, 15.35, layer=B, w=0.4, h=0.2),
    ]
    holes = [circle(30.0, 14.6, 0.45), circle(31.0, 15.75, 0.45)]
    zones = [Zone("zg1", "GND", "", 0, {IN1: [Poly(rect(0, 0, 60, 30), holes)]}),
             Zone("zg2", "GND", "", 0, {IN2: [Poly(rect(0, 0, 60, 30), holes)]})]
    fps = [Footprint(r, r, v, (0, 0), "top") for r, v in
           (("U1", "MCU"), ("U2", "FPGA"), ("R1", "33"), ("J1", "USB"), ("U3", "PHY"))]
    nets = [Net(i, n) for i, n in enumerate(["", "GND", "CLK", "CLK_R", "USB_P", "USB_N"])]
    return BoardSnapshot("signal_integrity.kicad_pcb", "", stackup(), nets, tracks, vias, pads, fps, zones,
                         [Poly(rect(0, 0, 60, 30))], source="synthetic")


def make_setup():
    from kipi.dc.pi.model import PISetup
    from kipi.si.model import new_channel
    setup = PISetup()
    setup.si_channels = [new_channel("CLK"), new_channel("USB_P", "USB_N")]
    return setup


HERE = Path(__file__).resolve().parent
NAME = "signal_integrity"


def write():
    from kipi.core.model.snapshot_io import save
    snap_path = HERE / f"{NAME}.kipi.json.gz"
    setup_path = HERE / f"{NAME}.kipi-setup.yml"
    save(make_board(), snap_path)
    make_setup().save(setup_path)
    return snap_path, setup_path


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1]))
    print("Wrote", *(p.name for p in write()))
