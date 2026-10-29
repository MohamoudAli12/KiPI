"""The Signal Integrity demo board: a KiCad project (schematic and PCB) with good and bad
versions of common interfaces, to show what KiPI's Signal Integrity mode finds.

    python examples/si_demo/make_project.py

writes, next to this file:

* ``si_demo.kicad_pro``, ``si_demo.kicad_sch``, ``si_demo.kicad_pcb``: the project. With
  ``kicad-cli`` on the PATH its zones are filled (KiPI reads the fills), and DRC, ERC and
  the schematic parity are checked; ``--no-kicad`` skips that.
* ``SI_Demo.pretty`` and ``SI_Demo.kicad_sym``, with ``fp-lib-table`` and
  ``sym-lib-table``: the parts' footprints and symbols, so the project opens on its own.
* ``si_demo.kipi-setup.yml``: KiPI's setup, a channel for each interface (when KiPI can be
  imported, as it can from the repository).

Open ``si_demo.kicad_pcb`` in KiPI (or in KiCad, then KiPI), switch to **SI** (Ctrl+4),
**Scan** the board, then pick a channel and **Solve**. See ``README.md`` for what each
interface shows.

The board: 100 × 70 mm, 6 layers, 1.6 mm FR-4 (εr 4.2, tan δ 0.02)::

    F.Cu    signal: microstrip over In1 (0.10 mm)
    In1.Cu  GND plane (with a slot under the bad USB 2.0 pair)
    In2.Cu  signal: stripline between In1 and In3 (0.20 mm each side)
    In3.Cu  GND plane
    In4.Cu  +3V3 plane
    B.Cu    (unused)

The trace sizes were found with KiPI's own field solver: F.Cu 0.15 mm is 49 Ω, 0.23 mm
40 Ω; F.Cu pairs 0.14/0.15 mm 86 Ω and 0.12/0.15 mm 91 Ω; In2 pair 0.16/0.15 mm 86 Ω.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import uuid as uuidlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
NAME = "si_demo"
LIB = "SI_Demo"
_NS = uuidlib.UUID("6f1c6a52-3d0e-4d1f-9a64-6b1c0de51d00")


def uid(*key) -> str:
    """A UUID fixed by ``key``, so the files come out the same every run."""
    return str(uuidlib.uuid5(_NS, "/".join(map(str, key))))


def q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def f(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# ============================================================================ stackup
COPPER = ["F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "B.Cu"]
LAYER_IDS = {"F.Cu": 0, "B.Cu": 2, "In1.Cu": 4, "In2.Cu": 6, "In3.Cu": 8, "In4.Cu": 10}
CU_T = {"F.Cu": 0.035, "In1.Cu": 0.0175, "In2.Cu": 0.0175, "In3.Cu": 0.0175,
        "In4.Cu": 0.0175, "B.Cu": 0.035}
# Dielectric under each copper layer but the last: (type, thickness).
DIELECTRIC = [("prepreg", 0.10), ("core", 0.20), ("prepreg", 0.20), ("core", 0.86),
              ("prepreg", 0.10)]
ER, TAND = 4.2, 0.02

# ============================================================================ sizes
W_SE = 0.15                   # 50 Ω microstrip
W_DQ = 0.23                   # 40 Ω microstrip (DDR4 DQ)
PCIE_F = (0.14, 0.15)         # 85 Ω microstrip pair (width, gap)
PCIE_IN2 = (0.16, 0.15)       # 85 Ω stripline pair
USB_F = (0.12, 0.15)          # 90 Ω microstrip pair
PCIE_BAD = (0.10, 0.30)       # ~109 Ω: too narrow, too far apart
VIA = (0.45, 0.2)             # diameter, drill
PWR_W = 0.3


# ============================================================================ footprints
@dataclass
class PadDef:
    num: str
    x: float
    y: float
    w: float
    h: float
    kind: str = "smd"             # smd | thru
    shape: str = "roundrect"      # roundrect | rect | circle
    drill: float = 0.0


@dataclass
class FpDef:
    name: str
    pads: List[PadDef]
    body: Tuple[float, float]     # fab outline (w, h), centred
    descr: str = ""
    tht: bool = False
    crt: float = 0.25             # courtyard margin round the body


def _chip(name, descr) -> FpDef:
    return FpDef(name, [PadDef("1", -0.48, 0, 0.56, 0.62), PadDef("2", 0.48, 0, 0.56, 0.62)],
                 (1.0, 0.5), descr)


# SoC: a 10 × 10 mm leadless package, its pins where the routing wants them.
#   right edge: PCIe TX0, TX1, USB 3 TX, USB 2 D (0.5 mm pitch within a pair)
#   top edge: DQ0–3 at 1.0 mm pitch, DQ4–7 at 0.35 mm pitch (too tight, on purpose)
#   bottom edge: CLK_GOOD, CLK_BAD; left edge: +3V3 and GND; a GND exposed pad.
U1_AT = (125.0, 135.0)
U1_PINS = [  # (number, name, net, side, absolute coordinate along that side)
    ("1", "PCIE_TX0_P", "PCIE_TX0_P", "R", 131.0),
    ("2", "PCIE_TX0_N", "PCIE_TX0_N", "R", 131.5),
    ("3", "PCIE_TX1_P", "PCIE_TX1_P", "R", 133.5),
    ("4", "PCIE_TX1_N", "PCIE_TX1_N", "R", 134.0),
    ("5", "USB3_TX_P", "USB3_TX_P", "R", 136.5),
    ("6", "USB3_TX_N", "USB3_TX_N", "R", 137.0),
    ("7", "USB_D_P", "USB_D_P", "R", 139.0),
    ("8", "USB_D_N", "USB_D_N", "R", 139.5),
    ("9", "DQ0", "DQ0", "T", 120.5),
    ("10", "DQ1", "DQ1", "T", 121.5),
    ("11", "DQ2", "DQ2", "T", 122.5),
    ("12", "DQ3", "DQ3", "T", 123.5),
    ("13", "DQ4", "DQ4", "T", 125.5),
    ("14", "DQ5", "DQ5", "T", 125.85),
    ("15", "DQ6", "DQ6", "T", 126.2),
    ("16", "DQ7", "DQ7", "T", 126.55),
    ("17", "CLK_GOOD", "CLK_GOOD", "B", 123.0),
    ("18", "CLK_BAD", "CLK_BAD", "B", 127.0),
    ("19", "VDD", "+3V3", "L", 131.0),
    ("20", "VSS", "GND", "L", 132.0),
    ("21", "VDD", "+3V3", "L", 133.5),
    ("22", "VSS", "GND", "L", 134.5),
    ("23", "VDD", "+3V3", "L", 136.0),
    ("24", "VSS", "GND", "L", 137.0),
    ("25", "VSS", "GND", "EP", 0.0),
]


def _edge_pads(at, pins, half=5.0, pad_len=0.6, pad_w=0.25, top_w=0.22):
    """Pads at the package's edges, from absolute coordinates along each side."""
    cx, cy = at
    out = []
    for num, _n, _net, side, pos in pins:
        if side == "R":
            out.append(PadDef(num, half, pos - cy, pad_len, pad_w))
        elif side == "L":
            out.append(PadDef(num, -half, pos - cy, pad_len, pad_w))
        elif side == "T":
            out.append(PadDef(num, pos - cx, -half, top_w, pad_len))
        elif side == "B":
            out.append(PadDef(num, pos - cx, half, pad_w, pad_len))
        elif side == "EP":
            out.append(PadDef(num, 0, 0, 4.0, 4.0, shape="rect"))
    return out


# DRAM: the DQ pads on its bottom edge under U1's, power on its top edge.
U2_AT = (123.5, 107.0)
U2_PINS = [(str(i + 1), f"DQ{i}", f"DQ{i}", "B", x) for i, x in
           enumerate([120.5, 121.5, 122.5, 123.5, 125.5, 125.85, 126.2, 126.55])] + [
    ("9", "VDD", "+3V3", "T", 121.0), ("10", "VSS", "GND", "T", 126.0)]

# Clock receiver: two CMOS inputs on its left edge.
U3_AT = (172.0, 160.0)
U3_PINS = [("1", "IN_A", "CLK_GOOD_R", "L", 161.0), ("2", "IN_B", "CLK_BAD", "L", 159.0),
           ("3", "VDD", "+3V3", "R", 159.0), ("4", "VSS", "GND", "R", 161.0)]

# Connectors: a column of pads (pin, net, y) at x, 1.6 × 0.6 mm.
# Connectors: a column of 1.6 × 0.3 mm pads (pin, net, y) at x, a pair's at 0.5 mm pitch.
J1_AT = (189.0, 132.625)
J1_PINS = [("1", "GND", 130.25), ("2", "PCIE_TX0_C_P", 131.0), ("3", "PCIE_TX0_C_N", 131.5),
           ("4", "GND", 132.5), ("5", "PCIE_TX1_C_P", 133.5), ("6", "PCIE_TX1_C_N", 134.0),
           ("7", "GND", 135.0)]
J2_AT = (189.0, 152.0)
J2_PINS = [("1", "GND", 148.75), ("2", "USB3_TX_C_P", 149.5), ("3", "USB3_TX_C_N", 150.0),
           ("4", "GND", 150.75), ("5", "GND", 153.25), ("6", "USB_D_P", 154.0),
           ("7", "USB_D_N", 154.5), ("8", "GND", 155.25)]


def _column(at, pins):
    return [PadDef(n, 0, y - at[1], 1.6, 0.3) for n, _net, y in pins]


FOOTPRINTS: Dict[str, FpDef] = {
    # 0201s stack at the pairs' 0.5 mm pitch, so a pair needs no fan-out to its caps.
    "C_0201": FpDef("C_0201", [PadDef("1", -0.3, 0, 0.3, 0.3), PadDef("2", 0.3, 0, 0.3, 0.3)],
                    (0.9, 0.3), "Capacitor 0201 (0603 Metric)", crt=0.08),
    "R_0402": _chip("R_0402", "Resistor 0402 (1005 Metric)"),
    "C_0402": _chip("C_0402", "Capacitor 0402 (1005 Metric)"),
    "SoC_QFN": FpDef("SoC_QFN", _edge_pads(U1_AT, U1_PINS), (10.0, 10.0),
                     "Demo SoC, 10 x 10 mm leadless"),
    "DRAM": FpDef("DRAM", _edge_pads(U2_AT, U2_PINS, half=2.5, top_w=0.22), (8.0, 5.0),
                  "Demo DRAM, DQ pads on its bottom edge"),
    "ClockRx": FpDef("ClockRx", _edge_pads(U3_AT, U3_PINS, half=2.0), (4.0, 4.0),
                     "Demo CMOS clock receiver"),
    "Conn_PCIe": FpDef("Conn_PCIe", _column(J1_AT, J1_PINS), (2.4, 5.6),
                       "Demo PCIe x2 TX connector, 1.6 x 0.3 mm pads"),
    "Conn_USB": FpDef("Conn_USB", _column(J2_AT, J2_PINS), (2.4, 7.4),
                      "Demo USB connector, 1.6 x 0.3 mm pads"),
    "PinHeader_1x02": FpDef("PinHeader_1x02", [
        PadDef("1", 0, 0, 1.7, 1.7, "thru", "rect", 1.0),
        PadDef("2", 0, 2.54, 1.7, 1.7, "thru", "circle", 1.0)], (2.54, 5.08),
        "Power input, 1x02 2.54 mm", tht=True),
    "TestPad": FpDef("TestPad", [PadDef("1", 0, 0, 1.0, 1.0, shape="circle")], (1.2, 1.2),
                     "SMD test pad, 1 mm"),
}


# ============================================================================ parts
@dataclass
class Part:
    ref: str
    value: str
    symbol: str
    fp: str
    at: Tuple[float, float]
    rot: float = 0.0
    nets: Dict[str, str] = field(default_factory=dict)        # pad -> net
    dnp: bool = False
    note: str = ""


PARTS: List[Part] = []


def part(ref, value, symbol, fp, at, rot=0.0, nets=None, dnp=False, note=""):
    p = Part(ref, value, symbol, fp, at, rot, dict(nets or {}), dnp, note)
    PARTS.append(p)
    return p


def _pins_nets(pins):
    return {num: net for num, _n, net, _s, _p in pins}


part("U1", "SoC", "SoC", "SoC_QFN", U1_AT, nets=_pins_nets(U1_PINS))
part("U2", "DRAM", "DRAM", "DRAM", U2_AT, nets=_pins_nets(U2_PINS))
part("U3", "ClockRx", "ClockRx", "ClockRx", U3_AT, nets=_pins_nets(U3_PINS))
part("J1", "PCIe_TX", "Conn_PCIe", "Conn_PCIe", J1_AT,
     nets={n: net for n, net, _y in J1_PINS})
part("J2", "USB", "Conn_USB", "Conn_USB", J2_AT, nets={n: net for n, net, _y in J2_PINS})
part("J3", "PWR_IN", "Conn_Power", "PinHeader_1x02", (104.5, 133.73),
     nets={"1": "+3V3", "2": "GND"})
# AC coupling, P above N: the good lanes' 0201s at the pair's pitch; the bad lane's 0402s
# 1.3 mm apart, so its pair splits to reach them.
part("C1", "100n", "C", "C_0201", (135.0, 131.0), nets={"1": "PCIE_TX0_P", "2": "PCIE_TX0_C_P"})
part("C2", "100n", "C", "C_0201", (135.0, 131.5), nets={"1": "PCIE_TX0_N", "2": "PCIE_TX0_C_N"})
part("C3", "100n", "C", "C_0402", (135.0, 133.1), nets={"1": "PCIE_TX1_P", "2": "PCIE_TX1_C_P"})
part("C4", "100n", "C", "C_0402", (135.0, 134.4), nets={"1": "PCIE_TX1_N", "2": "PCIE_TX1_C_N"})
part("C5", "100n", "C", "C_0201", (135.0, 136.5), nets={"1": "USB3_TX_P", "2": "USB3_TX_C_P"})
part("C6", "100n", "C", "C_0201", (135.0, 137.0), nets={"1": "USB3_TX_N", "2": "USB3_TX_C_N"})
# The good clock's source termination; the bad clock's optional test-point link, not fitted.
part("R1", "33", "R", "R_0402", (123.0, 143.0), 90,
     nets={"1": "CLK_GOOD_R", "2": "CLK_GOOD"})
part("R5", "0", "R", "R_0402", (129.0, 150.0), nets={"1": "CLK_BAD", "2": "CLK_TP"}, dnp=True,
     note="DNP: a test-point link; its empty pads stay on CLK_BAD as a stub")
part("TP1", "CLK_TP", "TestPoint", "TestPad", (131.5, 150.0), nets={"1": "CLK_TP"})
# Decoupling: +3V3 to GND, pads vertical (rot 90: pad 1 below, pad 2 above).
for k, (x, y, val) in enumerate([(117.5, 128.5, "100n"), (117.5, 133.75, "100n"),
                                 (117.5, 139.0, "100n"), (108.5, 135.0, "10u"),
                                 (123.5, 103.0, "100n"), (175.5, 156.0, "100n")]):
    part(f"C{10 + k}", val, "C", "C_0402", (x, y), 90, nets={"1": "GND", "2": "+3V3"})


def pad_xy(ref: str, num: str) -> Tuple[float, float]:
    """A pad's centre on the board."""
    p = next(x for x in PARTS if x.ref == ref)
    pd = next(x for x in FOOTPRINTS[p.fp].pads if x.num == num)
    a = math.radians(p.rot)
    # KiCad: rotation counter-clockwise on screen (y down).
    x = pd.x * math.cos(a) + pd.y * math.sin(a)
    y = -pd.x * math.sin(a) + pd.y * math.cos(a)
    return (round(p.at[0] + x, 6), round(p.at[1] + y, 6))


# ============================================================================ routing
@dataclass
class Track:
    pts: List[Tuple[float, float]]
    width: float
    layer: str
    net: str


@dataclass
class ViaDef:
    at: Tuple[float, float]
    net: str
    kind: str = "through"          # through | blind
    layers: Tuple[str, str] = ("F.Cu", "B.Cu")


TRACKS: List[Track] = []
VIAS: List[ViaDef] = []


def track(net, layer, width, *pts):
    TRACKS.append(Track([tuple(map(float, p)) for p in pts], width, layer, net))


def via(net, x, y, kind="through", layers=("F.Cu", "B.Cu")):
    VIAS.append(ViaDef((x, y), net, kind, layers))


def h45(a, b):
    """From ``a`` horizontally, then at 45° to ``b`` (straight on when it cannot)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    run = abs(dx) - abs(dy)
    if run <= 1e-9 or abs(dy) < 1e-9:
        return [a, b]
    return [a, (a[0] + math.copysign(run, dx), a[1]), b]


def d45(a, b):
    """From ``a`` at 45° first, then horizontally to ``b``: a pad's or via's jog to the
    pair's pitch stays on the pad."""
    return list(reversed(h45(b, a)))


def v45(a, b):
    """From ``a`` vertically, then at 45° to ``b``."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    run = abs(dy) - abs(dx)
    if run <= 1e-9 or abs(dx) < 1e-9:
        return [a, b]
    return [a, (a[0], a[1] + math.copysign(run, dy)), b]


def offset(pts, d):
    """``pts`` offset by ``d`` to its left as drawn on screen (y down: to +y for a line
    going +x), corners mitred."""
    lines = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        nx, ny = -uy, ux
        lines.append(((x0 + nx * d, y0 + ny * d), (ux, uy)))
    out = [lines[0][0]]
    for (p0, u0), (p1, u1) in zip(lines, lines[1:]):
        den = u0[0] * u1[1] - u0[1] * u1[0]
        t = ((p1[0] - p0[0]) * u1[1] - (p1[1] - p0[1]) * u1[0]) / den
        out.append((p0[0] + u0[0] * t, p0[1] + u0[1] * t))
    (xe, ye) = pts[-1]
    (x0, y0), (ux, uy) = lines[-1]
    out.append((xe + (-uy) * d, ye + ux * d))
    return [(round(x, 6), round(y, 6)) for x, y in out]


def join(*chains):
    """Chains of points end to end, a point shared where one ends and the next starts."""
    out = []
    for c in chains:
        for p in c:
            p = (round(p[0], 6), round(p[1], 6))
            if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
                out.append(p)
    return out


def pair(np_, nn, layer, size, centre, start=None, end=None):
    """A pair along centreline ``centre``, P on its upper side going +x; ``start`` and
    ``end``: (P point, N point) the legs come from and go to, the jog to the pair's pitch
    right at them."""
    w, gap = size
    d = (w + gap) / 2
    lp, ln = offset(centre, -d), offset(centre, d)
    for net, leg, i in ((np_, lp, 0), (nn, ln, 1)):
        chain = leg
        if start is not None:
            chain = join(d45(start[i], leg[0]), chain)
        if end is not None:
            chain = join(chain, list(reversed(d45(end[i], leg[-1]))))
        track(net, layer, w, *chain)


# ---------------------------------------------------------------- PCIe TX0: good
# F.Cu 85 Ω out of U1 → AC caps → blind vias (F.Cu–In2.Cu, no stub) with a GND via each
# side → In2 stripline 85 Ω → blind vias with GND vias → F.Cu → J1.
yc = 131.25
pair("PCIE_TX0_P", "PCIE_TX0_N", "F.Cu", PCIE_F, [(131.2, yc), (133.0, yc)],
     start=(pad_xy("U1", "1"), pad_xy("U1", "2")), end=(pad_xy("C1", "1"), pad_xy("C2", "1")))
v0 = ((139.5, yc - 0.3), (139.5, yc + 0.3))
pair("PCIE_TX0_C_P", "PCIE_TX0_C_N", "F.Cu", PCIE_F, [(136.6, yc), (139.35, yc)],
     start=(pad_xy("C1", "2"), pad_xy("C2", "2")), end=v0)
v1 = ((184.0, yc - 0.3), (184.0, yc + 0.3))
pair("PCIE_TX0_C_P", "PCIE_TX0_C_N", "In2.Cu", PCIE_IN2, [(139.65, yc), (183.85, yc)],
     start=v0, end=v1)
pair("PCIE_TX0_C_P", "PCIE_TX0_C_N", "F.Cu", PCIE_F, [(184.15, yc), (187.0, yc)],
     start=v1, end=(pad_xy("J1", "2"), pad_xy("J1", "3")))
for (x, y), net in zip(v0 + v1, ["PCIE_TX0_C_P", "PCIE_TX0_C_N"] * 2):
    via(net, x, y, "blind", ("F.Cu", "In2.Cu"))
for x in (139.5, 184.0):
    via("GND", x, yc - 1.0)
    via("GND", x, yc + 1.0)

# ---------------------------------------------------------------- PCIe TX1: bad
# 109 Ω (too narrow, too far apart); P detours 2.4 mm up and back, so it is ~2 mm longer
# than N (skew); through vias, so In2.Cu–B.Cu is a stub; no GND via near them.
yc = 133.75
pair("PCIE_TX1_P", "PCIE_TX1_N", "F.Cu", PCIE_BAD, [(131.2, yc), (133.0, yc)],
     start=(pad_xy("U1", "3"), pad_xy("U1", "4")), end=(pad_xy("C3", "1"), pad_xy("C4", "1")))
dP, dN = PCIE_BAD[0] / 2 + PCIE_BAD[1] / 2, PCIE_BAD[0] / 2 + PCIE_BAD[1] / 2
b0 = ((150.0, 133.35), (150.0, 134.15))
track("PCIE_TX1_C_P", "F.Cu", PCIE_BAD[0], *join(
    h45(pad_xy("C3", "2"), (136.6, yc - dP)), [(142.0, yc - dP), (144.2, yc - dP - 2.2),
                                                (146.2, yc - dP - 2.2), (148.4, yc - dP)],
    list(reversed(h45(b0[0], (148.4, yc - dP))))))
track("PCIE_TX1_C_N", "F.Cu", PCIE_BAD[0], *join(
    h45(pad_xy("C4", "2"), (136.6, yc + dN)), list(reversed(h45(b0[1], (148.4, yc + dN))))))
b1 = ((178.0, 133.35), (178.0, 134.15))
pair("PCIE_TX1_C_P", "PCIE_TX1_C_N", "In2.Cu", PCIE_BAD, [(151.0, yc), (177.0, yc)],
     start=b0, end=b1)
pair("PCIE_TX1_C_P", "PCIE_TX1_C_N", "F.Cu", PCIE_BAD, [(179.0, yc), (186.0, yc)],
     start=b1, end=(pad_xy("J1", "5"), pad_xy("J1", "6")))
for (x, y), net in zip(b0 + b1, ["PCIE_TX1_C_P", "PCIE_TX1_C_N"] * 2):
    via(net, x, y)

# ---------------------------------------------------------------- USB 3.2 TX: good
# F.Cu 90 Ω, AC caps, over solid In1 GND, to J2.
yc = 136.75
pair("USB3_TX_P", "USB3_TX_N", "F.Cu", USB_F, [(131.2, yc), (133.0, yc)],
     start=(pad_xy("U1", "5"), pad_xy("U1", "6")), end=(pad_xy("C5", "1"), pad_xy("C6", "1")))
pair("USB3_TX_C_P", "USB3_TX_C_N", "F.Cu", USB_F,
     [(136.6, yc), (152.0, yc), (165.0, yc + 13.0), (186.0, yc + 13.0)],
     start=(pad_xy("C5", "2"), pad_xy("C6", "2")), end=(pad_xy("J2", "2"), pad_xy("J2", "3")))

# ---------------------------------------------------------------- USB 2.0: bad
# Good geometry, but it crosses a 1 mm slot in the In1 GND plane: no reference there.
yc = 139.25
pair("USB_D_P", "USB_D_N", "F.Cu", USB_F,
     [(131.2, yc), (145.0, yc), (160.0, yc + 15.0), (186.0, yc + 15.0)],
     start=(pad_xy("U1", "7"), pad_xy("U1", "8")), end=(pad_xy("J2", "6"), pad_xy("J2", "7")))
SLOT = (164.5, 151.0, 165.5, 158.0)       # In1.Cu: no copper

# ---------------------------------------------------------------- DDR DQ
# DQ0–3 3W apart (good), DQ4–7 0.12 mm apart (bad): straight up from U1 to U2, 40 Ω.
for i in range(8):
    a, b = pad_xy("U1", str(9 + i)), pad_xy("U2", str(i + 1))
    track(f"DQ{i}", "F.Cu", W_DQ, a, b)

# ---------------------------------------------------------------- clocks
# CLK_GOOD: 33 Ω at the source, then ~65 mm to U3. CLK_BAD: no termination, the same
# length, and a stub to the not-fitted R5 (test point).
track("CLK_GOOD", "F.Cu", W_SE, pad_xy("U1", "17"), pad_xy("R1", "2"))
track("CLK_GOOD_R", "F.Cu", W_SE, pad_xy("R1", "1"), (123.0, 159.5), (124.5, 161.0),
      pad_xy("U3", "1"))
track("CLK_BAD", "F.Cu", W_SE, pad_xy("U1", "18"), (127.0, 157.5), (128.5, 159.0),
      pad_xy("U3", "2"))
track("CLK_BAD", "F.Cu", W_SE, (127.0, 150.0), pad_xy("R5", "1"))
track("CLK_TP", "F.Cu", W_SE, pad_xy("R5", "2"), pad_xy("TP1", "1"))

# ---------------------------------------------------------------- power
# Each power or ground pad of a part gets a via to its plane beside it.


def tie(ref, num, dx, dy, net, width=PWR_W):
    x, y = pad_xy(ref, num)
    via(net, x + dx, y + dy)
    track(net, "F.Cu", width, (x, y), (x + dx, y + dy))


for num, _n, net, side, _p in U1_PINS:
    if side == "L":
        tie("U1", num, -1.2, 0.0, net)
x0, y0 = U1_AT
for dx in (-1.2, 0.0, 1.2):
    for dy in (-1.2, 0.0, 1.2):
        via("GND", x0 + dx, y0 + dy)                       # under the exposed pad
for num, net, dx, dy in (("9", "+3V3", 0, -1.0), ("10", "GND", 0, -1.0)):
    tie("U2", num, dx, dy, net)
tie("U3", "3", 1.0, 0, "+3V3")
tie("U3", "4", 1.0, 0, "GND")
for ref in ("J1", "J2"):
    pins = J1_PINS if ref == "J1" else J2_PINS
    for n, net, _y in pins:
        if net == "GND":
            x, y = pad_xy(ref, n)
            via("GND", x + 0.6, y)
for k in range(6):
    ref = f"C{10 + k}"
    tie(ref, "1", 0, 0.7, "GND")
    tie(ref, "2", 0, -0.7, "+3V3")
# J3's pins are through-hole: they join the planes themselves.

BOARD = (100.0, 100.0, 200.0, 170.0)


def _seg_dist(p, a, b):
    ax, ay = a
    bx, by = b
    L2 = (bx - ax) ** 2 + (by - ay) ** 2
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / L2))
    return math.hypot(p[0] - (ax + t * (bx - ax)), p[1] - (ay + t * (by - ay)))


def stitch(step=4.0):
    """GND vias on a grid between the planes, where they hit nothing, none within 5 mm of
    the bad PCIe lane's vias (so they have no return via) or in the In1 slot."""
    bodies = []
    for p in PARTS:
        fd = FOOTPRINTS[p.fp]
        w, h = fd.body
        if p.rot in (90, 270):
            w, h = h, w
        bodies.append((p.at[0] - w / 2 - 0.8, p.at[1] - h / 2 - 0.8,
                       p.at[0] + w / 2 + 0.8, p.at[1] + h / 2 + 0.8))
    x0, y0, x1, y1 = BOARD
    y = y0 + 2.0
    while y < y1 - 1.0:
        x = x0 + 2.0
        while x < x1 - 1.0:
            p = (x, y)
            ok = all(not (b[0] <= x <= b[2] and b[1] <= y <= b[3]) for b in bodies)
            ok = ok and all(math.hypot(x - v.at[0], y - v.at[1]) > 1.0 for v in VIAS)
            ok = ok and all(math.hypot(x - bx, y - by) > 5.0 for bx, by in b0 + b1)
            ok = ok and not (SLOT[0] - 1 <= x <= SLOT[2] + 1 and SLOT[1] - 1 <= y <= SLOT[3] + 1)
            ok = ok and all(_seg_dist(p, a, b) > t.width / 2 + 0.4
                            for t in TRACKS for a, b in zip(t.pts, t.pts[1:]))
            if ok:
                via("GND", round(x, 3), round(y, 3))
            x += step
        y += step


stitch()


# ============================================================================ nets
def all_nets() -> List[str]:
    names = set()
    for p in PARTS:
        names |= set(p.nets.values())
    names |= {t.net for t in TRACKS} | {v.net for v in VIAS} | {"GND", "+3V3"}
    return sorted(names, key=lambda n: (n not in ("GND", "+3V3"), n))


NETS = all_nets()
NET_ID = {n: i + 1 for i, n in enumerate(NETS)}


# ============================================================================ writers
def font(size=1.0, thick=0.15, hide=False):
    return (f"(effects (font (size {f(size)} {f(size)}) (thickness {f(thick)}))"
            + (" (hide yes)" if hide else "") + ")")


def fp_body(fd: FpDef, pad_nets: Optional[Dict[str, str]] = None, rot: float = 0.0,
            board: bool = False) -> List[str]:
    """A footprint's graphics and pads (its pads' nets when on the board)."""
    w, h = fd.body
    out = [f'(fp_rect (start {f(-w / 2)} {f(-h / 2)}) (end {f(w / 2)} {f(h / 2)}) '
           f'(stroke (width 0.1) (type solid)) (fill no) (layer "F.Fab") (uuid {q(uid(fd.name, "fab", board))}))',
           f'(fp_rect (start {f(-w / 2 - fd.crt)} {f(-h / 2 - fd.crt)}) (end {f(w / 2 + fd.crt)} {f(h / 2 + fd.crt)}) '
           f'(stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd") (uuid {q(uid(fd.name, "crt", board))}))']
    for pd in fd.pads:
        if pd.kind == "thru":
            layers = '"*.Cu" "*.Mask"'
            head = f'(pad {q(pd.num)} thru_hole {pd.shape}'
            extra = f" (drill {f(pd.drill)})"
        else:
            layers = '"F.Cu" "F.Mask" "F.Paste"'
            head = f'(pad {q(pd.num)} smd {pd.shape}'
            extra = " (roundrect_rratio 0.25)" if pd.shape == "roundrect" else ""
        net = ""
        if pad_nets is not None and pad_nets.get(pd.num):
            n = pad_nets[pd.num]
            net = f" (net {NET_ID[n]} {q(n)})"
        out.append(f"{head} (at {f(pd.x)} {f(pd.y)}{(' ' + f(rot)) if rot else ''}) "
                   f"(size {f(pd.w)} {f(pd.h)}){extra} (layers {layers}){net} "
                   f"(uuid {q(uid(fd.name, pd.num, board, id(pad_nets)))}))")
    return out


def write_footprints(lib: Path):
    lib.mkdir(exist_ok=True)
    for fd in FOOTPRINTS.values():
        body = "\n\t".join(fp_body(fd))
        attr = "(attr through_hole)" if fd.tht else "(attr smd)"
        text = (f'(footprint {q(fd.name)}\n\t(version 20241229)\n\t(generator "kipi_si_demo")\n'
                f'\t(layer "F.Cu")\n\t(descr {q(fd.descr)})\n'
                f'\t(property "Reference" "REF**" (at 0 {f(-fd.body[1] / 2 - 1)} 0) (layer "F.SilkS") {font()})\n'
                f'\t(property "Value" {q(fd.name)} (at 0 {f(fd.body[1] / 2 + 1)} 0) (layer "F.Fab") {font()})\n'
                f'\t{attr}\n\t{body}\n)\n')
        (lib / f"{fd.name}.kicad_mod").write_text(text)


def stackup_text() -> str:
    lines = ['(layer "F.SilkS" (type "Top Silk Screen"))',
             '(layer "F.Paste" (type "Top Solder Paste"))',
             '(layer "F.Mask" (type "Top Solder Mask") (thickness 0.015) (epsilon_r 3.8) (loss_tangent 0.02))']
    for i, cu in enumerate(COPPER):
        lines.append(f'(layer {q(cu)} (type "copper") (thickness {f(CU_T[cu])}))')
        if i < len(DIELECTRIC):
            kind, t = DIELECTRIC[i]
            lines.append(f'(layer "dielectric {i + 1}" (type {q(kind)}) (thickness {f(t)}) '
                         f'(material "FR4") (epsilon_r {f(ER)}) (loss_tangent {f(TAND)}))')
    lines += ['(layer "B.Mask" (type "Bottom Solder Mask") (thickness 0.015) (epsilon_r 3.8) (loss_tangent 0.02))',
              '(layer "B.Paste" (type "Bottom Solder Paste"))',
              '(layer "B.SilkS" (type "Bottom Silk Screen"))',
              '(copper_finish "ENIG") (dielectric_constraints no)']
    return "(stackup\n\t\t\t" + "\n\t\t\t".join(lines) + ")"


def zone(net, layer, poly, prio, name, keepout=False):
    pts = " ".join(f"(xy {f(x)} {f(y)})" for x, y in poly)
    if keepout:
        return (f'(zone (net 0) (net_name "") (layer {q(layer)}) (uuid {q(uid("keepout", name))}) '
                f'(name {q(name)}) (hatch edge 0.5) (connect_pads (clearance 0)) (min_thickness 0.25) '
                f'(filled_areas_thickness no) (keepout (tracks allowed) (vias allowed) (pads allowed) '
                f'(copperpour not_allowed) (footprints allowed)) (placement (enabled no) (sheetname "")) '
                f'(fill (thermal_gap 0.5) (thermal_bridge_width 0.5)) (polygon (pts {pts})))')
    return (f'(zone (net {NET_ID[net]}) (net_name {q(net)}) (layer {q(layer)}) '
            f'(uuid {q(uid("zone", name))}) (name {q(name)}) (hatch edge 0.5) (priority {prio}) '
            f'(connect_pads (clearance 0.2)) (min_thickness 0.2) (filled_areas_thickness no) '
            f'(fill yes (thermal_gap 0.3) (thermal_bridge_width 0.3)) (polygon (pts {pts})))')


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def write_pcb(path: Path, sym_uuid: Dict[str, str], root: str):
    out = ["(kicad_pcb", "\t(version 20241229)", '\t(generator "kipi_si_demo")',
           '\t(generator_version "9.0")', "\t(general (thickness 1.6) (legacy_teardrops no))",
           '\t(paper "A4")', "\t(layers"]
    for cu in COPPER:
        kind = "power" if cu in ("In1.Cu", "In3.Cu", "In4.Cu") else "signal"
        out.append(f"\t\t({LAYER_IDS[cu]} {q(cu)} {kind})")
    for lid, name, kind in ((9, "F.Adhes", "user \"F.Adhesive\""), (11, "B.Adhes", "user \"B.Adhesive\""),
                            (13, "F.Paste", "user"), (15, "B.Paste", "user"),
                            (5, "F.SilkS", "user \"F.Silkscreen\""), (7, "B.SilkS", "user \"B.Silkscreen\""),
                            (1, "F.Mask", "user"), (3, "B.Mask", "user"),
                            (17, "Dwgs.User", "user \"User.Drawings\""), (19, "Cmts.User", "user \"User.Comments\""),
                            (25, "Edge.Cuts", "user"), (27, "Margin", "user"),
                            (31, "F.CrtYd", "user \"F.Courtyard\""), (29, "B.CrtYd", "user \"B.Courtyard\""),
                            (35, "F.Fab", "user"), (33, "B.Fab", "user")):
        out.append(f"\t\t({lid} {q(name)} {kind})")
    out.append("\t)")
    out.append(f"\t(setup\n\t\t{stackup_text()}\n\t\t(pad_to_mask_clearance 0)\n"
               "\t\t(allow_soldermask_bridges_in_footprints no)\n"
               "\t\t(tenting front back)\n\t\t(pcbplotparams (layerselection 0x00000000_00000000_55555555_5755f5ff)"
               " (plot_on_all_layers_selection 0x00000000_00000000_00000000_00000000) (outputformat 1)"
               " (outputdirectory \"\"))\n\t)")
    out.append('\t(net 0 "")')
    for n in NETS:
        out.append(f"\t(net {NET_ID[n]} {q(n)})")
    for p in PARTS:
        fd = FOOTPRINTS[p.fp]
        attrs = "through_hole" if fd.tht else "smd"
        if p.dnp:
            attrs += " dnp"
        body = "\n\t\t".join(fp_body(fd, p.nets, p.rot, board=p.ref))
        props = (f'\t\t(property "Reference" {q(p.ref)} (at 0 {f(-fd.body[1] / 2 - 0.8)} {f(p.rot)}) '
                 f'(layer "F.SilkS") (uuid {q(uid(p.ref, "refp"))}) {font(0.6, 0.1)})\n'
                 f'\t\t(property "Value" {q(p.value)} (at 0 {f(fd.body[1] / 2 + 0.8)} {f(p.rot)}) '
                 f'(layer "F.Fab") (uuid {q(uid(p.ref, "valp"))}) {font(0.6, 0.1)})\n'
                 f'\t\t(property "Footprint" {q(LIB + ":" + p.fp)} (at 0 0 {f(p.rot)}) (layer "F.Fab") (hide yes) '
                 f'(uuid {q(uid(p.ref, "fpp"))}) {font(1, 0.15)})\n'
                 f'\t\t(property "Datasheet" "" (at 0 0 {f(p.rot)}) (layer "F.Fab") (hide yes) '
                 f'(uuid {q(uid(p.ref, "dsp"))}) {font(1, 0.15)})\n'
                 f'\t\t(property "Description" {q(p.note)} (at 0 0 {f(p.rot)}) (layer "F.Fab") (hide yes) '
                 f'(uuid {q(uid(p.ref, "dep"))}) {font(1, 0.15)})\n')
        out.append(f'\t(footprint {q(LIB + ":" + p.fp)}\n\t\t(layer "F.Cu")\n'
                   f'\t\t(uuid {q(uid("fp", p.ref))})\n\t\t(at {f(p.at[0])} {f(p.at[1])}{(" " + f(p.rot)) if p.rot else ""})\n'
                   f'{props}\t\t(path {q("/" + sym_uuid[p.ref])})\n'
                   f'\t\t(sheetname "/")\n\t\t(sheetfile {q(NAME + ".kicad_sch")})\n'
                   f'\t\t(attr {attrs})\n\t\t{body}\n\t)')
    x0, y0, x1, y1 = BOARD
    out.append(f'\t(gr_rect (start {f(x0)} {f(y0)}) (end {f(x1)} {f(y1)}) (stroke (width 0.1) (type solid)) '
               f'(fill no) (layer "Edge.Cuts") (uuid {q(uid("edge"))}))')
    for i, t in enumerate(TRACKS):
        for k, (a, b) in enumerate(zip(t.pts, t.pts[1:])):
            out.append(f"\t(segment (start {f(a[0])} {f(a[1])}) (end {f(b[0])} {f(b[1])}) "
                       f"(width {f(t.width)}) (layer {q(t.layer)}) (net {NET_ID[t.net]}) "
                       f"(uuid {q(uid('seg', i, k))}))")
    for i, v in enumerate(VIAS):
        kind = " blind" if v.kind == "blind" else ""
        out.append(f"\t(via{kind} (at {f(v.at[0])} {f(v.at[1])}) (size {f(VIA[0])}) (drill {f(VIA[1])}) "
                   f"(layers {q(v.layers[0])} {q(v.layers[1])}) (net {NET_ID[v.net]}) (uuid {q(uid('via', i))}))")
    inner = rect(x0 + 0.3, y0 + 0.3, x1 - 0.3, y1 - 0.3)
    out.append("\t" + zone("GND", "In1.Cu", inner, 0, "GND_In1"))
    out.append("\t" + zone("GND", "In3.Cu", inner, 0, "GND_In3"))
    out.append("\t" + zone("+3V3", "In4.Cu", inner, 0, "3V3_In4"))
    out.append("\t" + zone("", "In1.Cu", rect(*SLOT), 0, "In1 slot under USB_D", keepout=True))
    out.append(")")
    path.write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------------------- symbols
GRID = 2.54


@dataclass
class SymDef:
    name: str
    ref: str
    pins: List[Tuple[str, str, str]]      # (number, name, side L/R/T/B)
    w: float = 10.16
    h: float = 10.16
    descr: str = ""


def _sym_from_pins(name, ref, pins, descr):
    """A box symbol, its pins on the sides the footprint has them (the exposed pad's on
    the left)."""
    return SymDef(name, ref, [(n, nm, "L" if s == "EP" else s) for n, nm, s in pins],
                  descr=descr)


def _sides(sd: SymDef):
    side = {"L": [], "R": [], "T": [], "B": []}
    for num, nm, s in sd.pins:
        side[s].append((num, nm))
    return side


def _box(sd: SymDef):
    """The body's half width and height: room for the pins on a 2.54 mm grid, at least
    ``sd.w`` × ``sd.h``."""
    side = _sides(sd)
    n_v = max(len(side["L"]), len(side["R"]))
    n_h = max(len(side["T"]), len(side["B"]))
    hh = GRID * ((n_v - 1) // 2 + 1) if n_v else 0.0
    hw = GRID * ((n_h - 1) // 2 + 1) if n_h else 0.0
    # The vertical sides' pins need the box to be on the grid too.
    hw = max(hw, sd.w / 2) if not n_v else max(hw, GRID * math.ceil(sd.w / 2 / GRID))
    hh = max(hh, sd.h / 2) if not n_h else max(hh, GRID * math.ceil(sd.h / 2 / GRID))
    return hw, hh


def _pin_points(sd: SymDef):
    """Each pin's connection point (symbol coordinates, y up) and angle, all on the grid."""
    side = _sides(sd)
    hw, hh = _box(sd)
    out = {}
    for s, lst in side.items():
        c = (len(lst) - 1) // 2
        for k, (num, nm) in enumerate(lst):
            if s in ("L", "R"):
                x = -(hw + GRID) if s == "L" else hw + GRID
                out[num] = (x, GRID * (c - k), 0 if s == "L" else 180, nm, s)
            else:
                y = hh + GRID if s == "T" else -(hh + GRID)
                out[num] = (GRID * (k - c), y, 270 if s == "T" else 90, nm, s)
    return out


SYMBOLS: Dict[str, SymDef] = {
    "R": SymDef("R", "R", [("1", "~", "T"), ("2", "~", "B")], w=2.032, h=5.08, descr="Resistor"),
    "C": SymDef("C", "C", [("1", "~", "T"), ("2", "~", "B")], w=2.032, h=5.08, descr="Capacitor"),
    "SoC": _sym_from_pins("SoC", "U", [(n, nm, s) for n, nm, _net, s, _p in U1_PINS],
                          "Demo SoC"),
    "DRAM": _sym_from_pins("DRAM", "U", [(n, nm, s) for n, nm, _net, s, _p in U2_PINS],
                           "Demo DRAM"),
    "ClockRx": _sym_from_pins("ClockRx", "U", [(n, nm, s) for n, nm, _net, s, _p in U3_PINS],
                              "Demo CMOS clock receiver"),
    "Conn_PCIe": _sym_from_pins("Conn_PCIe", "J", [(n, net, "L") for n, net, _y in J1_PINS],
                                "Demo PCIe x2 TX connector"),
    "Conn_USB": _sym_from_pins("Conn_USB", "J", [(n, net, "L") for n, net, _y in J2_PINS],
                               "Demo USB connector"),
    "Conn_Power": SymDef("Conn_Power", "J", [("1", "+3V3", "L"), ("2", "GND", "L")],
                         w=5.08, h=7.62, descr="Power input"),
    "TestPoint": SymDef("TestPoint", "TP", [("1", "1", "B")], w=2.54, h=2.54,
                        descr="Test point"),
}


def sym_lib_text(sd: SymDef, lib_prefix: str) -> str:
    fsz = "(effects (font (size 1.27 1.27)))"
    hid = "(effects (font (size 1.27 1.27)) (hide yes))"
    name = f"{lib_prefix}{sd.name}"
    hw, hh = _box(sd)
    pins = []
    for num, (x, y, ang, nm, s) in _pin_points(sd).items():
        pins.append(f"(pin passive line (at {f(x)} {f(y)} {ang}) (length 2.54) "
                    f"(name {q(nm)} {fsz}) (number {q(num)} {fsz}))")
    small = sd.name in ("R", "C", "TestPoint")
    return (f"(symbol {q(name)} (pin_names (offset 0.508){' (hide yes)' if small else ''}) "
            f"(exclude_from_sim no) (in_bom yes) (on_board yes)\n"
            f"\t(property \"Reference\" {q(sd.ref)} (at {f(hw + 1.27)} {f(hh + 1.27)} 0) {fsz})\n"
            f"\t(property \"Value\" {q(sd.name)} (at {f(hw + 1.27)} {f(-hh - 1.27)} 0) {fsz})\n"
            f"\t(property \"Footprint\" \"\" (at 0 0 0) {hid})\n"
            f"\t(property \"Datasheet\" \"\" (at 0 0 0) {hid})\n"
            f"\t(property \"Description\" {q(sd.descr)} (at 0 0 0) {hid})\n"
            f"\t(symbol {q(sd.name + '_0_1')} (rectangle (start {f(-hw)} {f(hh)}) (end {f(hw)} {f(-hh)}) "
            f"(stroke (width 0.254) (type default)) (fill (type background))))\n"
            f"\t(symbol {q(sd.name + '_1_1')} {' '.join(pins)})\n"
            f"\t(embedded_fonts no))")


# Where each part goes on the A3 sheet (mm, y down), and its rotation 0.
SHEET_AT = {"U1": (152.4, 139.7), "U2": (152.4, 50.8), "U3": (279.4, 215.9),
            "J1": (330.2, 101.6), "J2": (330.2, 165.1), "J3": (38.1, 139.7),
            "TP1": (228.6, 246.38), "R1": (139.7, 215.9), "R5": (203.2, 241.3)}
for k in range(6):
    SHEET_AT[f"C{1 + k}"] = (241.3 + 12.7 * (k // 2), 101.6 + 25.4 * (k % 2) + (0 if k < 4 else 50.8))
for k in range(6):
    SHEET_AT[f"C{10 + k}"] = (63.5 + 12.7 * k, 190.5)


def write_symbols(path: Path):
    body = "\n".join(sym_lib_text(sd, "") for sd in SYMBOLS.values())
    path.write_text(f'(kicad_symbol_lib (version 20241209) (generator "kipi_si_demo") '
                    f'(generator_version "9.0")\n{body}\n)\n')


def write_sch(path: Path) -> Tuple[Dict[str, str], str]:
    root = uid("root sheet")
    fsz = "(effects (font (size 1.27 1.27)))"
    out = ["(kicad_sch (version 20250114) (generator \"kipi_si_demo\") (generator_version \"9.0\")",
           f"\t(uuid {q(root)})", '\t(paper "A3")',
           '\t(title_block (title "KiPI Signal Integrity demo") (comment 1 "Good and bad versions of PCIe, USB, DDR and clock nets") '
           '(comment 2 "Generated by examples/si_demo/make_project.py"))',
           "\t(lib_symbols"]
    out += ["\t\t" + sym_lib_text(sd, LIB + ":") for sd in SYMBOLS.values()]
    out.append("\t)")
    sym_uuid = {}
    for p in PARTS:
        sd = SYMBOLS[p.symbol]
        X, Y = SHEET_AT[p.ref]
        su = uid("sym", p.ref)
        sym_uuid[p.ref] = su
        hw, hh = _box(sd)
        pins = " ".join(f"(pin {q(num)} (uuid {q(uid('pin', p.ref, num))}))" for num, *_ in sd.pins)
        out.append(
            f'\t(symbol (lib_id {q(LIB + ":" + sd.name)}) (at {f(X)} {f(Y)} 0) (unit 1) '
            f'(exclude_from_sim no) (in_bom yes) (on_board yes) (dnp {"yes" if p.dnp else "no"}) '
            f'(uuid {q(su)})\n'
            f'\t\t(property "Reference" {q(p.ref)} (at {f(X + hw + 1.27)} {f(Y - hh - 1.27)} 0) {fsz})\n'
            f'\t\t(property "Value" {q(p.value)} (at {f(X + hw + 1.27)} {f(Y + hh + 1.27)} 0) {fsz})\n'
            f'\t\t(property "Footprint" {q(LIB + ":" + p.fp)} (at {f(X)} {f(Y)} 0) (effects (font (size 1.27 1.27)) (hide yes)))\n'
            f'\t\t(property "Datasheet" "" (at {f(X)} {f(Y)} 0) (effects (font (size 1.27 1.27)) (hide yes)))\n'
            f'\t\t(property "Description" {q(p.note)} (at {f(X)} {f(Y)} 0) (effects (font (size 1.27 1.27)) (hide yes)))\n'
            f'\t\t{pins}\n'
            f'\t\t(instances (project {q(NAME)} (path {q("/" + root)} (reference {q(p.ref)}) (unit 1))))\n\t)')
        # A global label at each pin's end names its net.
        for num, (x, y, ang, _nm, s) in _pin_points(sd).items():
            net = p.nets.get(num)
            if not net:
                continue
            lx, ly = X + x, Y - y
            l_ang, just = {"L": (180, "right"), "R": (0, "left"),
                           "T": (90, "left"), "B": (270, "right")}[s]
            out.append(f'\t(global_label {q(net)} (shape passive) (at {f(lx)} {f(ly)} {l_ang}) '
                       f'(fields_autoplaced yes) (effects (font (size 1.27 1.27)) (justify {just})) '
                       f'(uuid {q(uid("label", p.ref, num))}) '
                       f'(property "Intersheetrefs" "${{INTERSHEET_REFS}}" (at {f(lx)} {f(ly)} 0) '
                       f'(effects (font (size 1.27 1.27)) (hide yes))))')
    out.append('\t(sheet_instances (path "/" (page "1")))')
    out.append("\t(embedded_fonts no)\n)")
    path.write_text("\n".join(out) + "\n")
    return sym_uuid, root


def write_project(path: Path):
    classes = [{
        "name": "Default", "clearance": 0.1, "track_width": W_SE, "via_diameter": VIA[0],
        "via_drill": VIA[1], "microvia_diameter": 0.3, "microvia_drill": 0.1,
        "diff_pair_width": PCIE_F[0], "diff_pair_gap": PCIE_F[1], "diff_pair_via_gap": 0.25,
        "bus_width": 12, "wire_width": 6, "line_style": 0, "priority": 2147483647,
        "pcb_color": "rgba(0, 0, 0, 0.000)", "schematic_color": "rgba(0, 0, 0, 0.000)"}]
    proj = {
        "board": {"design_settings": {
            "defaults": {"board_outline_line_width": 0.1, "copper_line_width": 0.2},
            "rules": {"min_clearance": 0.1, "min_connection": 0.1, "min_copper_edge_clearance": 0.3,
                      "min_hole_clearance": 0.15, "min_hole_to_hole": 0.25, "min_track_width": 0.1,
                      "min_via_annular_width": 0.1, "min_via_diameter": 0.4,
                      "min_through_hole_diameter": 0.2, "min_microvia_diameter": 0.2,
                      "min_microvia_drill": 0.1, "min_silk_clearance": 0.0,
                      "min_text_height": 0.5, "min_text_thickness": 0.08,
                      "solder_mask_to_copper_clearance": 0.0, "max_error": 0.005,
                      "min_resolved_spokes": 1, "min_groove_width": 0.0,
                      "use_height_for_length_calcs": True},
            "rule_severities": {"silk_overlap": "ignore", "silk_over_copper": "ignore",
                                "silk_edge_clearance": "ignore", "text_height": "ignore",
                                "text_thickness": "ignore", "lib_footprint_issues": "ignore",
                                "lib_footprint_mismatch": "ignore", "courtyards_overlap": "warning"},
            "track_widths": [0.0, W_SE, W_DQ, PCIE_F[0], USB_F[0], PWR_W],
            "via_dimensions": [{"diameter": 0.0, "drill": 0.0},
                               {"diameter": VIA[0], "drill": VIA[1]}],
            "diff_pair_dimensions": [{"gap": 0.0, "via_gap": 0.0, "width": 0.0},
                                     {"gap": PCIE_F[1], "via_gap": 0.25, "width": PCIE_F[0]},
                                     {"gap": USB_F[1], "via_gap": 0.25, "width": USB_F[0]},
                                     {"gap": PCIE_IN2[1], "via_gap": 0.25, "width": PCIE_IN2[0]}],
        }},
        "net_settings": {"classes": classes, "meta": {"version": 4}},
        "meta": {"filename": f"{NAME}.kicad_pro", "version": 3},
    }
    path.write_text(json.dumps(proj, indent=2) + "\n")


def write_tables(d: Path):
    (d / "fp-lib-table").write_text(
        f'(fp_lib_table\n\t(version 7)\n\t(lib (name "{LIB}")(type "KiCad")(uri "${{KIPRJMOD}}/{LIB}.pretty")'
        f'(options "")(descr "KiPI SI demo footprints"))\n)\n')
    (d / "sym-lib-table").write_text(
        f'(sym_lib_table\n\t(version 7)\n\t(lib (name "{LIB}")(type "KiCad")(uri "${{KIPRJMOD}}/{LIB}.kicad_sym")'
        f'(options "")(descr "KiPI SI demo symbols"))\n)\n')


# ============================================================================ KiPI setup
def write_kipi_setup(path: Path) -> bool:
    """A KiPI setup with a channel per interface, when KiPI can be imported."""
    sys.path.insert(0, str(HERE.parents[1]))
    try:
        from kipi.dc.pi.model import PISetup
        from kipi.si.model import match_generic, new_channel
    except Exception as e:  # noqa: BLE001
        print(f"KiPI not importable ({e}): no {path.name}")
        return False
    setup = PISetup()
    chans = [
        ("PCIe TX0 (good)", "PCIE_TX0_P", "PCIE_TX0_N", "pcie3", "U1", "J1", ["PCIE_TX1_P", "PCIE_TX1_N"]),
        ("PCIe TX1 (bad)", "PCIE_TX1_P", "PCIE_TX1_N", "pcie3", "U1", "J1", ["PCIE_TX0_P", "PCIE_TX0_N"]),
        ("USB3 TX (good)", "USB3_TX_P", "USB3_TX_N", "usb3g1", "U1", "J2", []),
        ("USB2 D (bad: plane slot)", "USB_D_P", "USB_D_N", "usb2", "U1", "J2", []),
        ("DQ1 (good: 3W)", "DQ1", "", "ddr4_dq", "U1", "U2", ["DQ0", "DQ2"]),
        ("DQ5 (bad: tight)", "DQ5", "", "ddr4_dq", "U1", "U2", ["DQ4", "DQ6"]),
        ("CLK_GOOD (series R)", "CLK_GOOD", "", "cmos", "U1", "U3", []),
        ("CLK_BAD (no termination)", "CLK_BAD", "", "cmos", "U1", "U3", []),
    ]
    for name, net, partner, proto, drv, rcv, agg in chans:
        ch = new_channel(net, partner, proto)
        ch.name, ch.driver_ref, ch.receiver_ref, ch.aggressors = name, drv, rcv, agg
        if proto == "cmos":
            # A strong, fast CMOS driver: with R1 the good clock's source is ~50 Ω; the
            # bad clock's 15 Ω meets its 50 Ω line unmatched and rings.
            ch.rise_s, ch.r_source_ohm = 0.3e-9, 15.0
        match_generic(ch)
        setup.si_channels.append(ch)
    setup.save(path)
    return True


# ============================================================================ main
def run(cmd) -> Tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-kicad", action="store_true",
                    help="do not fill the zones or check with kicad-cli")
    args = ap.parse_args(argv)
    pcb, sch = HERE / f"{NAME}.kicad_pcb", HERE / f"{NAME}.kicad_sch"
    write_footprints(HERE / f"{LIB}.pretty")
    write_symbols(HERE / f"{LIB}.kicad_sym")
    write_tables(HERE)
    write_project(HERE / f"{NAME}.kicad_pro")
    sym_uuid, root = write_sch(sch)
    write_pcb(pcb, sym_uuid, root)
    print(f"wrote {pcb.name}, {sch.name}: {len(PARTS)} parts, {len(NETS)} nets, "
          f"{sum(len(t.pts) - 1 for t in TRACKS)} track segments, {len(VIAS)} vias")
    ok = True
    cli = shutil.which("kicad-cli")
    if args.no_kicad or cli is None:
        print("kicad-cli not run: the zones are not filled (fill them in KiCad: B)")
    else:
        prl = HERE / f"{NAME}.kicad_prl"
        had_prl = prl.exists()
        with tempfile.TemporaryDirectory() as tmp:
            for what, cmd in (
                    ("DRC (zones refilled, schematic parity)",
                     [cli, "pcb", "drc", "--refill-zones", "--save-board", "--schematic-parity",
                      "--severity-error", "--severity-warning", "--exit-code-violations"]),
                    ("ERC", [cli, "sch", "erc", "--severity-error", "--severity-warning",
                             "--exit-code-violations"])):
                rpt = Path(tmp) / "report.rpt"
                code, out = run(cmd + ["-o", str(rpt), str(pcb if cmd[1] == "pcb" else sch)])
                print(f"{what}: {'clean' if code == 0 else 'VIOLATIONS'}")
                if code != 0:
                    ok = False
                    print(rpt.read_text() if rpt.exists() else out)
        if not had_prl and prl.exists():
            prl.unlink()                    # kicad-cli's local view settings
    write_kipi_setup(HERE / f"{NAME}.kipi-setup.yml")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
