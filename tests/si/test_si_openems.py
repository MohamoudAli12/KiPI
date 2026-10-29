"""The openEMS model: the input it writes, its mesh and its ports, and turning the port
probes into S-parameters. A real run is slow (minutes): it runs only with KIPI_RUN_OPENEMS
set, and openEMS installed."""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from helpers.si_boards import make_board
from kipi.si import openems as oe
from kipi.si.board import SIBoard
from kipi.si import sparams

REGION = (28.5, 13.5, 31.5, 16.5)          # around the pair's vias


@pytest.fixture(scope="module")
def board():
    return SIBoard(make_board())


def test_find_openems_from_the_environment(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "openEMS"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("KIPI_OPENEMS", str(tmp_path))
    assert oe.find_openems() == str(exe)


def test_graded_lines_keep_the_features_and_grade():
    lines = oe.graded_lines([0.0, 1.0, 1.05, 10.0], [0.05, 0.05, 0.05, 0.5], 1.0, 1.4)
    for x in (0.0, 1.0, 1.05, 10.0):
        assert np.min(np.abs(lines - x)) < 1e-6
    h = np.diff(lines)
    assert h.max() <= 1.0 + 1e-9 and h.min() > 0
    assert np.all(h[1:] / h[:-1] < 1.4 * 1.5) and np.all(h[:-1] / h[1:] < 1.4 * 1.5)


def test_model_xml(board):
    m = oe.build_model(board, REGION, ["USB_P"], oe.EMSettings(f_max_hz=10e9, fine_mm=0.1))
    assert len(m.ports) == 2
    assert {p.layer for p in m.ports} == {0, 31} or len({p.layer for p in m.ports}) == 2
    root = ET.fromstring(m.xml(0))
    exc = root.find("FDTD/Excitation")
    assert float(exc.get("f0")) == pytest.approx(oe.F0_SHARE * 10e9)
    names = [e.get("Name") for e in root.iter() if e.get("Name")]
    assert "port_excite_1" in names and "port_excite_2" not in names
    assert any(n.startswith("port_resist_2") for n in names)
    # The planes are whole around the GND stitching vias: holes only at the antipads.
    holes = [p for p in root.iter("Polygon") if p.get("Priority") == "60"]
    assert len(holes) == 4                     # two planes × the pair's two vias
    assert m.cells > 1000
    # The planes and dielectrics go on into the absorbing walls (no edges to resonate).
    xs = [float(v.get("X1")) for poly in root.iter("Polygon") if poly.get("Priority") == "50"
          for v in poly.iter("Vertex")]
    boxes = [float(b.find("P1").get("X")) for b in root.iter("Box")]
    assert min(xs) < m.mesh["x"][0] and min(boxes) < m.mesh["x"][0]


def test_dft_of_a_tone():
    t = np.arange(20000) * 1e-12
    v = np.cos(2 * np.pi * 5e9 * t)
    out = oe.dft(t, v, np.array([5e9]))
    assert abs(out[0]) == pytest.approx(t[-1] + 1e-12, rel=0.01)


@pytest.mark.skipif(not (os.environ.get("KIPI_RUN_OPENEMS") and oe.find_openems()),
                    reason="set KIPI_RUN_OPENEMS (and install openEMS) for the 3D run")
@pytest.mark.timeout(1800)          # a 3D run: minutes, not the suite's 300 s
def test_via_check_against_kipi(tmp_path):
    from kipi.dc.pi.model import PISetup
    from kipi.si.model import new_channel
    from kipi.si.verify import verify_via
    snap = make_board()
    setup = PISetup()
    setup.si_channels = [new_channel("USB_P", "USB_N")]
    setup.si.em_f_max_hz, setup.si.em_fine_mm, setup.si.em_end_db = 10e9, 0.1, -30
    setup.si.em_margin_mm = 1.5
    via = next(v for v in snap.vias if v.net == "USB_P")
    c = verify_via(snap, setup, "USB_P/USB_N", via.id, tmp_path)
    th = c.through()
    assert th is not None
    # The through path within a dB over the whole band (no resonance: the plane pair has
    # no edges in either). The reflection, a few % of the wave, openEMS puts higher, its
    # lumped ports adding their own (a plain trace shows the same): only its scale here.
    assert c.worst_db(*th) < 1.0
    assert c.worst_db(0, 0) < 15.0
    assert sparams.passivity(c.s_em) < 1.1


def test_ports_and_copper_are_outside_the_pml(board):
    s = oe.EMSettings(f_max_hz=10e9, fine_mm=0.1)
    m = oe.build_model(board, REGION, ["USB_P"], s)
    n = s.pml_cells
    for p in m.ports:
        for k, axis in enumerate("xyz"):
            lines = m.mesh[axis]
            for v in (p.start[k], p.stop[k]):
                assert lines[n] < v < lines[-1 - n], (axis, v)
    top = board.stack.copper[0].z_mid_mm
    assert top < m.mesh["z"][-1 - n]


@pytest.mark.skipif(not (os.environ.get("KIPI_RUN_OPENEMS") and oe.find_openems()),
                    reason="set KIPI_RUN_OPENEMS (and install openEMS) for the 3D run")
def test_microstrip_against_kipi(board, tmp_path):
    from kipi.si import sparams
    from kipi.si.extract import BuildOptions
    from kipi.si.verify import kipi_model
    region = (10.0, 3.0, 14.0, 7.0)               # 4 mm of the CLK microstrip
    m = oe.build_model(board, region, ["CLK"],
                       oe.EMSettings(f_max_hz=10e9, fine_mm=0.1, end_db=-30))
    r = oe.run(m, tmp_path)
    k = kipi_model(board, ["CLK"], region, m.ports, r.freqs_hz, BuildOptions(0.5))
    assert np.max(np.abs(sparams.db(r.s[:, 1, 0]) - sparams.db(k[:, 1, 0]))) < 0.2
    phase = np.degrees(np.angle(r.s[:, 1, 0] / k[:, 1, 0]))
    assert np.max(np.abs(phase)) < 8
    assert sparams.db(r.s[:, 0, 0]).max() < -15
