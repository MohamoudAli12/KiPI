"""The channel solver end to end on the synthetic board, and its waveform checks."""

import numpy as np
import pytest

from helpers.si_boards import make_board
from kipi.dc.pi.model import PISetup
from kipi.si.model import new_channel
from kipi.si.scan import FAIL, PASS
from kipi.si.solver import SIError, edge_metrics, solve_channel


@pytest.fixture(scope="module")
def board():
    return make_board()


def _setup(*channels):
    s = PISetup()
    s.si_channels = list(channels)
    return s


def test_single_ended_channel(board):
    r = solve_channel(board, _setup(new_channel("CLK")), "CLK")
    assert r.port_names[0].startswith("TX U1") and "CLK_R" in r.port_names[1]
    assert r.nets == ["CLK", "CLK_R"]
    # Through 33 Ω into 50 Ω ports at DC: S21 = 100 / 133.
    assert abs(abs(r.s[0, 1, 0]) - 100 / 133) < 0.01
    assert r.eye.height_v > 2.0 and r.eye.width_ui > 0.7
    assert r.waves is not None and max(abs(v).max() for v in r.waves.v.values()) > 3.0
    assert {c.check for c in r.checks} >= {"impedance", "eye", "overshoot", "ringback"}


def test_pair_channel(board):
    r = solve_channel(board, _setup(new_channel("USB_P", "USB_N")), "USB_P/USB_N")
    assert r.mm.shape[1:] == (4, 4) and r.z_ref == 100.0
    k = int(np.argmin(abs(r.freqs - r.nyquist_hz)))
    assert -1 < r.il_db[k] < 0
    assert r.conv_db[r.freqs <= r.nyquist_hz].max() < -20
    assert r.skew_s is not None and r.skew_s < 5e-12
    # The TDR settles near the pair's Zdiff (≈ 93 Ω on this board) before the far end.
    assert 80 < np.median(r.tdr_z[(r.tdr_t > 50e-12) & (r.tdr_t < 150e-12)]) < 110
    assert r.eye.width_ui > 0.7
    assert {"RX diff", "TX diff"} <= set(r.waves.v)
    assert r.status != FAIL


def test_unknown_channel_and_net(board):
    with pytest.raises(SIError):
        solve_channel(board, _setup(), "nope")
    with pytest.raises(SIError):
        solve_channel(board, _setup(new_channel("NOPE")), "NOPE")


def test_edge_metrics():
    t = np.linspace(0, 40e-9, 4001)
    v = np.where((t > 5e-9) & (t < 25e-9), 1.0, 0.0)
    clean = edge_metrics(t, v, 0.0, 1.0, 0.35, 0.65)
    assert clean["overshoot_v"] == 0 and clean["monotonic"]
    assert abs(clean["ringback_v"] - 0.35) < 1e-9
    ring = v + np.where(t > 5e-9, 0.5 * np.exp(-(t - 5e-9) / 2e-9) * np.cos(2e9 * 6.283 * (t - 5e-9)), 0)
    ring[t >= 25e-9] = 0
    m = edge_metrics(t, ring, 0.0, 1.0, 0.35, 0.65)
    assert m["overshoot_v"] > 0.3 and m["ringback_v"] < 0.35
    assert m["ringback_v"] != clean["ringback_v"]
    assert PASS  # the module's constants import


def test_a_pairs_impedance_is_judged_over_its_whole_route():
    """1 mm off target in a 15 mm route is a WARN, though only 5 mm of it has a Zdiff
    (the rest runs beside an aggressor or with P alone)."""
    from kipi.si.extract import SectionInfo
    from kipi.si.scan import WARN
    from kipi.si.solver import ChannelResult, SISettings, _checks
    ch = new_channel("USB_P", "USB_N")
    f = np.linspace(1e7, 1e9, 10)
    res = ChannelResult(ch, ["USB_P", "USB_N"], [], f, np.zeros((10, 4, 4)))
    res.sections = [SectionInfo(["USB_P", "USB_N"], [], [50, 50], zdiff=70.0, length_mm=1.0),
                    SectionInfo(["USB_P", "USB_N"], [], [50, 50], zdiff=90.0, length_mm=4.0),
                    SectionInfo(["USB_P", "USB_N", "AGG"], [], [50, 50, 50], length_mm=6.0),
                    SectionInfo(["USB_P"], [], [55], length_mm=4.0),
                    SectionInfo(["USB_N"], [], [55], length_mm=30.0)]
    _checks(res, ch, SISettings(), ["USB_P"])
    imp = next(c for c in res.checks if c.check == "impedance")
    assert imp.status == WARN and "off over 1.0 of 15.0 mm" in imp.text


def test_a_pairs_overlay_covers_its_whole_route(board):
    """Every segment of the pair is coloured, with its copper: grey (NaN) only where a leg
    runs without its partner."""
    from kipi.si.viz import channel_overlay
    r = solve_channel(board, _setup(new_channel("USB_P", "USB_N")), "USB_P/USB_N")
    ov = channel_overlay(board, r)
    n = sum(len(s.segs) for s in r.sections)
    assert len(ov.segments) == len(ov.widths) == len(ov.z_spans) == n
    assert np.allclose(ov.widths, 0.2)
    lone = [np.isnan(v) for s in r.sections for v in [s.zdiff if s.zdiff is not None
                                                      else np.nan] * len(s.segs)]
    assert np.array_equal(np.isnan(ov.segment_mv), lone) and not all(lone)
    assert (ov.z_spans[:, 1] > ov.z_spans[:, 0]).all()


def test_the_ends_are_marked_where_the_parts_are(board):
    """TX over the driving part's pads on the channel's nets, RX over the receiving part's,
    each a set end or the solver's guess; swapped ends swap the labels; none with no net."""
    from kipi.si.viz import end_labels
    ch = new_channel("CLK")
    labels = end_labels(board, PISetup().si, ch)
    assert [t for t, _ in labels] == ["TX U1", "RX U2"]
    pads = {r: [p for n in ("CLK", "CLK_R") for p in board.items_on_net(n)[2]
                if p.footprint == r] for r in ("U1", "U2")}
    for (_t, xyz), ref in zip(labels, ("U1", "U2")):
        assert np.allclose(xyz[:2], np.mean([p.position for p in pads[ref]], axis=0))
    ch.driver_ref, ch.receiver_ref = "U2", "U1"
    swapped = end_labels(board, PISetup().si, ch)
    assert [t for t, _ in swapped] == ["TX U2", "RX U1"]
    assert np.allclose(swapped[0][1], labels[1][1]) and np.allclose(swapped[1][1], labels[0][1])
    assert end_labels(board, PISetup().si, new_channel("")) == []


def test_tdr_note_names_a_source_termination():
    """The demo's CLK_GOOD: R1 (33 Ω) 2.5 mm from U1, so the TDR from U1 reads Z0 + 33 Ω
    from ~31 ps; a channel with no series resistor gets no note."""
    from pathlib import Path
    from types import SimpleNamespace
    from kipi.si.model import new_channel
    from kipi.si.solver import _tdr_note
    from kipi.sources.kicad_pcb.reader import read_board
    pcb = Path(__file__).resolve().parents[2] / "examples" / "si_demo" / "si_demo.kicad_pcb"
    snap = read_board(pcb)
    chan = SimpleNamespace(net_delay=lambda n: {"CLK_GOOD": 15.5e-12}.get(n, 400e-12))
    ch = new_channel("CLK_GOOD", "", "cmos")
    note = _tdr_note(snap, ch, chan, ["CLK_GOOD", "CLK_GOOD_R"], "U1")
    assert "R1 (33 Ω) from 31 ps" in note and "plus 33 Ω" in note
    # From U3's end the step reaches R1 only after the long line.
    assert "from 800 ps" in _tdr_note(snap, ch, chan, ["CLK_GOOD", "CLK_GOOD_R"], "U3")
    assert _tdr_note(snap, new_channel("CLK_BAD", "", "cmos"), chan, ["CLK_BAD"], "U1") == ""
