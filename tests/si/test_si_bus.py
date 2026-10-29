"""Buses: the nets between a transmitting and a receiving part as channels, the Add bus
dialog and the channel editor's transmitter and receiver pickers."""

import os

import pytest

from helpers.si_boards import example, make_board, make_setup
from kipi.core.model.board import BoardSnapshot, Footprint, Net, Track
from kipi.si.bus import bus_nets, connected_parts, make_channels, parts_on


def ddr_lane() -> BoardSnapshot:
    """A byte lane's slice from U1 to U2: DQ0, DQ1, DQ3 and the DQS0 pair straight, DQ2
    through series resistor R5."""
    F, pads, tracks = example.F, [], []

    def pad(net, ref, num, x, y):
        pads.append(example._pad(f"{ref}.{num}", net, ref, num, x, y))

    for k, n in enumerate(["DQ0", "DQ1", "DQ3", "DQS0_P", "DQS0_N"]):
        y = 2.0 + k
        pad(n, "U1", str(k + 1), 2.0, y)
        pad(n, "U2", str(k + 1), 40.0, y)
        tracks.append(Track(n, F, 0.1, (2.0, y), (40.0, y)))
    pad("DQ2", "U1", "9", 2.0, 10.0)
    pad("DQ2", "R5", "1", 20.0, 10.0)
    pad("DQ2_R", "R5", "2", 21.0, 10.0)
    pad("DQ2_R", "U2", "9", 40.0, 10.0)
    tracks += [Track("DQ2", F, 0.1, (2, 10), (20, 10)),
               Track("DQ2_R", F, 0.1, (21, 10), (40, 10))]
    fps = [Footprint(r, r, v, (0, 0), "top")
           for r, v in (("U1", "SOC"), ("U2", "DDR4"), ("R5", "22"))]
    names = ["", "GND", "DQ0", "DQ1", "DQ2", "DQ2_R", "DQ3", "DQS0_P", "DQS0_N"]
    return BoardSnapshot("ddr.kicad_pcb", "", example.stackup(),
                         [Net(i, n) for i, n in enumerate(names)], tracks, [], pads, fps, [],
                         [example.Poly(example.rect(0, 0, 50, 20))], source="synthetic")


def test_bus_between_two_parts():
    snap = ddr_lane()
    assert connected_parts(snap, "U1") == [("U2", 6)]       # not R5: a series part
    bus = bus_nets(snap, "U1", "U2")
    assert [(b.name, b.group, b.index) for b in bus] == [
        ("DQS0_P/DQS0_N", "DQS", 0), ("DQ0", "DQ", 0), ("DQ1", "DQ", 1), ("DQ2", "DQ", 2),
        ("DQ3", "DQ", 3)]
    assert bus[3].series == ("DQ2_R",)
    chans = make_channels(snap, bus, "U1", "U2", aggressors=1, taken=["DQ0"])
    by = {c.name: c for c in chans}
    assert "DQ0" not in by and len(chans) == 4
    assert by["DQ2"].aggressors == ["DQ1", "DQ3"]
    assert by["DQS0_P/DQS0_N"].partner == "DQS0_N"
    assert all((c.driver_ref, c.receiver_ref) == ("U1", "U2") for c in chans)
    # The other direction (a read): the same nets, the ends swapped.
    back = make_channels(snap, bus_nets(snap, "U2", "U1"), "U2", "U1")
    assert {c.name for c in back} == {"DQS0_P/DQS0_N", "DQ0", "DQ1", "DQ2_R", "DQ3"}


def test_parts_on_skips_series_parts():
    snap = make_board()
    assert parts_on(snap, ["CLK", "CLK_R"]) == [("U1", 1), ("U2", 1)]
    assert parts_on(snap, ["USB_P", "USB_N"]) == [("U3", 2), ("J1", 2)]


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("qtpy")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_bus_dialog(app):
    from qtpy.QtCore import Qt
    from kipi.si.gui.bus_dialog import BusDialog
    d = BusDialog(ddr_lane(), taken=["DQ1"], tx="U1", rx="U2")
    assert d.cmb_tx.currentData() == "U1" and d.cmb_rx.currentData() == "U2"
    assert d.tbl.rowCount() == 5
    assert [b.name for b in d.chosen()] == ["DQS0_P/DQS0_N", "DQ0", "DQ2", "DQ3"]
    d.cmb_group.setCurrentIndex(d.cmb_group.findData("DQS"))
    d._tick_group(0)
    assert [b.name for b in d.chosen()] == ["DQS0_P/DQS0_N"]
    d.tbl.item(1, 0).setCheckState(Qt.Checked)
    assert "2 channels" in d.lbl.text()
    d._accept()
    assert [c.name for c in d.channels] == ["DQS0_P/DQS0_N", "DQ0"]
    assert d.channels[1].aggressors == ["DQ1"]
    d._swap()
    assert (d.cmb_tx.currentData(), d.cmb_rx.currentData()) == ("U2", "U1")


def test_channel_ends_pickers(app):
    from kipi.si.gui.panel import SIPanel
    snap, setup = make_board(), make_setup()
    p = SIPanel()
    p.set_board(snap, setup)
    p.ch_list.select_name("CLK")
    ed = p.ch_list.editor
    texts = [ed.cmb_drv.itemText(i) for i in range(ed.cmb_drv.count())]
    assert texts[0].startswith("guessed (U") and any(t.startswith("U2") for t in texts)
    assert not any(t.startswith("R1") for t in texts)        # a series part: not an end
    ch = setup.si_channels[0]
    ed.cmb_rcv.setCurrentIndex(ed.cmb_rcv.findData("U1"))
    ed.cmb_rcv.activated.emit(ed.cmb_rcv.currentIndex())
    assert ch.receiver_ref == "U1" and ch.driver_ref == ""
    ed._swap_ends()
    assert (ch.driver_ref, ch.receiver_ref) == ("U1", "U2")
    ed._swap_ends()
    assert (ch.driver_ref, ch.receiver_ref) == ("U2", "U1")


def test_one_end_set_guesses_the_other():
    from kipi.si.model import Channel
    from kipi.si.solver import ends
    snap, nets = make_board(), ["CLK", "CLK_R"]
    ch = Channel("CLK", "CLK")
    assert ends(snap, ch, nets) == ("U1", "U2")
    ch.receiver_ref = "U1"
    assert ends(snap, ch, nets) == ("U2", "U1")         # not U1 driving itself
    ch.receiver_ref, ch.driver_ref = "", "U2"
    assert ends(snap, ch, nets) == ("U2", "U1")
