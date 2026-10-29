"""The Signal Integrity panel and plots: a channel from the Quick Scan, a solved result
shown with its checks, the impedance overlay, every plot view, and the report."""

import os

import pytest

pytest.importorskip("qtpy")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QFormLayout  # noqa: E402

from helpers.si_boards import make_board  # noqa: E402
from kipi.dc.pi.model import PISetup  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def solved():
    from kipi.si.model import new_channel
    from kipi.si.scan import scan, settings_of
    from kipi.si.solver import solve_channel
    snap = make_board()
    setup = PISetup()
    setup.si_channels = [new_channel("USB_P", "USB_N")]
    result = solve_channel(snap, setup, "USB_P/USB_N")
    result.solved_setup = setup
    return snap, setup, result, scan(snap, None, settings_of(setup.si))


@pytest.fixture
def panel(app):
    from kipi.si.gui.panel import SIPanel
    p = SIPanel()
    p.set_board(make_board(), PISetup())
    yield p
    p.close()


def test_without_a_channel_it_says_what_to_add(panel):
    assert not panel.btn_solve.isEnabled()
    assert "Add a channel" in panel.lbl_errors.text()


def test_a_scanned_net_becomes_a_channel(panel, solved):
    _, _, _, sc = solved
    panel.show_scan(sc)
    assert panel.tbl_scan.rowCount() == len(sc.nets)
    row = next(r for r in range(panel.tbl_scan.rowCount())
               if panel.tbl_scan.item(r, 0).text().startswith("USB"))
    panel.tbl_scan.selectRow(row)
    assert panel.overlay() is not None          # the scan's impedance in 3D
    panel.btn_to_channel.click()
    names = [c.name for c in panel.setup.si_channels]
    assert len(names) == 1 and "USB_P" in names[0] and "USB_N" in names[0]
    assert panel.setup.si_channels[0].protocol == "usb2"
    assert panel.btn_solve.isEnabled()
    panel.btn_to_channel.click()                # the same net again: no second channel
    assert len(panel.setup.si_channels) == 1


def test_the_editor_writes_overrides(panel):
    panel.ch_list.add(panel._new_channel())
    ed = panel.ch_list.editor
    ch = ed.obj
    ed.spins["rate_bps"].setValue(2.5)
    assert ch.rate_bps == pytest.approx(2.5e9)
    ed.spins["rate_bps"].setValue(0)            # the minimum: the interface's own
    assert ch.rate_bps is None


def test_a_result_shows_its_checks_and_plots(app, panel, solved):
    from kipi.si.gui.plot import VIEWS, SIPlot
    snap, setup, result, _ = solved
    panel.set_board(snap, setup)
    shown = []
    panel.result_selected.connect(shown.append)
    panel.show_result(result)
    assert shown and shown[-1] is result
    assert panel.tbl_checks.rowCount() == len(result.checks)
    assert panel.cmb_via.count() >= 1           # the pair's vias, for the 3D check
    ov = panel.overlay()
    assert ov is not None and len(ov.segments) > 0
    plot = SIPlot()
    plot.set_result(result)
    for key, _ in VIEWS:
        plot.set_view(key)                      # each draws without raising
    plot.close()


def test_the_pdf_report(app, solved, tmp_path):
    from kipi.si.html_report import write_si_report
    snap, setup, result, sc = solved
    out = write_si_report(tmp_path / "r.pdf", result, snap, setup, sc)
    assert out.stat().st_size > 10_000
    html = write_si_report(tmp_path / "r.html", result, snap, setup, sc)
    text = html.read_text(encoding="utf-8")
    assert "Signal Integrity report" in text and "download=" in text


def test_results_round_trip(solved, tmp_path):
    from kipi.si import report
    snap, setup, result, sc = solved
    f = report.save_results(tmp_path / "x.kipi-si-results", {result.channel.name: result}, sc,
                            "b", setup)
    d = report.load_results(f)
    r = d["results"][result.channel.name]
    assert r.s.shape == result.s.shape and abs(r.eye.height_v - result.eye.height_v) < 1e-12
    assert len(d["scan"].nets) == len(sc.nets)
    s4p = report.export_touchstone(tmp_path / "c", r)
    assert s4p.suffix == ".s4p" and "Port[4]" in s4p.read_text()


def test_the_3d_check_says_why_it_cannot_run(app, panel):
    """A channel with no vias has nothing for the openEMS check: the button is off and the
    list says why, rather than sitting empty."""
    from kipi.si.model import new_channel
    from kipi.si.solver import solve_channel
    snap = make_board()
    setup = PISetup()
    setup.si_channels = [new_channel("CLK")]
    panel.set_board(snap, setup)
    assert "Solve" in panel.cmb_via.placeholderText()
    panel.show_result(solve_channel(snap, setup, "CLK"))
    assert panel.cmb_via.count() == 0 and not panel.btn_em.isEnabled()
    assert "no vias" in panel.cmb_via.placeholderText()
    assert "no vias" in panel.btn_em.toolTip()



def test_solve_is_under_the_channel_settings(panel):
    # Set a channel up and solve it without leaving its settings: Solve is the form's
    # last row, below the buffers.
    ed = panel.ch_list.editor
    assert panel.btn_solve is ed.btn_solve
    last = ed.form.itemAt(ed.form.rowCount() - 1, QFormLayout.ItemRole.SpanningRole)
    assert last is not None and last.widget() is panel.btn_solve


def test_not_fitted_parts_round_trip(panel):
    import dataclasses

    from kipi.si.model import channel_from_dict, new_channel
    setup = PISetup()
    ch = new_channel("CLK")
    ch.name = "CLK"
    setup.si_channels.append(ch)
    panel.set_board(make_board(), setup)
    ed = panel.ch_list.editor
    ed.ed_nofit.setText("R1; C7")
    ed.ed_nofit.editingFinished.emit()
    assert setup.si_channels[0].not_fitted == ["R1", "C7"]
    assert channel_from_dict(dataclasses.asdict(ch)).not_fitted == ["R1", "C7"]


def test_the_build_lists_the_boards_variants(panel):
    snap = make_board()
    snap.variants = {"lite": "no clock buffer"}
    next(f for f in snap.footprints if f.reference == "R1").variant_dnp = {"lite": True}
    setup = PISetup()
    panel.set_board(snap, setup)
    c = panel.cmb_variant
    assert [c.itemData(i) for i in range(c.count())] == ["", "lite"]
    assert "1 not fitted" in c.itemText(1)
    c.setCurrentIndex(1)
    assert setup.si.variant == "lite"
