"""The Signal Integrity panel and plots: a channel from the Quick Scan, a solved result
shown with its checks, the impedance overlay, every plot view, and the report."""

import os

import pytest

pytest.importorskip("qtpy")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QFormLayout  # noqa: E402

from helpers.si_boards import make_board  # noqa: E402
from kipi.dc.pi.model import PISetup  # noqa: E402
from kipi.si.model import new_channel  # noqa: E402


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


def test_add_makes_channels_for_the_nets_ticked_in_board(panel):
    add = panel.ch_list.buttons.itemAt(0).widget()
    add.click()                                 # nothing ticked: nothing added, and why
    assert panel.setup.si_channels == []
    assert "Tick the nets" in panel.lbl_errors.text()
    panel.set_nets(["USB_N", "CLK", "GND"])     # a pair's one leg: the pair, P its net
    add.click()
    chans = [(c.net, c.partner) for c in panel.setup.si_channels]
    assert chans == [("CLK", ""), ("USB_P", "USB_N")]
    assert panel.ch_list.current().net == "CLK"
    add.click()                                 # the same nets again: no second channel
    assert len(panel.setup.si_channels) == 2
    assert "has a channel already" in panel.lbl_errors.text()


def test_the_editor_writes_overrides(panel):
    panel.ch_list.add(new_channel("USB_P", "USB_N"))
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
    ov = panel.overlay()
    assert ov is not None and len(ov.segments) > 0
    panel.chk_res_3d.setChecked(False)          # Settings' "Solved channel's impedance"
    assert panel.overlay() is None
    panel.chk_res_3d.setChecked(True)           # the module's setup, as it was
    plot = SIPlot()
    plot.set_result(result)
    for key, _ in VIEWS:
        plot.set_view(key)                      # each draws without raising
    plot.close()


def test_the_solved_channels_compared(app, solved):
    from kipi.si.gui.plot import SIPlot
    results = _two_results(solved[2])
    plot = SIPlot()
    plot.set_result(results[0])
    plot.set_view("compare")
    assert "Solve channels" in plot.figure.axes[0].get_title()   # no results given yet
    plot.results_source = lambda: results
    plot.refresh_compare()
    assert len(plot.figure.axes) == 4 and "2 solved channels" in plot.lbl.text()
    names = [t.get_text() for t in plot.figure.axes[0].get_yticklabels()]
    assert names == ["USB_P/USB_N", "USB again"]
    plot.close()


def test_the_pdf_report(app, solved, tmp_path):
    from kipi.si.html_report import write_si_report
    snap, setup, result, sc = solved
    out = write_si_report(tmp_path / "r.pdf", result, snap, setup, sc)
    assert out.stat().st_size > 10_000
    html = write_si_report(tmp_path / "r.html", result, snap, setup, sc)
    text = html.read_text(encoding="utf-8")
    assert "Signal Integrity report" in text and "download=" in text


def _two_results(result):
    import copy
    import dataclasses
    other = copy.copy(result)
    other.channel = dataclasses.replace(result.channel, name="USB again")
    return [result, other]


def test_one_report_of_several_channels(app, solved, tmp_path):
    from kipi.si.html_report import multi_report_name, write_si_reports
    snap, setup, result, sc = solved
    results = _two_results(result)
    html = write_si_reports(tmp_path / "r.html", results, snap, setup, sc)
    text = html.read_text(encoding="utf-8")
    assert "2 channels" in text and "id='si-pick'" in text
    assert "<option value='ch0'>USB_P/USB_N" in text and "<option value='ch1'>USB again" in text
    assert "id='ch0'" in text and "id='ch1'" in text and "id='ch1-checks'" in text
    assert text.count("Download the S-parameters") == 2 and text.count("id='scan'") == 1
    out = write_si_reports(tmp_path / "r.pdf", results, snap, setup, sc)
    assert out.stat().st_size > 10_000
    # A single channel gets the usual report.
    one = write_si_reports(tmp_path / "one.html", results[:1], snap, setup, sc)
    assert "si-pick" not in one.read_text(encoding="utf-8")
    assert multi_report_name("b", 3).startswith("b_3_channels_si_report_")


def test_the_panel_asks_for_a_report_of_the_ticked_channels(panel, solved, monkeypatch):
    from qtpy.QtCore import Qt
    from kipi.si.gui.report_dialog import ReportChannelsDialog
    _, _, result, _ = solved
    results = _two_results(result)
    panel.setup.si_channels = [r.channel for r in results]
    panel._fill_channels()
    assert not panel.btn_reports.isEnabled()
    for r in results:
        panel.show_result(r)
    assert panel.btn_reports.isEnabled()
    assert panel.solved_results() == results

    def pick_second(dlg):
        dlg.lst.item(0).setCheckState(Qt.Unchecked)
        return ReportChannelsDialog.Accepted
    monkeypatch.setattr(ReportChannelsDialog, "exec", pick_second)
    asked = []
    panel.reports_requested.connect(lambda rs, fmt: asked.append((rs, fmt)))
    panel.btn_reports.click()
    assert asked == [([results[1]], panel.report_option.fmt())]


def test_the_channels_dialog_needs_one_ticked(app, solved):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDialogButtonBox
    from kipi.si.gui.report_dialog import ReportChannelsDialog
    results = _two_results(solved[2])
    dlg = ReportChannelsDialog(results)
    ok = dlg.buttons.button(QDialogButtonBox.Ok)
    assert dlg.chosen() == results and ok.isEnabled()
    dlg._tick_all(Qt.Unchecked)
    assert dlg.chosen() == [] and not ok.isEnabled()
    dlg.close()


def test_results_round_trip(solved, tmp_path):
    from kipi.si import report
    snap, setup, result, sc = solved
    f = report.save_results(tmp_path / "x.kipi-si-results", {result.channel.name: result}, sc,
                            "b", setup)
    d = report.load_results(f)
    r = d["results"][result.channel.name]
    assert r.s.shape == result.s.shape and abs(r.eye.height_v - result.eye.height_v) < 1e-12
    assert r.eye.vbathtub.shape == result.eye.vbathtub.shape
    assert len(d["scan"].nets) == len(sc.nets)
    s4p = report.export_touchstone(tmp_path / "c", r)
    assert s4p.suffix == ".s4p" and "Port[4]" in s4p.read_text()


def test_the_3d_view_settings(panel, solved):
    """Settings' 3D view: the Quick Scan's and a result's impedance and the channel's TX
    and RX labels,
    each stored in the setup, redrawing the view, and loaded with the board."""
    from kipi.si.model import settings_from_dict
    _, _, _, sc = solved
    panel.show_scan(sc)
    redraws = []
    panel.overlay_changed.connect(lambda: redraws.append("overlay"))
    panel.focus_changed.connect(lambda: redraws.append("focus"))
    assert panel.chk_scan_3d.isChecked() and panel.overlay() is not None
    panel.chk_scan_3d.setChecked(False)
    assert not panel.setup.si.scan_3d and panel.overlay() is None and "overlay" in redraws

    ch = new_channel("CLK")
    ch.name = "CLK"
    panel.setup.si_channels.append(ch)
    panel.ch_list.reload()
    panel.tabs.setCurrentWidget(panel.ch_list)
    assert panel.ch_list.select_name("CLK")
    assert [t for t, _ in panel.end_labels()] == ["TX U1", "RX U2"]
    redraws.clear()
    panel.chk_end_labels.setChecked(False)
    assert not panel.setup.si.end_labels_3d and panel.end_labels() == [] and "focus" in redraws

    panel.chk_res_3d.setChecked(False)
    assert not panel.setup.si.result_3d

    import dataclasses
    back = settings_from_dict(dataclasses.asdict(panel.setup.si))
    assert (back.scan_3d, back.result_3d, back.end_labels_3d) == (False, False, False)
    setup = PISetup()
    setup.si = back
    panel.set_board(make_board(), setup)
    assert not any(c.isChecked() for c in (panel.chk_scan_3d, panel.chk_res_3d,
                                           panel.chk_end_labels))


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
