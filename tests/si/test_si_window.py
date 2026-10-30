"""SI mode in the main window: a Quick Scan, a channel's solve and its report, run as
the window runs them (the work done in place here, not on a thread)."""

import os
from pathlib import Path

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("matplotlib")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QRadioButton  # noqa: E402

from helpers.si_boards import make_board, make_setup  # noqa: E402
from kipi.core.progress import NULL  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_scan_solve_and_report(app, no_3d, tmp_path, opened_in_browser):
    from kipi.gui.main_window import MainWindow
    snap, setup = make_board(), make_setup()
    w = MainWindow()
    w._run = lambda fn, done, name, **k: done(fn(NULL))      # no background thread here
    try:
        w.snapshot, w._setup_path = snap, tmp_path / "board.kipi-setup.yml"
        w.solve_panel.set_board(snap, setup)
        w.findChild(QRadioButton, "mode_si").setChecked(True)
        panel = w.si_panel
        panel.set_board(snap, setup)
        panel.btn_scan.click()
        assert panel.scan_result is not None and panel.tbl_scan.rowCount() >= 2
        panel.cmb_channel.setCurrentIndex(panel.cmb_channel.findData("USB_P/USB_N"))
        assert panel.btn_solve.isEnabled()
        panel.btn_solve.click()
        r = panel.result
        assert r is not None and r.channel.name == "USB_P/USB_N"
        assert w.si_plot.result is r
        assert w.views.currentWidget() is w.si_plot
        panel.report_option.btn.click()
        written, = tmp_path.glob("*_si_report_*.html")
        assert r.report_path == written or Path(r.report_path) == written
        assert [Path(p) for p in opened_in_browser] == [written]
    finally:
        w.close()


def test_nets_list_follows_the_mode(app, no_3d, tmp_path):
    from qtpy.QtCore import Qt
    from kipi.gui.main_window import MainWindow
    snap, setup = make_board(), make_setup()
    w = MainWindow()
    w._run = lambda fn, done, name, **k: None          # no meshing (the 3D view is off)
    try:
        w.snapshot = snap
        w.solve_panel.set_board(snap, setup)
        setup.nets, setup.si_nets = ["GND"], None
        w._populate_nets(setup.nets, setup)

        def shown():
            return [it.data(Qt.UserRole) for it in w._net_items() if not it.isHidden()]

        def ticked():
            return set(w._checked_nets())
        assert "USB_P" not in shown() and ticked() == {"GND"}        # PI: power and ground
        w.findChild(QRadioButton, "mode_si").setChecked(True)
        assert w.chk_power_only.text() == "Signal nets only"
        rows = shown()
        assert rows[:2] in (["USB_N", "USB_P"], ["USB_P", "USB_N"])  # pairs first
        assert "CLK" in rows
        # SI's own ticks: the channels' nets (and the main ground), not PI's.
        assert {"USB_P", "USB_N", "CLK"} <= ticked()
        assert setup.nets == ["GND"]
        next(it for it in w._net_items() if it.data(Qt.UserRole) == "CLK").setCheckState(
            Qt.Unchecked)
        w.findChild(QRadioButton, "mode_dc").setChecked(True)
        assert ticked() == {"GND"} and "CLK" not in setup.si_nets
        assert w.chk_power_only.text() == "Power && ground only"
        w.findChild(QRadioButton, "mode_si").setChecked(True)
        assert "CLK" not in ticked() and "USB_P" in ticked()
    finally:
        w.close()


def test_selected_nets_are_picked_out(app, no_3d, tmp_path):
    from kipi.gui.main_window import MainWindow
    snap, setup = make_board(), make_setup()
    w = MainWindow()
    w._run = lambda fn, done, name, **k: None
    shown = []
    w.viewer.set_focus_nets = lambda nets: shown.append(set(nets or ()) or None)
    focus = lambda: shown[-1]                                        # noqa: E731
    try:
        w.snapshot = snap
        w.solve_panel.set_board(snap, setup)
        w.findChild(QRadioButton, "mode_si").setChecked(True)
        panel = w.si_panel
        panel.set_board(snap, setup)
        panel.tabs.setCurrentWidget(panel.ch_list)
        panel.ch_list.select_name("CLK")
        assert focus() == {"CLK", "CLK_R"}     # through its series resistor
        it = next(w.net_list.item(i) for i in range(w.net_list.count())
                  if w.net_list.item(i).data(0x0100) == "USB_P")
        it.setSelected(True)                               # the Nets list's pick wins
        assert focus() == {"USB_P"}
        w.net_list.clearSelection()
        panel.tabs.setCurrentWidget(panel.tabs.widget(3))  # Settings: no net in focus
        assert focus() is None
    finally:
        w.close()


def test_solve_the_ticked_channels(app, no_3d, tmp_path):
    """Ticked channels are solved together, each one's status on its row; with none
    ticked, Solve solves the channel picked."""
    from qtpy.QtCore import Qt
    from kipi.gui.main_window import MainWindow
    snap, setup = make_board(), make_setup()
    w = MainWindow()
    asked = []

    def run(fn, done, name, on_partial=None):
        out = fn(NULL)
        for item in out:
            on_partial(item)
        done(out)
    w._run = run
    try:
        w.snapshot, w._setup_path = snap, tmp_path / "board.kipi-setup.yml"
        w.solve_panel.set_board(snap, setup)
        w.findChild(QRadioButton, "mode_si").setChecked(True)
        panel = w.si_panel
        panel.set_board(snap, setup)
        panel.solve_requested.connect(lambda _s, names: asked.append(list(names)))
        names = [c.name for c in setup.si_channels]
        assert len(names) >= 2 and panel.btn_solve.text() == "Solve"
        lst = panel.ch_list.lst
        for i in range(len(names)):
            lst.item(i).setCheckState(Qt.Checked)
        assert all(c.ticked for c in setup.si_channels)
        assert panel.btn_solve.text() == f"Solve {len(names)} ticked"
        assert panel.ch_list.chk_all.isChecked()
        panel.btn_solve.click()
        assert asked[-1] == names
        assert set(panel.results()) == set(names)
        for i, ch in enumerate(setup.si_channels):
            assert lst.item(i).text().endswith(panel.results()[ch.name].status)
        panel.ch_list.chk_all.click()                       # untick all
        assert not any(c.ticked for c in setup.si_channels)
        assert panel.btn_solve.text() == "Solve"
        panel.btn_solve.click()
        assert asked[-1] == [panel.cmb_channel.currentData()]
    finally:
        w.close()


def test_reopen_saved_results(app, no_3d, tmp_path, monkeypatch):
    """Save results… then Open results… in the panel brings the solved channel and the
    Quick Scan back, shown as after the solve, without solving again."""
    from qtpy.QtWidgets import QFileDialog
    from kipi.gui.main_window import MainWindow
    snap, setup = make_board(), make_setup()
    w = MainWindow()
    w._run = lambda fn, done, name, **k: done(fn(NULL))
    saved = tmp_path / "board.kipi-si-results"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(saved), ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(saved), ""))
    try:
        w.snapshot, w._setup_path = snap, tmp_path / "board.kipi-setup.yml"
        w.solve_panel.set_board(snap, setup)
        w.findChild(QRadioButton, "mode_si").setChecked(True)
        panel = w.si_panel
        panel.set_board(snap, setup)
        panel.btn_scan.click()
        panel.cmb_channel.setCurrentIndex(panel.cmb_channel.findData("USB_P/USB_N"))
        panel.btn_solve.click()
        solved = panel.result
        panel.btn_save.click()
        assert saved.is_file()

        panel.set_board(snap, setup)                      # forget the solve
        assert panel.result is None and not panel.results()
        assert panel.btn_open.isEnabled()
        panel.btn_open.click()
        r = panel.results()["USB_P/USB_N"]
        assert r.status == solved.status and r.s.shape == solved.s.shape
        assert panel.result is r and w.si_plot.result is r
        assert panel.scan_result is not None
        panel.set_busy(True)
        assert not panel.btn_open.isEnabled()
    finally:
        w.close()


def test_picking_another_channel_keeps_the_view(app, no_3d, tmp_path):
    """A solve brings its plots to the front; picking another solved channel then leaves
    the view in front (3D or the plots) and the panel's tab as they are."""
    from qtpy.QtCore import Qt
    from kipi.gui.main_window import MainWindow
    snap, setup = make_board(), make_setup()
    w = MainWindow()

    def run(fn, done, name, on_partial=None):
        out = fn(NULL)
        for item in out:
            on_partial(item)
        done(out)
    w._run = run
    try:
        w.snapshot, w._setup_path = snap, tmp_path / "board.kipi-setup.yml"
        w.solve_panel.set_board(snap, setup)
        w.findChild(QRadioButton, "mode_si").setChecked(True)
        panel = w.si_panel
        panel.set_board(snap, setup)
        names = [c.name for c in setup.si_channels]
        for i in range(len(names)):
            panel.ch_list.lst.item(i).setCheckState(Qt.Checked)
        panel.btn_solve.click()
        assert set(panel.results()) == set(names)
        assert w.views.currentWidget() is w.si_plot

        for view in (w.viewer, w.si_plot):
            w.views.setCurrentWidget(view)
            panel.tabs.setCurrentWidget(panel.ch_list)
            for name in names:
                panel.cmb_channel.setCurrentIndex(panel.cmb_channel.findData(name))
                assert panel.result is panel.results()[name]
                assert w.si_plot.result is panel.result
                assert w.views.currentWidget() is view
                assert panel.tabs.currentWidget() is panel.ch_list
    finally:
        w.close()


def test_si_nets_fold_into_sections(app, no_3d, tmp_path):
    from qtpy.QtCore import Qt
    from kipi.gui.main_window import MainWindow
    snap, setup = make_board(), make_setup()
    w = MainWindow()
    w._run = lambda fn, done, name, **k: None
    try:
        w.snapshot = snap
        w.solve_panel.set_board(snap, setup)
        w._nets_folded = set()
        w.findChild(QRadioButton, "mode_si").setChecked(True)
        w.chk_power_only.setChecked(False)

        def headers():
            return {w.net_list.item(i).data(Qt.UserRole + 3): w.net_list.item(i)
                    for i in range(w.net_list.count())
                    if w.net_list.item(i).data(Qt.UserRole) is None}

        def shown():
            return {it.data(Qt.UserRole) for it in w._net_items() if not it.isHidden()}
        h = headers()
        assert set(h) == {"pairs", "single", "planes"}
        assert h["pairs"].text().startswith("▾ Differential pairs")
        assert not h["pairs"].flags() & Qt.ItemIsUserCheckable
        assert {"USB_P", "USB_N", "CLK"} <= shown()
        # A header's click folds its section, and ticks and the 3D view skip headers.
        w._net_header_clicked(h["pairs"])
        assert "USB_P" not in shown() and "CLK" in shown()
        assert h["pairs"].text().startswith("▸ Differential pairs")
        assert None not in w._checked_nets()
        w._net_header_clicked(h["pairs"])
        assert "USB_P" in shown()
        # Signal nets only: the planes' section keeps its ticked nets (the main ground),
        # and goes, header and all, when none is ticked.
        w.chk_power_only.setChecked(True)
        planes = [it for it in w._net_items() if it.data(Qt.UserRole + 3) == "planes"]
        assert all(it.isHidden() == (it.checkState() != Qt.Checked) for it in planes)
        w._set_checked(lambda it: it.data(Qt.UserRole + 1) == "signal")
        assert headers()["planes"].isHidden()
    finally:
        w.close()


def test_right_click_sets_a_net_single_ended_or_paired(app, no_3d, tmp_path):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QMenu
    from kipi.gui.main_window import MainWindow
    snap, setup = make_board(), make_setup()
    w = MainWindow()
    w._run = lambda fn, done, name, **k: None
    try:
        w.snapshot = snap
        w.solve_panel.set_board(snap, setup)
        w._nets_folded = set()
        w.findChild(QRadioButton, "mode_si").setChecked(True)

        def section(net):
            return next(it.data(Qt.UserRole + 3) for it in w._net_items()
                        if it.data(Qt.UserRole) == net)

        def row(net):
            return next(it for it in w._net_items() if it.data(Qt.UserRole) == net)
        assert section("USB_P") == "pairs" and section("CLK") == "single"
        menu = QMenu()
        w._net_kind_menu(menu, [row("USB_P")])
        acts = {a.text(): a for a in menu.actions()}
        assert not acts["Make a differential pair (select its two nets)"].isEnabled()
        acts["Make single-ended"].trigger()
        # The pair is broken: both legs single-ended, the one set tagged so.
        assert section("USB_P") == "single" and section("USB_N") == "single"
        assert "[set]" in row("USB_P").text()
        assert setup.si.manual_single == ["USB_P"]
        ticked = set(w._checked_nets())
        menu = QMenu()
        w._net_kind_menu(menu, [row("CLK"), row("USB_N")])
        acts = {a.text(): a for a in menu.actions()}
        acts["Make a differential pair (CLK +, USB_N −)"].trigger()
        assert section("CLK") == "pairs" and setup.si.manual_pairs == [["CLK", "USB_N"]]
        assert set(w._checked_nets()) == ticked                     # ticks kept
        menu = QMenu()
        w._net_kind_menu(menu, [row("USB_P"), row("CLK"), row("USB_N")])
        next(a for a in menu.actions() if a.text().startswith("Back to")).trigger()
        assert section("USB_P") == "pairs" and section("CLK") == "single"
        assert setup.si.manual_pairs == [] and setup.si.manual_single == []
    finally:
        w.close()
