# Portions Copyright (c) 2026 Tomachie LLC.
"""The analysis mode switch at the right end of the toolbar: DC by default, AC shows its
own (placeholder) setup panel and keeps the DC setup for when DC comes back."""

import os

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pyvistaqt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QLabel, QRadioButton, QToolBar  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, no_3d):
    from kipi.gui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.close()


def _mode(w, key) -> QRadioButton:
    return w.findChild(QRadioButton, f"mode_{key}")


def test_mode_switch_leads_the_toolbar_grouped_pi_and_si(window):
    tb = window.findChild(QToolBar)
    first = tb.widgetForAction(tb.actions()[0])
    assert first.objectName() == "mode_switch"
    lay = first.layout()
    order = [lay.itemAt(i).widget().text() for i in range(lay.count())
             if isinstance(lay.itemAt(i).widget(), (QLabel, QRadioButton))]
    assert order == ["Power Integrity", "DC", "AC", "Noise", "Signal Integrity", "SI"]


def test_power_tree_buttons_are_pi_only(window):
    bar = window.setup_bar
    for mode, shown in (("si", False), ("noise", True), ("si", False), ("dc", True)):
        _mode(window, mode).setChecked(True)
        assert bar.btn_autofill.isHidden() is not shown
        assert bar.btn_import.isHidden() is not shown
        assert not bar.btn_save.isHidden()


def test_dc_is_the_default(window):
    assert window.mode == "dc"
    assert _mode(window, "dc").isChecked() and not _mode(window, "ac").isChecked()
    assert window.setup_stack.currentWidget() is window.solve_panel
    assert window.solve_dock.windowTitle() == "PI setup"


def test_ac_swaps_the_setup_panel_and_dc_brings_it_back(window):
    setup = window.solve_panel.setup
    _mode(window, "ac").setChecked(True)
    assert window.mode == "ac" and not _mode(window, "dc").isChecked()
    assert window.setup_stack.currentWidget() is window.ac_panel
    assert window.solve_dock.windowTitle() == "AC setup"
    assert "AC" in window.act_open_results.toolTip()       # opens AC results here
    _mode(window, "dc").setChecked(True)
    assert window.mode == "dc"
    assert window.setup_stack.currentWidget() is window.solve_panel
    assert window.solve_panel.setup is setup          # the DC setup was kept
    assert "DC" in window.act_open_results.toolTip()


def test_mode_does_not_change_while_a_task_runs(window, monkeypatch):
    from qtpy.QtWidgets import QMessageBox
    monkeypatch.setattr(window, "_busy", lambda: True)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    _mode(window, "ac").setChecked(True)
    assert window.mode == "dc"
    assert _mode(window, "dc").isChecked() and not _mode(window, "ac").isChecked()
    assert window.setup_stack.currentWidget() is window.solve_panel


def test_mode_import_and_save_are_greyed_out_while_a_task_runs(app, window):
    import threading
    from tree_boards import board
    assert not window.setup_bar.isEnabled()                  # no board yet
    window.snapshot = board()
    release = threading.Event()
    task = window._run(lambda progress: release.wait(10), lambda _r: None, "Test task")
    try:
        assert not _mode(window, "ac").isEnabled() and not _mode(window, "dc").isEnabled()
        assert not window.setup_bar.isEnabled()
    finally:
        release.set()
        task.wait(5000)
        app.processEvents()
    assert _mode(window, "ac").isEnabled() and _mode(window, "dc").isEnabled()
    assert window.setup_bar.isEnabled()
    assert window.progress_bar.value() == 0          # idle shows an empty bar, not a full one


def _plot_tab_shown(w) -> bool:
    i = w.views.indexOf(w.plot)
    return w.views.isTabVisible(i) if hasattr(w.views, "isTabVisible") else w.views.isTabEnabled(i)


def test_impedance_plot_is_a_tab_beside_the_3d_view_in_ac_only(window):
    assert window.views.indexOf(window.viewer) == 0 and window.views.indexOf(window.plot) == 1
    assert not _plot_tab_shown(window)
    _mode(window, "ac").setChecked(True)
    assert _plot_tab_shown(window)
    window.views.setCurrentWidget(window.plot)
    _mode(window, "dc").setChecked(True)
    assert not _plot_tab_shown(window)
    assert window.views.currentWidget() is window.viewer


def test_si_mode_shows_its_panel_and_plots(window):
    _mode(window, "si").setChecked(True)
    assert window.mode == "si"
    assert window.setup_stack.currentWidget() is window.si_panel
    assert window.solve_dock.windowTitle() == "Signal Integrity setup"
    i = window.views.indexOf(window.si_plot)
    if hasattr(window.views, "isTabVisible"):
        assert window.views.isTabVisible(i)
        assert not window.views.isTabVisible(window.views.indexOf(window.noise_plot))
    _mode(window, "dc").setChecked(True)
    if hasattr(window.views, "isTabVisible"):
        assert not window.views.isTabVisible(i)


def test_setup_dock_fits_the_mode_shown(window, app):
    """The setup stack is as wide as the page it shows, not its widest one (DC's), so SI's
    Scan button is not scrolled out of a narrow dock; the dock widens to a wide page."""
    window._fitted = True                  # the offscreen screen is smaller than a window
    window.resize(1500, 950)
    window.show()
    area = window.setup_stack.parentWidget().parentWidget()
    for mode in ("si", "dc"):
        window.set_mode(mode)
        app.processEvents()
        page = window.setup_stack.currentWidget()
        assert window.setup_stack.minimumSizeHint().width() == page.minimumSizeHint().width()
        assert area.horizontalScrollBar().maximum() == 0, mode
