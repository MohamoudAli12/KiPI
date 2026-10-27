"""The window opens at once: the 3D view is built once a stand-in for it has been drawn,
the AC and Coupled Noise panels and plots the first time they are wanted, and the plugin
connects to KiCad with the window already up."""

import os
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pyvistaqt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication  # noqa: E402

from phase_boards import board  # noqa: E402

pytestmark = pytest.mark.usefixtures("no_3d")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    from kipi.gui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.close()


def _until(app, done, timeout=10.0):
    t0 = time.time()
    while not done() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    return done()


def test_the_slow_parts_wait(window):
    assert window._viewer is None and window._ac_panel is None and window._plot is None
    assert window.views.tabText(0) == "3D view"
    assert window.views.count() == 1                  # the plots' tabs come with the plots
    assert set(window._panels()) == {"dc"}


def test_the_stand_in_readies_the_window_for_the_views_opengl(window):
    """Qt recreates a window on screen, which vanishes for a moment, when the first OpenGL
    widget is added to it; the stand-in is one, so the window opens with OpenGL."""
    from qtpy.QtOpenGLWidgets import QOpenGLWidget
    assert isinstance(window.views.widget(0), QOpenGLWidget)


def test_the_view_takes_its_stand_ins_place_once_drawn(app, window):
    ran = []
    window.show()
    window.when_ready(lambda: ran.append(window._viewer is not None))
    assert _until(app, lambda: ran)
    assert ran == [True]
    assert window.views.widget(0) is window.viewer
    assert window.views.tabText(0) == "3D view" and window.views.currentIndex() == 0


def test_not_showing_the_view_is_built_when_asked_for(window):
    ran = []
    window.when_ready(lambda: ran.append(window._viewer is not None))
    assert ran == [True]


def test_a_mode_switch_builds_its_panel(window):
    from qtpy.QtWidgets import QRadioButton
    window.findChild(QRadioButton, "mode_ac").setChecked(True)
    assert window._ac_panel is not None
    assert window.setup_stack.currentWidget() is window.ac_panel
    assert window.views.isTabVisible(window.views.indexOf(window.plot))
    assert not window.views.isTabVisible(window.views.indexOf(window.noise_plot))


def test_panels_built_late_get_the_board_setup_nets_and_dc_result(app, window, tmp_path):
    snap = board()
    snap.project_dir = str(tmp_path)
    window._set_snapshot(snap)
    assert _until(app, lambda: not window._busy() and not window._sync_timer.isActive(), 30)
    window.sync_nets()
    tree = object()
    window._set_tree(tree, {})
    assert window._ac_panel is None
    for panel in (window.ac_panel, window.noise_panel):
        assert panel.snapshot is window.snapshot
        assert panel.setup is window.solve_panel.setup
        assert set(panel._nets) == set(window._checked_set())
    assert window.ac_panel.dc_result is tree
    assert set(window._panels()) == {"dc", "ac", "noise"}


def test_a_source_given_later_is_read(window, monkeypatch):
    read = []
    monkeypatch.setattr(window, "reload_board", lambda: read.append(1))
    caps = SimpleNamespace(reload=True, refill_zones=False, select_items=False)
    source = SimpleNamespace(name="KiCad", description="KiCad", caps=caps, files=lambda: [])
    window.set_source(source)
    assert read == [1] and window.act_reload.isVisible() is not None


def test_failing_to_connect_says_why_and_closes(window, monkeypatch):
    from kipi.gui import app as gui_app
    shown, exits = [], []
    monkeypatch.setattr("qtpy.QtWidgets.QMessageBox.critical",
                        lambda parent, title, text: shown.append(text))
    fake_app = SimpleNamespace(exit=exits.append)

    def boom():
        raise RuntimeError("KiCad is not running")

    window.show()
    gui_app._connect(fake_app, window, boom)
    assert "KiCad is not running" in shown[0] and exits == [1]
    assert not window.isVisible()


def test_connecting_hands_the_window_its_source(window, monkeypatch):
    from kipi.gui import app as gui_app
    given = []
    monkeypatch.setattr(window, "set_source", lambda s, t: given.append((s, t)))
    gui_app._connect(SimpleNamespace(exit=None), window, lambda: ("src", "tool"))
    assert given == [("src", "tool")]
