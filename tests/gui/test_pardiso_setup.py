"""PARDISO comes with the plugin's requirements; turned off, it is removed and left unused."""

import os
import sys

import pytest

pytest.importorskip("qtpy")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QSettings  # noqa: E402
from qtpy.QtWidgets import QApplication, QDialog, QLabel  # noqa: E402

from kipi.gui import pardiso_setup as ps  # noqa: E402
from kipi.launcher.install import Installer  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = QSettings(str(tmp_path / "kipi.ini"), QSettings.IniFormat)
    monkeypatch.setattr(ps, "_settings", lambda: s)
    return s


@pytest.fixture
def here(monkeypatch):
    """A computer that can have PARDISO, without it installed; nothing really runs pip."""
    monkeypatch.setenv("KIPI_DIRECT_SOLVER", "")    # an opt-out sets it; undone after
    monkeypatch.setattr(ps, "available", lambda: (True, ""))
    monkeypatch.setattr(ps, "installed", lambda: False)
    calls = []
    monkeypatch.setattr(ps, "install", lambda parent=None: calls.append("install"))
    monkeypatch.setattr(ps, "uninstall", lambda parent=None: calls.append("uninstall"))
    return calls


def test_not_offered_outside_a_virtual_environment(monkeypatch):
    monkeypatch.setattr(ps, "pardiso_supported", lambda: (True, ""))
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    ok, why = ps.available()
    assert not ok and "environment" in why


def test_dialog_lists_what_it_installs_and_is_ticked(app):
    d = ps.PardisoDialog()
    text = " ".join(lb.text() for lb in d.findChildren(QLabel))
    for p in ps.packages():
        assert p.name in text
    assert d.wanted()


def _answer(monkeypatch, accept: bool, tick: bool):
    def exec_(self):
        self.tick.setChecked(tick)
        return QDialog.Accepted if accept else QDialog.Rejected
    monkeypatch.setattr(ps.PardisoDialog, "exec", exec_)


def test_start_asks_nothing(app, settings, here, monkeypatch):
    """KiCad installs PARDISO with the requirements; the start no longer offers it."""
    monkeypatch.setattr(ps.PardisoDialog, "exec", lambda self: pytest.fail("asked"))
    ps.startup()
    assert here == [] and settings.value(ps.CHOICE_KEY) is None


def test_requirements_install_it_where_mkl_is_published():
    from pathlib import Path
    from packaging.requirements import Requirement
    lines = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text().splitlines()
    req = next(Requirement(x) for x in lines if x.startswith("pypardiso"))
    env = lambda system, machine: {"sys_platform": system, "platform_machine": machine}  # noqa: E731
    assert req.marker.evaluate(env("win32", "AMD64"))
    assert req.marker.evaluate(env("linux", "x86_64"))
    assert not req.marker.evaluate(env("darwin", "arm64"))
    assert not req.marker.evaluate(env("darwin", "x86_64"))
    assert not req.marker.evaluate(env("linux", "aarch64"))


def test_removal_waits_for_the_next_start(app, settings, here, monkeypatch):
    monkeypatch.setattr(ps, "installed", lambda: True)
    monkeypatch.setattr(ps.QMessageBox, "information", lambda *a, **k: None)
    _answer(monkeypatch, accept=True, tick=False)
    ps.manage()
    assert here == [] and ps._flag(settings.value(ps.REMOVE_KEY))
    ps.startup()
    assert here == ["uninstall"] and settings.value(ps.REMOVE_KEY) is None


def test_turned_off_but_still_there_is_not_used(app, settings, here, monkeypatch):
    monkeypatch.setattr(ps, "installed", lambda: True)
    settings.setValue(ps.CHOICE_KEY, "off")
    ps.startup()
    assert os.environ.get("KIPI_DIRECT_SOLVER") == "superlu"


def test_startup_never_raises(settings, monkeypatch):
    def boom():
        raise RuntimeError("no")
    monkeypatch.setattr(ps, "available", boom)
    ps.startup()


def test_installer_passes_pip_options_first(tmp_path):
    seen = []
    inst = Installer(None, tmp_path, lambda line: None)
    inst._run = seen.append
    inst.install(packages=["pypardiso"], options=ps.PIP_OPTIONS)
    cmd = seen[0]
    assert cmd[-4:] == ["--no-input", "--only-binary", ":all:", "pypardiso"]
