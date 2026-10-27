"""PARDISO in the KiCad plugin's environment, and turning it off.

PARDISO (pypardiso and Intel MKL) is in ``requirements.txt`` where MKL is published
(Windows and Linux, x86-64), so KiCad installs it with the rest of the plugin's environment
(``pip install --no-input --only-binary :all: -r requirements.txt``). *About → Manage
dependencies…* turns it off or adds it back. KiCad puts it back on every plugin reload, so
turned off it is also left unused (KIPI_DIRECT_SOLVER=superlu).

The choice is kept in KiPI's QSettings, outside the environment, so it outlives the
environment. Removing PARDISO waits for the next start, before anything has loaded MKL
(Windows will not delete a library that is in use). Wherever any of this cannot work
(outside a virtual environment, no pip, no network), KiPI goes on with SciPy's solver.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional

from qtpy.QtCore import QEventLoop, QSettings, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QLabel, QMessageBox, QProgressDialog, QVBoxLayout,
)

from kipi.launcher import deps
from kipi.launcher.install import PARDISO_PACKAGES, Canceled, Installer, InstallError
from kipi.launcher.setup_env import PARDISO_CONS, PARDISO_PROS
from kipi.launcher.system import pardiso_supported

log = logging.getLogger(__name__)

CHOICE_KEY = "plugin/pardiso"            # "on" or "off"; unset until asked
REMOVE_KEY = "plugin/pardiso_remove"     # remove it at the next start
# What KiCad passes to pip for requirements.txt: wheels only, and never a question.
PIP_OPTIONS = ["--no-input", "--only-binary", ":all:"]


def _settings() -> QSettings:
    return QSettings("KiPI", "KiPI")


def _flag(value) -> bool:
    # QSettings may hand a bool back as "true" / "false" (an INI file, the registry).
    return value is True or str(value).lower() == "true"


def installed() -> bool:
    """Whether pypardiso is in the environment (found, not imported: MKL is slow to load)."""
    importlib.invalidate_caches()
    return importlib.util.find_spec("pypardiso") is not None


def in_venv() -> bool:
    """KiCad runs plugins in a virtual environment of their own; anywhere else (a system
    Python) KiPI installs nothing."""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def available() -> tuple:
    """Whether KiPI can add PARDISO here, and if not, why."""
    ok, why = pardiso_supported()
    if not ok:
        return False, why
    if not in_venv():
        return False, "KiPI is not running in a plugin environment of its own"
    return True, ""


def packages() -> List[deps.Package]:
    """What installing PARDISO adds: this platform's list saved with the release (sizes
    included), else just pypardiso by name."""
    plan = deps.manifest_plan(deps.platform_key())
    return (plan.pardiso if plan is not None else []) or [deps.Package(deps.PARDISO, "")]


def _installer(on_line: Callable[[str], None]) -> Installer:
    return Installer(None, Path(sys.prefix), on_line)


class _Worker(QThread):
    line = Signal(str)

    def __init__(self, work: Callable[[Installer], None]):
        super().__init__()
        self.installer = _installer(self.line.emit)
        self.work = work
        self.error: Optional[str] = None

    def run(self):
        try:
            self.work(self.installer)
        except Canceled:
            self.error = "canceled"
        except (InstallError, OSError) as e:
            self.error = str(e)


def _run_pip(parent, title: str, work: Callable[[Installer], None]) -> Optional[str]:
    """Run ``work`` on a thread behind a progress dialog; None when it worked, else why not."""
    worker = _Worker(work)
    dlg = QProgressDialog(title, "Cancel", 0, 0, parent)
    dlg.setWindowTitle("KiPI")
    dlg.setWindowModality(Qt.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setMinimumWidth(480)

    def on_line(text: str):
        log.info("pip: %s", text)
        shown = text if len(text) <= 90 else text[:87] + "…"
        dlg.setLabelText(f"{title}\n\n{shown}")

    worker.line.connect(on_line)
    dlg.canceled.connect(worker.installer.cancel)
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    worker.start()
    dlg.show()
    loop.exec()
    dlg.close()
    return worker.error


def install(parent=None) -> Optional[str]:
    """Install PARDISO into this environment; None when it worked, else why not."""
    names = [deps.PARDISO]
    error = _run_pip(parent, "Installing PARDISO (Intel MKL)…",
                     lambda inst: inst.install(packages=names, options=PIP_OPTIONS))
    if error is None:
        from kipi.core.solver import direct
        direct.retry_pardiso()          # use it from now on, without a restart
    return error


def uninstall(parent=None) -> Optional[str]:
    names = sorted({p.name for p in packages()} | set(PARDISO_PACKAGES))
    return _run_pip(parent, "Removing PARDISO…", lambda inst: inst.uninstall(names))


class PardisoDialog(QDialog):
    """What PARDISO is, what installing it adds, and a tick to install (or keep) it."""

    def __init__(self, parent=None, is_installed: bool = False, ticked: bool = True):
        super().__init__(parent)
        self.setWindowTitle("KiPI: optional faster solver")
        lay = QVBoxLayout(self)
        rows = packages()
        known = [p.size for p in rows if p.size]
        total = sum(known)
        if is_installed:
            intro = "PARDISO, Intel MKL's multithreaded solver, is installed in KiPI's " \
                    "plugin environment."
        else:
            intro = ("KiCad has installed the packages KiPI needs. KiPI can also install "
                     "PARDISO, Intel MKL's multithreaded solver, into its plugin environment. "
                     "It is optional, and KiPI works without it.")
        lines = [f"<p>{intro}</p>", "<p><b>Installing it adds:</b></p><ul>"]
        for p in rows:
            size = f", {deps.fmt_size(p.size)}" if p.size else ""
            lines.append(f"<li>{p.name} {p.version}{size}</li>")
        lines.append("</ul>")
        if total and len(known) == len(rows):
            lines.append(f"<p>{deps.fmt_size(total)} to download, about "
                         f"{deps.fmt_size(int(total * deps.INSTALLED_PER_DOWNLOAD))} on "
                         "disk.</p>")
        lines.append("<p>" + "<br>".join(f"+ {t}" for t in PARDISO_PROS)
                     + "<br>" + "<br>".join(f"− {t}" for t in PARDISO_CONS) + "</p>")
        lines.append("<p>Change this later under <i>About → Manage dependencies…</i></p>")
        text = QLabel("".join(lines), self)
        text.setTextFormat(Qt.RichText)
        text.setWordWrap(True)
        lay.addWidget(text)

        self.tick = QCheckBox("Keep PARDISO installed" if is_installed
                              else "Install PARDISO (recommended)", self)
        self.tick.setChecked(ticked)
        lay.addWidget(self.tick)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok, self)
        buttons.button(QDialogButtonBox.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        lay.addWidget(buttons)
        self.setMinimumWidth(560)

    def wanted(self) -> bool:
        return self.tick.isChecked()


def _remove_pending(parent) -> None:
    s = _settings()
    if not _flag(s.value(REMOVE_KEY, False)):
        return
    s.remove(REMOVE_KEY)
    if not installed():
        return
    error = uninstall(parent)
    if error is not None:
        log.warning("Removing PARDISO failed: %s", error)
        QMessageBox.warning(parent, "KiPI", f"PARDISO could not be removed: {error}\n\n"
                            "KiPI will not use it while it is turned off.")
    else:
        log.info("PARDISO removed")


def _off_unless_chosen() -> None:
    """While PARDISO is turned off but still there, keep KiPI from using it."""
    import os
    if _settings().value(CHOICE_KEY) == "off" and installed() \
            and not os.environ.get("KIPI_DIRECT_SOLVER"):
        os.environ["KIPI_DIRECT_SOLVER"] = "superlu"


def startup() -> None:
    """On the plugin's start, before anything loads MKL: finish a removal asked for last
    time, and leave PARDISO unused if it was turned off. Never raises; KiPI goes on with
    SciPy's solver whatever happens here."""
    try:
        _startup()
    except Exception:       # noqa: BLE001 - an optional solver must never stop KiPI
        log.exception("PARDISO setup failed")


def _startup() -> None:
    can, why = available()
    if not can:
        log.info("PARDISO not offered: %s", why)
        return
    _remove_pending(None)
    _off_unless_chosen()


def _apply(parent, want: bool, was_installed: bool, now: bool) -> None:
    """Act on a choice: install now, or remove at the next start (``now``: KiPI has not
    loaded MKL yet, so it can go at once)."""
    s = _settings()
    if want:
        s.setValue(CHOICE_KEY, "on")
        s.remove(REMOVE_KEY)
        if was_installed:
            return
        error = install(parent)
        if error is None:
            log.info("PARDISO installed")
            return
        log.warning("Installing PARDISO failed: %s", error)
        s.setValue(CHOICE_KEY, "off")    # don't ask again on every start
        if error != "canceled":
            QMessageBox.warning(
                parent, "KiPI", f"PARDISO could not be installed: {error}\n\nKiPI uses "
                "SciPy's solver instead. Try again under About → Manage dependencies…")
        return
    s.setValue(CHOICE_KEY, "off")
    if not was_installed:
        return
    if now:
        error = uninstall(parent)
        if error is not None:
            log.warning("Removing PARDISO failed: %s", error)
        return
    s.setValue(REMOVE_KEY, True)
    QMessageBox.information(parent, "KiPI", "PARDISO is removed the next time KiPI starts. "
                            "Until then KiPI keeps using it.")


def manage(parent=None) -> None:
    """About → Manage dependencies… in the plugin: add or remove PARDISO."""
    can, why = available()
    if not can:
        QMessageBox.information(parent, "KiPI", f"PARDISO is not available here: {why}. "
                                "KiPI uses SciPy's solver.")
        return
    is_installed = installed()
    removing = _flag(_settings().value(REMOVE_KEY, False))
    dlg = PardisoDialog(parent, is_installed=is_installed, ticked=not removing)
    if dlg.exec() != QDialog.Accepted:
        return
    want = dlg.wanted()
    if want and removing:
        _settings().remove(REMOVE_KEY)
        _settings().setValue(CHOICE_KEY, "on")
        return
    if want == is_installed and not removing:
        _settings().setValue(CHOICE_KEY, "on" if want else "off")
        return
    _apply(parent, want, was_installed=is_installed, now=False)
