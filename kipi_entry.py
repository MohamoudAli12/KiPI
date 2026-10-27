"""KiCad IPC plugin action entry point (see plugin.json).

KiCad starts this in the plugin's own venv with KICAD_API_SOCKET / KICAD_API_TOKEN
set, so kipy connects to the instance that launched us.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    # Windows: on to KiPI.exe, so Task Manager shows KiPI rather than Python (#124).
    from kipi.launcher.named_exe import relaunch
    code = relaunch([os.path.abspath(__file__), *sys.argv[1:]])
    if code is not None:
        return code
    from kipi.gui import app as kipi_app
    from kipi.gui.app import run, setup_logging

    setup_logging(console=False)
    # KiCad captures plugin stdout/stderr through a pipe it does not always drain; a full
    # pipe blocks the writer forever. Send everything to this session's log folder.
    kipi_app.detach_std_streams()
    os.environ.setdefault("QT_API", "pyside6")
    from kipi.host import Host, set_current
    set_current(Host.PLUGIN)
    from kipi.gui import single_instance
    instance = single_instance.claim()
    if instance is None:        # another KiPI is open; it was brought to the front
        return 0
    try:
        def connect_kicad():
            # Before anything loads MKL: finish a PARDISO removal asked for last time (#129).
            from kipi.gui import pardiso_setup
            pardiso_setup.startup()
            from kipi.core.progress import ProgressReporter
            from kipi.sources.kicad_ipc.extract import connect
            from kipi.sources.kicad_ipc.source import InstalledKiCad, KiCadLiveSource
            kicad, board = connect(progress=ProgressReporter())
            source = KiCadLiveSource(kicad, board)
            return source, InstalledKiCad(source)

        # The window first, so the click in KiCad is answered at once (#55); KiPI connects
        # to KiCad once it is showing.
        return run(instance=instance, connect=connect_kicad)
    finally:
        instance.release()


if __name__ == "__main__":
    sys.exit(main())
