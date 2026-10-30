"""The setup window's theme and scale (kipi.launcher.appearance), without a display."""

from kipi.launcher import appearance
from kipi.launcher.appearance import DARK, LIGHT, saved_theme, ui_scale


def _no_xft():
    return None


def test_theme_follows_kipis_setting(tmp_path):
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    assert saved_theme(env, "linux", tmp_path) == DARK          # never chosen: KiPI's default
    conf = tmp_path / "KiPI" / "KiPI.conf"
    conf.parent.mkdir()
    conf.write_text('[about]\nstats="{\\"a\\": 1}"\n\n[appearance]\ntheme=light\n')
    assert saved_theme(env, "linux", tmp_path) == LIGHT
    conf.write_text("[appearance]\ntheme=purple\n")
    assert saved_theme(env, "linux", tmp_path) == DARK
    conf.write_text("not an ini file at all [")
    assert saved_theme(env, "linux", tmp_path) == DARK


def test_theme_from_the_macos_plist(tmp_path):
    import plistlib
    path = tmp_path / "Library" / "Preferences" / "com.kipi.KiPI.plist"
    path.parent.mkdir(parents=True)
    path.write_bytes(plistlib.dumps({"appearance.theme": "light"}, fmt=plistlib.FMT_BINARY))
    assert saved_theme({}, "darwin", tmp_path) == LIGHT


def test_scale_by_hand_wins_and_is_clamped():
    assert ui_scale(96, 2160, {"KIPI_UI_SCALE": "1.5"}, "linux", _no_xft) == 1.5
    assert ui_scale(96, 1080, {"KIPI_UI_SCALE": "9"}, "linux", _no_xft) == appearance.MAX_SCALE
    assert ui_scale(96, 2160, {"KIPI_UI_SCALE": "0.5"}, "linux", _no_xft) == 1.0
    assert ui_scale(96, 2160, {"KIPI_UI_SCALE": "big"}, "linux", _no_xft) == 2.0


def test_scale_from_the_desktop_on_linux():
    assert ui_scale(96, 2160, {}, "linux", lambda: 144.0) == 1.5          # Xft.dpi
    assert ui_scale(96, 2160, {"QT_FONT_DPI": "192"}, "linux", _no_xft) == 2.0
    assert ui_scale(96, 1080, {"QT_SCALE_FACTOR": "1.25"}, "linux", _no_xft) == 1.25
    assert ui_scale(96, 1080, {"GDK_SCALE": "2", "GDK_DPI_SCALE": "0.75"}, "linux",
                    _no_xft) == 1.5
    # QT_SCALE_FACTOR=1 says nothing; the next hint decides.
    assert ui_scale(96, 2160, {"QT_SCALE_FACTOR": "1", "GDK_SCALE": "2"}, "linux",
                    _no_xft) == 2.0


def test_scale_from_the_screen_when_nothing_says():
    assert ui_scale(96, 1080, {}, "linux", _no_xft) == 1.0
    assert ui_scale(96, 1440, {}, "linux", _no_xft) == 1.25
    assert ui_scale(96, 2160, {}, "linux", _no_xft) == 2.0
    assert ui_scale(96, 2880, {}, "linux", _no_xft) == 2.75


def test_scale_on_windows_and_macos():
    # Windows: Tk knows the DPI once the process is DPI aware; desktop hints are not read.
    assert ui_scale(144, 2160, {"QT_FONT_DPI": "288"}, "win32", _no_xft) == 1.5
    assert ui_scale(96, 2160, {}, "win32", _no_xft) == 1.0                 # 100% chosen
    assert ui_scale(72, 2880, {}, "darwin", _no_xft) == 1.0                # points already
