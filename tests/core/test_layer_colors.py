"""Copper layer colors follow the PCB editor's color theme, else KiCad's default theme."""

import json

import pytest

from kipi.sources.kicad_ipc import colors
from kipi.core.model.board import CopperLayer
from kipi.core.viz import layer_colors


@pytest.fixture
def kicad_home(tmp_path, monkeypatch):
    """A KiCad settings folder with one folder per version, and nothing else to find:
    3rd-party themes only under ``tmp_path / "third"``, never this machine's."""
    monkeypatch.setattr(colors, "_config_roots", lambda: [tmp_path])
    monkeypatch.setattr(colors, "_third_party_dirs", lambda version: [tmp_path / "third"])

    def version(name, theme=None, files=None):
        folder = tmp_path / name
        (folder / "colors").mkdir(parents=True)
        settings = {} if theme is None else {"appearance": {"color_theme": theme}}
        (folder / "pcbnew.json").write_text(json.dumps(settings))
        for fname, text in (files or {}).items():
            (folder / "colors" / fname).write_text(text)
        return folder

    return version


def _theme(name, copper):
    return json.dumps({"meta": {"name": name}, "board": {"copper": copper}})


def test_layers_map_to_theme_keys():
    assert layer_colors.layer_key(3, 0, 4) == "f"
    assert layer_colors.layer_key(34, 3, 4) == "b"
    assert layer_colors.layer_key(4, 1, 4) == "in1"
    assert layer_colors.layer_key(6, 2, 4) == "in3"        # In3.Cu even without In2.Cu
    assert layer_colors.layer_key(99, 2, 4) == "in2"       # unknown id: by place
    assert layer_colors.layer_key(0, 0, 1) == "f"


def test_default_theme_is_kicads():
    d = layer_colors.DEFAULT_COPPER
    assert len(d) == 32
    assert d["f"] == (200, 52, 52) and d["b"] == (77, 127, 196)
    assert d["in1"] == (127, 200, 127) and d["in13"] == d["in5"] and d["in30"] == d["in8"]


def test_colors_parse():
    assert colors.parse_color("rgb(1, 2, 3)") == (1, 2, 3)
    assert colors.parse_color("rgba(10, 20, 30, 0.800)") == (10, 20, 30)
    assert colors.parse_color("red") is None


def test_the_editors_theme_is_used(kicad_home):
    kicad_home("10.0", "Mine", {"Mine.json": _theme("Mine", {"f": "rgb(1, 2, 3)",
                                                             "in2": "rgba(4, 5, 6, 1)",
                                                             "b": "nonsense"})})
    found, source = colors.copper_theme("10.0.1")
    assert found["f"] == (1, 2, 3) and found["in2"] == (4, 5, 6)
    assert found["b"] == colors.DEFAULT_COPPER["b"]  # unreadable: default
    assert found["in1"] == colors.DEFAULT_COPPER["in1"]  # left out: default
    assert "Mine" in source


def test_a_theme_is_found_by_its_display_name(kicad_home):
    kicad_home("10.0", "Pretty", {"pretty_v2.json": _theme("Pretty", {"f": "rgb(9, 9, 9)"})})
    assert colors.copper_theme()[0]["f"] == (9, 9, 9)


def test_a_theme_installed_from_the_content_manager_is_found_by_its_path(
        kicad_home, tmp_path):
    # KiCad names such a theme by its file, under 3rdparty/colors, not by a name.
    f = tmp_path / "3rdparty" / "colors" / "com_example_mono" / "mono.json"
    f.parent.mkdir(parents=True)
    f.write_text(_theme("Mono High Contrast", {"f": "rgb(245, 127, 0)"}))
    kicad_home("10.0", str(f), {})
    found, source = colors.copper_theme("10.0")
    assert found["f"] == (245, 127, 0)
    assert source == "KiCad theme 'Mono High Contrast'"


def test_a_content_manager_theme_whose_path_is_gone_is_found_again(kicad_home, tmp_path):
    # Settings brought over from KiCad 9 still point into 9.0/3rdparty; the theme now
    # lives under this version's 3rd-party folder.
    f = tmp_path / "third" / "colors" / "com_example_mono" / "mono.json"
    f.parent.mkdir(parents=True)
    f.write_text(_theme("Mono", {"f": "rgb(1, 1, 1)"}))
    folder = kicad_home("10.0", "/old/kicad/9.0/3rdparty/colors/com_example_mono/mono.json")
    assert colors.copper_theme("10.0")[0]["f"] == (1, 1, 1)
    # By display name too: a 3rd-party theme KiCad names that way.
    (folder / "pcbnew.json").write_text(json.dumps({"appearance": {"color_theme": "Mono"}}))
    assert colors.copper_theme("10.0")[0]["f"] == (1, 1, 1)


def test_the_3rd_party_folder_follows_kicads_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("KICAD10_3RD_PARTY", str(tmp_path))
    assert colors._third_party_dirs("10.0")[0] == tmp_path


def test_the_running_versions_settings_win(kicad_home):
    kicad_home("9.0", "Nine", {"Nine.json": _theme("Nine", {"f": "rgb(9, 0, 0)"})})
    kicad_home("10.0", "Ten", {"Ten.json": _theme("Ten", {"f": "rgb(10, 0, 0)"})})
    assert colors.copper_theme("9.0.4")[0]["f"] == (9, 0, 0)
    assert colors.copper_theme("")[0]["f"] == (10, 0, 0)      # else the newest
    assert colors.copper_theme("11.0.0")[0]["f"] == (10, 0, 0)


@pytest.mark.parametrize("theme, files", [
    ("", {}),                                                 # built-in default
    (None, {}),                                               # never chosen
    ("_builtin_classic", {}),
    ("Gone", {}),                                             # theme file missing
    ("Bad", {"Bad.json": "{not json"}),
])
def test_anything_unreadable_gives_the_default_theme(kicad_home, theme, files):
    kicad_home("10.0", theme, files)
    found, source = colors.copper_theme("10.0")
    assert found == colors.DEFAULT_COPPER
    assert "default" in source


def test_no_kicad_settings_gives_the_default_theme(kicad_home):
    found, source = colors.copper_theme("10.0")
    assert found == colors.DEFAULT_COPPER and "no KiCad settings" in source


def test_meshes_use_the_colors_given():
    from kipi.core.viz import meshes
    top = CopperLayer(id=3, name="F.Cu", index=0, thickness_mm=0.035, z_bottom_mm=1.6)
    inner = CopperLayer(id=5, name="In2.Cu", index=1, thickness_mm=0.035, z_bottom_mm=1.0)
    try:
        meshes.set_layer_colors({"f": (255, 0, 0)})
        assert meshes.layer_color(top, 4) == (1.0, 0.0, 0.0)
        assert meshes.layer_color(inner, 4) == tuple(
            v / 255 for v in colors.DEFAULT_COPPER["in2"])
    finally:
        meshes.set_layer_colors(colors.DEFAULT_COPPER)
