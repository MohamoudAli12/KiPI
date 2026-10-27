"""One rail's power tree filled from the board, the parts library and the parts' fields."""

from types import SimpleNamespace

import pytest

from kipi.core.model import snapshot_io
from kipi.core.model.board import Footprint, Net, Pad
from kipi.dc.pi.importers.rail_autofill import (
    AutofillError, MissingPart, PartsLibrary, autofill_rail, parse_pin_currents,
    parse_quantity, parse_tolerance, rating_checks, source_candidates, write_stubs,
)
from kipi.dc.pi.model import PERCENT, DcLink, Load, PinRef, PISetup, Vrm
from kipi.dc.solver.post import SeriesResult
from kipi.sources.kicad_pcb.reader import _fields
from kipi.sources.kicad_pcb.sexpr import parse
from synthetic import make_board

PARTS = {  # ref: (value, fields)
    "U1": ("LDO", {"MPN": "LDO-5"}),
    "FB1": ("120@100MHz", {"MPN": "BEAD-1"}),
    "R1": ("0.1", {}),
    "R2": ("10k", {}),
    "R3": ("1k", {}),
    "C1": ("10u", {}),
    "TP1": ("TP", {}),
    "U2": ("OPAMP", {"MPN": "OPA-1"}),
    "U3": ("PGA", {"MPN": "PGA-1", "PI_I": "3,4=230mA; 12=0.4mA"}),
    "U4": ("MYSTERY", {"MPN": "MYS-1"}),
    "U5": ("SPARE", {"PI_ROLE": "ignore"}),
    "U6": ("BIG", {"MPN": "OPA-1", "PI_I": "50mA", "PI_TOL": "±3%"}),
    "U9": ("OTHER", {}),
}
PINS = [  # (ref, pin, net)
    ("U1", "1", "+5V"), ("U1", "2", "+5V"), ("U1", "3", "GND"), ("U1", "9", "+5V3"),
    ("FB1", "1", "VA"), ("FB1", "2", "+5V"),
    ("R1", "1", "+5V"), ("R1", "2", "VB"),
    ("R2", "1", "+5V"), ("R2", "2", "SDA"),
    ("R3", "1", "+5V"), ("R3", "2", "GND"),
    ("C1", "1", "+5V"), ("C1", "2", "GND"),
    ("TP1", "1", "+5V"),
    ("U2", "5", "+5V"), ("U2", "2", "GND"),
    ("U3", "3", "VA"), ("U3", "4", "VA"), ("U3", "12", "+3V3"), ("U3", "5", "GND"),
    ("U4", "1", "VB"), ("U4", "2", "GND"),
    ("U5", "1", "+5V"), ("U5", "2", "GND"),
    ("U6", "5", "+5V"), ("U6", "2", "GND"),
    ("U9", "1", "+3V3"), ("U9", "2", "GND"),
]
LIBRARY = {
    "kipi_parts": 1,
    "parts": {
        "LDO-5": {"vrm": {"out_pins": ["1", "2"], "current_limit_a": 1.0}},
        "BEAD-1": {"link": {"resistance_ohm": 0.055, "current_max_a": 2.0}},
        "OPA-1": {"supplies": [{"name": "V+", "pins": ["5"], "current_a": 0.002,
                                "tolerance": {"min": -5, "max": 5, "unit": "%"}}]},
        "PGA-1": {"supplies": [{"name": "VA", "pins": ["3", "4"], "current_a": 0.1}]},
    },
}


def board():
    snap = make_board()
    snap.pads, snap.footprints = [], []
    names = sorted({n for _, _, n in PINS})
    snap.nets = [Net(0, "")] + [Net(i + 1, n) for i, n in enumerate(names)]
    for i, (ref, pin, net) in enumerate(PINS):
        snap.pads.append(Pad(f"p{i}", net, ref, pin, (float(i), 0.0), "smd"))
    for ref, (value, fields) in PARTS.items():
        ids = [p.id for p in snap.pads if p.footprint == ref]
        snap.footprints.append(Footprint(f"f{ref}", ref, value, (0.0, 0.0), "top", ids,
                                         fields=dict(fields)))
    snap.project_dir = ""
    snap.invalidate_index()
    return snap


def fill(snap=None, base=None, **kw):
    return autofill_rail(snap or board(), "+5V", "U1", base,
                         PartsLibrary.from_dict(LIBRARY), **kw)


def by_ref(items, side="supply"):
    return {getattr(i, side).ref: i for i in items}


# ---------------------------------------------------------------------------
# Values in fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,unit,expect", [
    ("230mA", "A", 0.23), ("0.23", "A", 0.23), ("2 A", "A", 2.0), ("63mW", "W", 0.063),
    ("1e-3", "A", 1e-3), ("400uA", "A", 4e-4), ("5V", "A", None), ("", "A", None),
    ("lots", "A", None)])
def test_parse_quantity(text, unit, expect):
    assert parse_quantity(text, unit) == pytest.approx(expect) if expect else \
        parse_quantity(text, unit) is None


def test_parse_pin_currents():
    total, groups = parse_pin_currents("3,4=230mA; 12=0.4mA")
    assert total is None
    assert [(g.pins, g.current) for g in groups] == [(["3", "4"], pytest.approx(0.23)),
                                                     (["12"], pytest.approx(4e-4))]
    assert parse_pin_currents("1.2A") == (pytest.approx(1.2), [])
    assert parse_pin_currents("3=lots") == (None, [])


@pytest.mark.parametrize("text,expect", [
    ("±5%", (-5.0, 5.0, PERCENT)), ("+/-2.5%", (-2.5, 2.5, PERCENT)),
    ("-3/+5%", (-3.0, 5.0, PERCENT)), ("±100mV", (-0.1, 0.1, "±V")), ("5", None)])
def test_parse_tolerance(text, expect):
    assert parse_tolerance(text) == expect


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def test_walk_finds_source_links_and_loads():
    r = fill()
    s = r.setup
    assert r.nets == ["+5V", "VA", "VB"]
    [vrm] = s.vrms
    assert (vrm.output, vrm.kind, vrm.voltage, vrm.current_limit_a) == (
        PinRef("U1", "+5V", ["1", "2"]), "source", 5.0, 1.0)
    links = by_ref(s.links, "side_a")
    assert set(links) == {"FB1", "R1"}
    assert links["FB1"].resistance == pytest.approx(0.055)          # from the library
    assert (links["FB1"].side_a.net, links["FB1"].side_b.net) == ("+5V", "VA")
    assert links["R1"].resistance == pytest.approx(0.1)             # from its value
    loads = by_ref(s.loads)
    # Not loads: the pull-up, the resistor to ground, the capacitor, the test point, the
    # ignored part, and a part on another rail.
    assert set(loads) == {"U2", "U3", "U4", "U6"}
    assert set(s.nets) == {"+5V", "VA", "VB"}
    assert any("R2" in n and "Not followed" in n for n in r.notes)


def test_currents_from_library_fields_and_missing():
    r = fill()
    loads = by_ref(r.setup.loads)
    assert loads["U2"].current == pytest.approx(0.002)              # library
    assert (loads["U2"].tol_min, loads["U2"].tol_max) == (-5, 5)
    # U3's PI_I by pins beats the library; only its pins on VA count.
    assert [(g.pins, g.current) for g in loads["U3"].groups] == [(["3", "4"],
                                                                   pytest.approx(0.23))]
    assert loads["U3"].total_current() == pytest.approx(0.23)
    # U6: a library part, but its own PI_I and PI_TOL win.
    assert loads["U6"].current == pytest.approx(0.05)
    assert (loads["U6"].tol_min, loads["U6"].tol_max) == (-3.0, 3.0)
    assert loads["U4"].current == 0.0
    assert [(m.key, m.kind, m.refs) for m in r.missing] == [("MYS-1", "load", ["U4"])]
    assert any(n.startswith("U4 on VB: no current") for n in r.notes)


def test_link_with_no_resistance_is_a_noted_short():
    snap = board()
    next(f for f in snap.footprints if f.reference == "FB1").fields = {"MPN": "BEAD-X"}
    r = fill(snap)
    assert by_ref(r.setup.links, "side_a")["FB1"].resistance == 0.0
    assert ("BEAD-X", "link") in [(m.key, m.kind) for m in r.missing]


def test_threshold_follows_bigger_resistors():
    r = fill(max_link_ohm=20e3)
    assert "R2" in by_ref(r.setup.links, "side_a")
    assert "SDA" in r.nets


def test_ground_pins_when_asked():
    r = fill(include_ground=True)
    assert r.setup.vrms[0].ground.net == "GND"
    assert all(l.ground is not None and l.ground.net == "GND" for l in r.setup.loads)
    assert "GND" in r.setup.nets
    assert all(l.ground is None for l in fill().setup.loads)


def test_merge_keeps_other_rails_and_replaces_this_one():
    base = PISetup(
        vrms=[Vrm("OLD", PinRef("U1", "+5V"), 5.0), Vrm("U9 3V3", PinRef("U9", "+3V3"), 3.3)],
        loads=[Load("old load", PinRef("U2", "VA"), 1.0), Load("U9 load", PinRef("U9", "+3V3"), 0.1)],
        links=[DcLink("old link", PinRef("FB1", "+5V"), PinRef("FB1", "VA"), 1.0)],
        nets=["+3V3"])
    base.thermal.enabled = True
    s = fill(base=base).setup
    assert [v.name for v in s.vrms] == ["U9 3V3", "U1 LDO"]
    assert "U9 load" in [l.name for l in s.loads] and "old load" not in [l.name for l in s.loads]
    assert "old link" not in [k.name for k in s.links]
    assert s.nets[0] == "+3V3" and s.thermal.enabled
    assert base.vrms[0].name == "OLD"                               # base not changed


def test_bad_rail_or_source():
    with pytest.raises(AutofillError, match="no net"):
        autofill_rail(board(), "NOPE", "U1", library=PartsLibrary())
    with pytest.raises(AutofillError, match="no pads on"):
        autofill_rail(board(), "+5V", "U9", library=PartsLibrary())


def test_source_candidates_put_the_regulator_first():
    lib = PartsLibrary.from_dict(LIBRARY)
    cands = source_candidates(board(), "+5V", lib)
    assert cands[0] == "U1"
    assert not {"C1", "R1", "R2", "FB1", "TP1"} & set(cands)


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------

def test_library_errors_and_lookup(tmp_path):
    with pytest.raises(AutofillError, match="kipi_parts"):
        PartsLibrary.from_dict({"parts": {}})
    with pytest.raises(AutofillError, match="role"):
        PartsLibrary.from_dict({"kipi_parts": 1, "parts": {"X": {"role": "boss"}}})
    with pytest.raises(AutofillError, match="number"):
        PartsLibrary.from_dict({"kipi_parts": 1, "parts": {"X": {"link": {"resistance_ohm": "a"}}}})
    user, proj = tmp_path / "user.yml", tmp_path / "proj.yml"
    user.write_text("kipi_parts: 1\nparts:\n  A: {link: {resistance_ohm: 1.0}}\n"
                    "  B: {link: {resistance_ohm: 2.0}}\n")
    proj.write_text("kipi_parts: 1\nparts:\n  B: {link: {resistance_ohm: 3.0}}\n")
    lib = PartsLibrary.load([user, proj, tmp_path / "missing.yml"])
    assert lib.files == [user, proj]
    assert lib.parts["A"].link.resistance_ohm == 1.0
    assert lib.parts["B"].link.resistance_ohm == 3.0               # the project's wins
    fp = Footprint("f", "R1", "B", (0, 0), "top", fields={"MANUFACTURER PART NUMBER": "A"})
    assert lib.key_of(fp) == "A" and lib.lookup(fp).link.resistance_ohm == 1.0
    fp.fields = {}
    assert lib.lookup(fp).link.resistance_ohm == 3.0                # by value


def test_write_stubs_new_append_and_reread(tmp_path):
    path = tmp_path / "pi_parts.yml"
    missing = [MissingPart("MYS-1", "load", ["U4"], ["1"], "VB"),
               MissingPart("BEAD-X", "link", ["FB1", "FB2"], ["2"], "+5V")]
    written, n = write_stubs(path, missing)
    assert (written, n) == (path, 2)
    lib = PartsLibrary.load([path])
    assert lib.parts["MYS-1"].supplies[0].pins == ["1"]
    assert lib.parts["MYS-1"].supplies[0].current_a is None
    assert lib.parts["BEAD-X"].link.resistance_ohm is None
    assert write_stubs(path, missing) == (path, 0)                  # nothing new
    path.write_text("# mine\nkipi_parts: 1\nparts:\n  OLD: {link: {resistance_ohm: 1.0}}\n")
    assert write_stubs(path, missing) == (path, 2)
    text = path.read_text()
    assert text.startswith("# mine") and set(PartsLibrary.load([path]).parts) == {
        "OLD", "MYS-1", "BEAD-X"}
    path.write_text("parts:\n  OLD: {link: {resistance_ohm: 1.0}}\nkipi_parts: 1\n")
    written, n = write_stubs(path, missing)
    assert written.name == "pi_parts.missing.yml" and n == 2


# ---------------------------------------------------------------------------
# Fields on the board, and ratings
# ---------------------------------------------------------------------------

def test_fields_read_from_a_kicad_pcb_footprint():
    fp = parse('(footprint "lib:R" (layer "F.Cu") (property "Reference" "R1") '
               '(property "Value" "0.1") (property "MPN" "ERJ-2") (property "PI_PMAX" "63mW"))')
    assert _fields(fp) == {"MPN": "ERJ-2", "PI_PMAX": "63mW"}


def test_snapshot_keeps_fields_and_reads_old_ones():
    snap = board()
    d = snapshot_io.to_dict(snap)
    again = snapshot_io.from_dict(d)
    assert next(f for f in again.footprints if f.reference == "U3").fields["PI_I"] == \
        "3,4=230mA; 12=0.4mA"
    for f in d["footprints"]:
        f.pop("fields")
    assert all(f.fields == {} for f in snapshot_io.from_dict(d).footprints)


def test_rating_checks():
    snap = board()
    r1 = next(f for f in snap.footprints if f.reference == "R1")
    r1.fields["PI_PMAX"] = "1mW"
    setup = fill(snap).setup
    series = [SeriesResult("FB1 120@100MHz", 0.055, 0.5, 5.0, 4.97, 0.01375),
              SeriesResult("R1 0.1", 0.1, 0.2, 5.0, 4.98, 0.004)]
    tree = SimpleNamespace(groups={"g": SimpleNamespace(result=SimpleNamespace(
        fields=SimpleNamespace(series=series)))})
    checks = dict(rating_checks(tree, snap, setup, PartsLibrary.from_dict(LIBRARY)))
    assert checks == {"FB1 120@100MHz: 0.5 A of 2 A (25 %)": True,
                      "R1 0.1: 4 mW of 1 mW (400 %)": False}
