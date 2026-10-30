"""Resistor values read from footprint value fields."""

import pytest

from kipi.core.model.board import Footprint
from kipi.core.model.parts import (
    parse_capacitance, parse_resistance, part_value, resistor_ohms, value_fields,
)
from synthetic import make_board


@pytest.mark.parametrize("text, ohms", [
    ("0.005", 0.005), ("5m", 0.005), ("5mR", 0.005), ("5 mΩ", 0.005), ("10 mohm", 0.01),
    ("10mOhms", 0.01), ("100", 100), ("100R", 100), ("100Ω", 100), ("4.7k", 4700),
    ("4k7", 4700), ("0R1", 0.1), ("R005", 0.005), ("1M", 1e6), ("1M5", 1.5e6), ("0", 0.0),
    ("10m 1% 0603", 0.01), ("0.01/1W", 0.01), ("2.2 kΩ", 2200),
])
def test_parses_resistor_values(text, ohms):
    assert parse_resistance(text) == pytest.approx(ohms)


@pytest.mark.parametrize("text", ["", "DNP", "10uF", "10mm", "1.5k2", "R", "BLM18"])
def test_rejects_non_resistances(text):
    assert parse_resistance(text) is None


def test_only_resistor_refs_are_read():
    snap = make_board()
    snap.footprints += [Footprint("f3", "R7", "10m", (0, 0), "top"),
                        Footprint("f4", "L1", "10m", (0, 0), "top"),
                        Footprint("f5", "RN1", "10m", (0, 0), "top")]
    assert resistor_ohms(snap, "R7") == pytest.approx(0.01)
    assert resistor_ohms(snap, "L1") is None
    assert resistor_ohms(snap, "RN1") is None
    assert resistor_ohms(snap, "R99") is None
    assert resistor_ohms(None, "R7") is None


def test_a_val_field_stands_in_for_an_unreadable_value():
    # A project keeping the library name in Value and the value itself in "Val".
    snap = make_board()
    snap.footprints += [Footprint("f6", "R8", "R_33R_0402", (0, 0), "top", fields={"Val": "33R"}),
                        Footprint("f7", "R9", "10m", (0, 0), "top", fields={"Val": "1k"}),
                        Footprint("f8", "R10", "R_0402", (0, 0), "top", fields={"MPN": "RC0402"})]
    assert resistor_ohms(snap, "R8") == pytest.approx(33)
    assert resistor_ohms(snap, "R9") == pytest.approx(0.01)     # Value first
    assert resistor_ohms(snap, "R10") is None


def test_part_value_picks_the_field_its_parser_reads():
    fp = Footprint("f", "C1", "C_220n_0402", (0, 0), "top", fields={"VAL": "220n"})
    assert part_value(fp, parse_capacitance) == "220n"
    assert value_fields(fp) == ["220n"]
    fp = Footprint("f", "C2", "C_0402", (0, 0), "top", fields={"Val": "DNP"})
    assert part_value(fp, parse_capacitance) == "C_0402"


def test_pdn_capacitors_take_their_value_from_a_val_field():
    from kipi.ac.checks import capacitor_candidates
    from kipi.core.model.board import Pad
    snap = make_board()
    snap.pads += [Pad(id="c1", net="+3V3", footprint="C110", number="1", position=(5, 0),
                      pad_type="smd"),
                  Pad(id="c2", net="GND", footprint="C110", number="2", position=(5, 1),
                      pad_type="smd")]
    snap.footprints.append(Footprint("f9", "C110", "C_220n_0402", (5, 0), "top",
                                     pad_ids=["c1", "c2"], fields={"Val": "220n"}))
    (c,) = capacitor_candidates(snap, ["+3V3", "GND"])
    assert (c.ref, c.value) == ("C110", "220n")
