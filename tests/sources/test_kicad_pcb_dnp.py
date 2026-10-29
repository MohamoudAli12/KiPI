"""Do-not-populate and KiCad 10's assembly variants, read from a ``.kicad_pcb`` file: the
syntax is KiCad 10's own writer's (pcb_io_kicad_sexpr.cpp, formatVariants and the
footprint's variants), which kicad-cli 10 reads back and applies."""

import dataclasses

from helpers.si_boards import make_board
from kipi.core.model.board import Footprint
from kipi.core.model.snapshot_io import from_dict, to_dict
from kipi.sources.kicad_pcb.reader import _dnp, _variant_dnp, read_variants
from kipi.sources.kicad_pcb.sexpr import parse

BOARD = """(kicad_pcb (version 20250000)
  (variants (variant (name "USB4") (description "USB4 enclosure")) (variant (name "PCIe")))
  (footprint "C_0402" (attr smd)
    (property "Reference" "C28")
    (variant (name "USB4") (dnp yes)))
  (footprint "C_0402" (attr smd dnp)
    (property "Reference" "C30")
    (variant (name "USB4") (dnp no) (exclude_from_bom no)))
  (footprint "R_0402" (attr smd exclude_from_bom)
    (property "Reference" "R1")))"""


def _fps():
    root = parse(BOARD)
    fps = [c for c in root if type(c) is list and c[0] == "footprint"]
    return root, {next(p[2] for p in fp if type(p) is list and p[0] == "property"): fp
                  for fp in fps}


def test_dnp_and_each_variants_own():
    root, fps = _fps()
    assert read_variants(root) == {"USB4": "USB4 enclosure", "PCIe": ""}
    assert [_dnp(fps[r]) for r in ("C28", "C30", "R1")] == [False, True, False]
    assert _variant_dnp(fps["C28"]) == {"USB4": True}
    assert _variant_dnp(fps["C30"]) == {"USB4": False}
    assert _variant_dnp(fps["R1"]) == {}


def test_a_variant_falls_back_to_the_parts_own_dnp():
    c28 = Footprint("1", "C28", "", (0, 0), "top", variant_dnp={"USB4": True})
    c30 = Footprint("2", "C30", "", (0, 0), "top", dnp=True, variant_dnp={"USB4": False})
    assert [c28.is_dnp(), c28.is_dnp("USB4"), c28.is_dnp("PCIe")] == [False, True, False]
    assert [c30.is_dnp(), c30.is_dnp("USB4"), c30.is_dnp("PCIe")] == [True, False, True]


def test_snapshots_keep_them():
    snap = dataclasses.replace(make_board(), variants={"USB4": "enclosure"})
    snap.footprints[0].dnp = True
    snap.footprints[1].variant_dnp = {"USB4": True}
    back = from_dict(to_dict(snap))
    assert back.variants == {"USB4": "enclosure"}
    assert back.dnp_refs() == snap.dnp_refs() and back.dnp_refs("USB4") == snap.dnp_refs("USB4")
    old = to_dict(snap)                     # a snapshot saved before they were read
    old.pop("variants")
    for f in old["footprints"]:
        f.pop("dnp"), f.pop("variant_dnp")
    assert from_dict(old).dnp_refs() == [] and from_dict(old).variants == {}
