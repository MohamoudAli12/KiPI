"""The Signal Integrity example's checked-in files are what its make_board.py builds, and
its setup's channels solve."""

import json

from helpers.si_boards import example, make_board, make_setup
from kipi.core.model.snapshot_io import load, to_dict
from kipi.dc.pi.model import PISetup


def test_files_match_the_generator():
    snap = load(example.HERE / f"{example.NAME}.kipi.json.gz")
    saved, built = (json.loads(json.dumps(to_dict(s))) for s in (snap, make_board()))
    saved.pop("source"), built.pop("source")
    assert saved == built
    setup = PISetup.load(example.HERE / f"{example.NAME}.kipi-setup.yml")
    assert [c.name for c in setup.si_channels] == [c.name for c in make_setup().si_channels]
    assert setup.si_channels[1].protocol == "usb2"
