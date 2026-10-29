"""The Signal Integrity example board (examples/signal_integrity/make_board.py), for the
tests: ``make_board(stitch=True)``, and its layer ids."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "si_example", ROOT / "examples" / "signal_integrity" / "make_board.py")
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)

make_board = example.make_board
make_setup = example.make_setup
F, IN1, IN2, B, IDS = example.F, example.IN1, example.IN2, example.B, example.IDS
