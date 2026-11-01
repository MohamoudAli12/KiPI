"""Differential pairs by their net names, and each pair's likely impedance target.

Two nets are a pair when their names differ only in a polarity marker: ``P``/``N`` as a
letter of its own between separators or at an end (``USB_P``/``USB_N``,
``TX0_P_C``/``TX0_N_C``), ``+``/``-`` (``/USB_D+``), or ``DP``/``DN``. The one with ``P``,
``+`` or ``DP`` is the positive leg.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_SWAPS = [
    (re.compile(r"(?<![A-Za-z])DP(?![A-Za-z])"), "DP", "DN"),
    (re.compile(r"(?<![A-Za-z])DN(?![A-Za-z])"), "DN", "DP"),
    (re.compile(r"(?<![A-Za-z])P(?![A-Za-z])"), "P", "N"),
    (re.compile(r"(?<![A-Za-z])N(?![A-Za-z])"), "N", "P"),
    (re.compile(r"\+"), "+", "-"),
    (re.compile(r"-"), "-", "+"),
]
_POSITIVE = ("DP", "P", "+")

# Typical differential impedance by interface, from a lower-case piece of the net name.
DIFF_TARGETS = (("usb4", 85.0), ("tbt", 85.0), ("thunderbolt", 85.0), ("usb", 90.0),
                ("pcie", 85.0), ("pci", 85.0), ("sata", 100.0), ("hdmi", 100.0),
                ("tmds", 100.0), ("dp_", 100.0), ("lvds", 100.0), ("mipi", 100.0),
                ("csi", 100.0), ("dsi", 100.0), ("eth", 100.0), ("sgmii", 100.0),
                ("xaui", 100.0), ("can", 120.0), ("rs485", 120.0),
                ("ddr", 80.0), ("dqs", 80.0), ("ck_", 80.0))
DEFAULT_DIFF_OHM = 100.0


def _partner(name: str) -> Optional[Tuple[str, bool]]:
    """The name with its last polarity marker swapped, and whether ``name`` is the
    positive leg; None when it has none."""
    best = None
    for rx, a, b in _SWAPS:
        for m in rx.finditer(name):
            if best is None or m.start() > best[0]:
                best = (m.start(), m.end(), a, b)
    if best is None:
        return None
    s, e, a, b = best
    return name[:s] + b + name[e:], a in _POSITIVE


def find_pairs(nets: Iterable[str]) -> List[Tuple[str, str]]:
    """(positive, negative) net pairs among ``nets``."""
    names = set(nets)
    out = []
    seen = set()
    for n in sorted(names):
        if n in seen:
            continue
        p = _partner(n)
        if p is None:
            continue
        other, positive = p
        if other in names and other not in seen:
            seen |= {n, other}
            out.append((n, other) if positive else (other, n))
    return out


# PCIe's lane names: the spec's PETp0/PERn1, and PET0_P, M2_PER3_N and the like.
_PCIE_LANE = re.compile(r"(?<![a-z])pe[tr][pn]?\d")


def by_name(net: str, table, pcie=None):
    """The value of the first of ``table``'s (piece, value) whose piece is in ``net``'s
    name (lower case), or ``pcie`` for a PCIe lane's name; None for none. The net's own
    name decides first, its sheet path only when that says nothing: on a sheet called
    "TS-USB4 Components", M2_PER0_P is a PCIe lane."""
    low = net.lower()
    own = low.rsplit("/", 1)[-1]
    for part in (own, low[:len(low) - len(own)]):
        for key, value in table:
            if key in part:
                return value
        if pcie is not None and _PCIE_LANE.search(part):
            return pcie
    return None


def diff_target(net: str) -> float:
    z = by_name(net, DIFF_TARGETS, 85.0)
    return DEFAULT_DIFF_OHM if z is None else z


Manual = Tuple[Sequence[Sequence[str]], Sequence[str]]     # (pairs, single-ended nets)


def pair_map(nets: Iterable[str], manual: Optional[Manual] = None) -> Dict[str, str]:
    """Each paired net's partner, both ways. ``manual``: the user's own pairs and
    single-ended nets, which win over the names: a net the user set either way leaves its
    pair by name, and the user's pairs are pairs whatever their names."""
    names = list(nets)
    pairs, single = manual or ((), ())
    pairs = [tuple(x) for x in pairs if len(x) == 2]
    forced = set(single) | {n for x in pairs for n in x}
    out = {}
    for p, n in find_pairs(names):
        if p not in forced and n not in forced:
            out[p], out[n] = n, p
    have = set(names)
    for p, n in pairs:
        if p in have and n in have:
            out[p], out[n] = n, p
    return out


def positive_first(a: str, b: str, manual: Optional[Manual] = None) -> Tuple[str, str]:
    """The pair ``a``, ``b`` as (positive, negative): as the user set it, else by the
    names' polarity markers, else by name."""
    for p, n in (manual or ((), ()))[0]:
        if {p, n} == {a, b}:
            return p, n
    x = _partner(a)
    if x is not None and x[0] == b:
        return (a, b) if x[1] else (b, a)
    return (a, b) if a < b else (b, a)


def manual_of(si) -> Manual:
    """The user's pairs and single-ended nets from the SI settings (none on an older
    setup)."""
    return (list(getattr(si, "manual_pairs", None) or []),
            list(getattr(si, "manual_single", None) or []))


def set_kind(si, nets: Sequence[str], kind: str):
    """Set ``nets`` as ``kind`` in the SI settings: "pair" (two nets, the first the
    positive leg), "single" (each single-ended) or "auto" (by name again). Whatever they
    were set as before is dropped first."""
    drop = set(nets)
    si.manual_pairs = [list(x) for x in getattr(si, "manual_pairs", None) or []
                       if not drop & set(x)]
    si.manual_single = [n for n in getattr(si, "manual_single", None) or [] if n not in drop]
    if kind == "pair":
        if len(nets) != 2:
            raise ValueError("a pair is two nets")
        si.manual_pairs.append(list(nets))
    elif kind == "single":
        si.manual_single.extend(nets)
