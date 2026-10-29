"""Buses: the nets between a transmitting part and a receiving part, as channels.

A memory interface, an ADC's data lanes or a parallel bus is many nets from one part to
another. :func:`bus_nets` finds every signal net that joins the two parts (directly, or
through series resistors and AC-coupling capacitors), pairs the differential ones (DQS,
CK) by name and groups the rest by name (DQ0–DQ7, A0–A15). :func:`make_channels` turns
the chosen ones into channels with both ends set and each net's neighbours in its group
as its aggressors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from kipi.core.model.board import BoardSnapshot

_INDEX = re.compile(r"^(.*?)(\d+)([_\-]?[PN]|[+\-]|_?[PN])?$", re.IGNORECASE)
_LEG = re.compile(r"([_\-]?[PN]|[+\-]|_(DP|DM|T|C))$", re.IGNORECASE)


@dataclass
class BusNet:
    net: str                    # at the transmitting part (a pair's positive leg)
    partner: str = ""           # a pair's negative leg
    group: str = ""             # "DQ", "A", "DQS", … (the name without its index)
    index: Optional[int] = None
    series: Tuple[str, ...] = ()    # nets beyond series parts, to the receiver

    @property
    def name(self) -> str:
        return f"{self.net}/{self.partner}" if self.partner else self.net


def parts_on(snap: BoardSnapshot, nets: Sequence[str]) -> List[Tuple[str, int]]:
    """The parts with pads on ``nets`` other than 2-pin series parts: (reference, how many
    of their pads are on them), ICs first, then by that count."""
    from kipi.si.extract import series_parts
    skip = {r for r, *_ in series_parts(snap, list(nets))}
    count: Dict[str, int] = {}
    for n in nets:
        for p in snap.items_on_net(n)[2]:
            if p.footprint and p.footprint not in skip:
                count[p.footprint] = count.get(p.footprint, 0) + 1
    return sorted(count.items(), key=lambda kv: (not kv[0].upper().startswith("U"),
                                                 -kv[1], kv[0]))


class _Index:
    """The board's pads by net and by part, and the series parts' chains between nets
    (:func:`kipi.si.extract.follow_series`'s rule), found once for a whole bus."""

    def __init__(self, snap: BoardSnapshot, not_fitted: Sequence[str] = ()):
        from kipi.core.model.nets import is_ground_name, is_power_name
        from kipi.si.scan import signal_nets
        self.signals = set(signal_nets(snap))
        self.by_net: Dict[str, set] = {}
        by_fp: Dict[str, list] = {}
        for p in snap.pads:
            if p.footprint and p.net:
                self.by_net.setdefault(p.net, set()).add(p.footprint)
                by_fp.setdefault(p.footprint, []).append(p)
        self.pads_of = by_fp
        # A 2-pin R, C, L or ferrite between two signal nets is a series part.
        self.series: Dict[str, List[Tuple[str, str]]] = {}
        self.series_refs = set()
        for fp, pads in by_fp.items():
            nets = {p.net for p in pads}
            if len(pads) != 2 or len(nets) != 2 or not re.match(r"(R|C|L|FB)\d", fp) \
                    or fp in not_fitted:
                continue
            a, b = sorted(nets)
            if any(is_ground_name(n) or is_power_name(n) for n in (a, b)):
                continue
            self.series_refs.add(fp)
            self.series.setdefault(a, []).append((fp, b))
            self.series.setdefault(b, []).append((fp, a))

    def chain(self, net: str, limit: int = 4) -> List[str]:
        out, frontier = [net], [net]
        while frontier and len(out) < limit:
            for _, other in self.series.get(frontier.pop(), []):
                if other not in out:
                    out.append(other)
                    frontier.append(other)
        return out

    def parts(self, nets) -> set:
        return set().union(*(self.by_net.get(n, set()) for n in nets)) - self.series_refs

    def nets_of(self, ref: str) -> List[str]:
        return sorted({p.net for p in self.pads_of.get(ref, []) if p.net in self.signals})


def connected_parts(snap: BoardSnapshot, ref: str, index: Optional[_Index] = None
                    ) -> List[Tuple[str, int]]:
    """The parts joined to part ``ref`` by signal nets (through series parts too), with
    how many nets join them: the candidates to receive what ``ref`` sends."""
    ix = index or _Index(snap)
    count: Dict[str, int] = {}
    for net in ix.nets_of(ref):
        for other in ix.parts(ix.chain(net)) - {ref}:
            count[other] = count.get(other, 0) + 1
    return sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))


def bus_nets(snap: BoardSnapshot, tx: str, rx: str, index: Optional[_Index] = None
             ) -> List[BusNet]:
    """Every signal net from part ``tx`` to part ``rx``: paired by name where both legs
    go between them, grouped and numbered by name, in group then index order."""
    from kipi.si.pairs import pair_map
    ix = index or _Index(snap)
    found: Dict[str, Tuple[str, ...]] = {}
    for net in ix.nets_of(tx):
        chain = ix.chain(net)
        if rx in ix.parts(chain):
            found[net] = tuple(chain[1:])
    pairs = pair_map(list(found))
    out, used = [], set()
    for net in found:
        if net in used:
            continue
        partner = pairs.get(net, "")
        if partner and partner in found:
            pos, neg = _order_legs(net, partner)
            used |= {pos, neg}
            group, index_ = _group(pos, True)
            out.append(BusNet(pos, neg, group, index_, found[pos]))
        else:
            used.add(net)
            group, index_ = _group(net, False)
            out.append(BusNet(net, "", group, index_, found[net]))
    _rank_unnumbered(out)
    out.sort(key=lambda b: (not b.partner, b.group.lower(),
                            -1 if b.index is None else b.index, b.net))
    return out


def _natural(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _rank_unnumbered(bus: List[BusNet]):
    """Names with no number at their end ("D1A", "D1B", …): grouped by their leading
    letters ("D") and numbered in natural order, so lanes named alike still neighbour."""
    by: Dict[Tuple[bool, str], List[BusNet]] = {}
    for b in bus:
        if b.index is not None:
            continue
        m = re.match(r"[A-Za-z_]+", b.group)
        if m and m.group(0) != b.group and re.search(r"\d", b.group):
            by.setdefault((bool(b.partner), m.group(0).rstrip("_")), []).append(b)
    for (_, prefix), items in by.items():
        if len(items) < 2:
            continue
        for k, b in enumerate(sorted(items, key=lambda b: _natural(b.group))):
            b.group, b.index = prefix, k


def _order_legs(a: str, b: str) -> Tuple[str, str]:
    """(positive, negative) of a pair's two legs, by name."""
    neg = re.compile(r"([_\-]N|N$|-$|_DM$|_C$)", re.IGNORECASE)
    return (b, a) if neg.search(a.rsplit("/", 1)[-1]) and not neg.search(
        b.rsplit("/", 1)[-1]) else (a, b)


def _group(net: str, pair: bool) -> Tuple[str, Optional[int]]:
    """A net's group name and index: "/DDR/DQ3" -> ("DQ", 3), "DQS0_P" -> ("DQS", 0)."""
    name = net.rsplit("/", 1)[-1]
    if pair:
        name = _LEG.sub("", name)
    m = _INDEX.match(name)
    if m and m.group(1):
        return m.group(1).rstrip("_-[") or name, int(m.group(2))
    return name, None


def neighbours(bus: Sequence[BusNet], item: BusNet, n: int = 1) -> List[str]:
    """The nets of the ``n`` items on each side of ``item`` in its group (by index): its
    likeliest aggressors, routed beside it as a bus usually is."""
    same = [b for b in bus if b.group == item.group and b is not item
            and b.index is not None and item.index is not None]
    near = sorted(same, key=lambda b: abs(b.index - item.index))
    near = [b for b in near if abs(b.index - item.index) <= n]
    return [x for b in near for x in (b.net, b.partner) if x]


def make_channels(snap: BoardSnapshot, bus: Sequence[BusNet], tx: str, rx: str,
                  protocol: Optional[str] = None, aggressors: int = 1,
                  taken: Sequence[str] = ()) -> List:
    """Channels for ``bus``'s nets from ``tx`` to ``rx``: interface ``protocol`` (None:
    guessed from each name), the ``aggressors`` nearest in its group on each side as its
    aggressors. Channels named in ``taken`` are left out."""
    from kipi.si.model import new_channel
    out = []
    for b in bus:
        if b.name in taken:
            continue
        ch = new_channel(b.net, b.partner, protocol)
        ch.name = b.name
        ch.driver_ref, ch.receiver_ref = tx, rx
        ch.aggressors = neighbours(bus, b, aggressors) if aggressors else []
        out.append(ch)
    return out
