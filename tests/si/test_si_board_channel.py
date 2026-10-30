"""Channels extracted from a board: routes, coupled sections, vias and series parts, and the
Quick Scan's checks."""

import math

import numpy as np
import pytest

from helpers.si_boards import make_board
from kipi.si import closed_form as cf
from kipi.si import sparams
from kipi.si.board import SIBoard
from kipi.si.extract import (
    BuildOptions, build_channel, follow_series, parse_value, part_value, series_parts,
    via_stack,
)
from kipi.si.pairs import diff_target, find_pairs
from kipi.si.scan import FAIL, PASS, WARN, ScanSettings, scan


@pytest.fixture(scope="module")
def board():
    return SIBoard(make_board())


def test_pairs_by_name():
    assert find_pairs(["USB_P", "USB_N", "/D+", "/D-", "TX0_P_C", "TX0_N_C", "RX_DP",
                       "RX_DN", "CANH", "SPI_MOSI"]) == [
        ("/D+", "/D-"), ("RX_DP", "RX_DN"), ("TX0_P_C", "TX0_N_C"), ("USB_P", "USB_N")]
    assert diff_target("USB_P") == 90 and diff_target("PCIE_TX0_P") == 85
    assert diff_target("FOO_P") == 100
    # The net's own name first, its sheet's only when that says nothing; PCIe's lane names.
    assert diff_target("/USB/D_P") == 90
    assert diff_target("/TS-USB4 Components/M2_PER0_P") == 85
    assert diff_target("/Hub/USB_DP") == 90 and diff_target("/X/PETp0") == 85


def test_an_interface_is_guessed_from_the_nets_own_name():
    from kipi.si.model import new_channel
    guess = {n: new_channel(n, n.replace("_P", "_N")).protocol for n in (
        "/TS-USB4 Components/M2_PER0_P", "/TS-USB4 Components/M2_PET3_P",
        "/TS-PCIe Components/PCIe_PER0_P", "/USB/D_P", "/Dock/USB4_TX1_P", "/SATA_A_P",
        "/FPGA/USRIO_1_P")}
    assert guess == {"/TS-USB4 Components/M2_PER0_P": "pcie3",
                     "/TS-USB4 Components/M2_PET3_P": "pcie3",
                     "/TS-PCIe Components/PCIe_PER0_P": "pcie3", "/USB/D_P": "usb2",
                     "/Dock/USB4_TX1_P": "usb4g3", "/SATA_A_P": "sata3",
                     "/FPGA/USRIO_1_P": "lvds"}


def test_matched_generic_buffers_take_the_interfaces_pad_capacitance():
    from kipi.si.model import new_channel
    serdes, cmos = new_channel("PCIE_TX0_P", "PCIE_TX0_N"), new_channel("CLK")
    assert serdes.driver.c_pf == serdes.receiver.c_pf == pytest.approx(0.5)
    assert cmos.driver.c_pf == cmos.receiver.c_pf == pytest.approx(3.0)


def test_values():
    assert parse_value("100n") == pytest.approx(1e-7)
    assert parse_value("4k7") == pytest.approx(4700)
    assert parse_value("0R1") == pytest.approx(0.1)
    assert parse_value("33") == 33
    assert parse_value("10uF") == pytest.approx(1e-5)
    assert parse_value("DNP") is None


def test_part_value_falls_back_to_a_val_field():
    # A project keeping the library name in Value and the value itself in "Val".
    from types import SimpleNamespace as NS
    assert part_value(NS(value="C_220n_0402", fields={"Val": "220n"})) == "220n"
    assert part_value(NS(value="100n", fields={"Val": "1u"})) == "100n"
    assert part_value(NS(value="C_220n_0402", fields={"VALUE": "220nF"})) == "220nF"
    assert part_value(NS(value="C_220n_0402", fields={"MPN": "GRM155"})) == "C_220n_0402"


def test_series_parts(board):
    snap = board.snap
    assert follow_series(snap, "CLK") == ["CLK", "CLK_R"]
    assert series_parts(snap, ["CLK", "CLK_R"]) == [("R1", "R", "33", "CLK|CLK_R")]


def test_not_fitted_parts_are_not_followed(board):
    # An assembly variant leaves R1 off: the channel stops at its empty pads, which stay
    # copper (a stub), with nothing joining them.
    snap = board.snap
    assert follow_series(snap, "CLK", not_fitted=["R1"]) == ["CLK"]
    assert series_parts(snap, ["CLK", "CLK_R"], not_fitted=["R1"]) == []
    ch = build_channel(board, [["CLK"]], BuildOptions(not_fitted=("R1",)))
    assert ch.series == []
    assert not [b for b in ch.circuit.blocks if b.label == "R1"]


def test_single_ended_channel(board):
    ch = build_channel(board, [["CLK"], ["CLK_R"]], BuildOptions())
    assert ch.series == ["R1"]
    zs = {round(z, 1) for s in ch.sections for z in s.z0}
    assert len(zs) == 1
    # Hammerstad–Jensen gives 50.4 Ω uncoated; the 15 µm mask pulls it down a little.
    z0 = zs.pop()
    assert 0.92 * cf.microstrip(0.36, 0.2, 4.2, 0.035)[0] < z0 < cf.microstrip(0.36, 0.2, 4.2, 0.035)[0]
    c = ch.circuit
    c.add_port("u1", ("pad", "u1"))
    c.add_port("u2", ("pad", "u2"))
    f = sparams.uniform_grid(20e9, 40e6)
    s = c.solve_s(f)
    # The 33 Ω resistor in a 50 Ω system: |S21| = 100/133 at low frequency.
    assert abs(s[0, 1, 0]) == pytest.approx(100 / 133, rel=0.01)
    t, z = sparams.tdr(f, s[:, 0, 0], 50.0, 40e-12)
    td = 15.0 * math.sqrt(3.0) / 299.79e0 * 1e-12 * 1e3        # rough: εeff ≈ 3
    early = (t > 20e-12) & (t < 0.8 * 2 * td * 1e0)
    assert np.median(z[early]) == pytest.approx(z0, rel=0.03)
    assert z.max() > 70                                            # the resistor


def test_a_pair_keeps_its_zdiff_beside_an_aggressor():
    """Sections with an aggressor in them still give the pair's Zdiff (the aggressor held
    at 0 V), and a section of P with the aggressor alone gives none."""
    import dataclasses
    from kipi.core.model.board import Net, Track
    from helpers.si_boards import F
    snap = make_board()
    snap = dataclasses.replace(
        snap, nets=snap.nets + [Net(len(snap.nets), "AGG")],
        tracks=snap.tracks + [Track("AGG", F, 0.2, (8.0, 14.6), (25.0, 14.6))])
    b = SIBoard(snap)
    ch = build_channel(b, [["USB_P", "USB_N", "AGG"]], BuildOptions(),
                       pairs=[("USB_P", "USB_N")])
    three = [s for s in ch.sections if set(s.nets) == {"USB_P", "USB_N", "AGG"}]
    assert three and all(s.zdiff is not None and 85 < s.zdiff < 100 for s in three)
    assert all(s.zdiff is None for s in ch.sections if "USB_N" not in s.nets)
    # A single-ended victim with an aggressor: no Zdiff anywhere.
    ch = build_channel(b, [["USB_P", "AGG"]], BuildOptions(), pairs=[])
    assert all(s.zdiff is None for s in ch.sections)


def test_a_bend_is_a_few_sections():
    """A 90° arc is cut into 30° chords (each a field solve), not the 3D view's 5.6°."""
    import dataclasses
    from kipi.core.model.board import Track
    from kipi.si.extract import net_segments
    from helpers.si_boards import F
    snap = make_board()
    r = 0.5
    snap = dataclasses.replace(snap, tracks=[Track(
        "CLK", F, 0.2, (10.0, 20.0), (10.0 + r, 20.0 + r),
        (10.0 + r - r * math.cos(math.pi / 4), 20.0 + r * math.sin(math.pi / 4)))])
    segs = net_segments(snap, "CLK", 1.0)
    assert len(segs) == 3
    arc = math.pi / 2 * r
    assert sum(s.length_mm for s in segs) == pytest.approx(arc, rel=0.012)


def test_a_pair_stays_coupled_round_a_bend():
    """Round a bend the outer leg's chords are longer than the inner's: they still pair,
    coupled over the inner's length with the outer going on alone for the rest, so each
    leg keeps its own length."""
    import dataclasses
    from kipi.core.model.board import Track
    from helpers.si_boards import F
    c45 = math.cos(math.pi / 4)

    def leg(net, r, y):
        return [Track(net, F, 0.2, (5.0, y), (20.0, y)),
                Track(net, F, 0.2, (20.0, y), (20.0 + r, 16.0),
                      (20.0 + r * c45, 16.0 - r * c45)),
                Track(net, F, 0.2, (20.0 + r, 16.0), (20.0 + r, 30.0))]
    snap = dataclasses.replace(make_board(), tracks=leg("USB_P", 1.0, 15.0) +
                               leg("USB_N", 0.65, 15.35))
    ch = build_channel(SIBoard(snap), [["USB_P", "USB_N"]], BuildOptions())
    assert not [s for s in ch.sections if s.nets == ["USB_P"]]      # P never alone
    bend = [s for s in ch.sections if len(s.segs) == 2 and
            s.segs[0].length_mm > 1.2 * s.segs[1].length_mm]
    assert bend and all(s.zdiff is not None for s in bend)
    assert sum(1 for b in ch.circuit.blocks if b.label.endswith("(alone)")) >= len(bend)
    arc_p, arc_n = math.pi / 2 * 1.0, math.pi / 2 * 0.65
    assert ch.net_length("USB_P") == pytest.approx(15 + 14 + arc_p, rel=0.002)
    assert ch.net_length("USB_N") == pytest.approx(15 + 14 + arc_n, rel=0.002)


def test_offset_legs_pair_all_along():
    """The legs are cut into segments from their own ends, so one leg's cuts fall in the
    middle of the other's segments: the second leg is cut where the first's are, and none
    of either runs alone."""
    import dataclasses
    from kipi.core.model.board import Track
    from helpers.si_boards import F
    snap = dataclasses.replace(make_board(), tracks=[
        Track("USB_P", F, 0.2, (5.0, 15.0), (20.0, 15.0)),
        Track("USB_N", F, 0.2, (5.4, 15.35), (20.3, 15.35))])
    ch = build_channel(SIBoard(snap), [["USB_P", "USB_N"]], BuildOptions())
    assert not [s for s in ch.sections if len(s.segs) == 1]
    assert all(s.zdiff is not None for s in ch.sections)
    assert ch.net_length("USB_P") == pytest.approx(15.0)
    assert ch.net_length("USB_N") == pytest.approx(14.9)


def test_a_tight_bend_with_the_reference_inside_stays_coupled():
    """Round a tight bend with P on the inside, P's chords are a fraction of N's: N's
    chords still pair with them (the rest of each is its tail), as does P's chord beside
    the end of N's straight."""
    import dataclasses
    from kipi.core.model.board import Track
    from helpers.si_boards import F
    c45 = math.cos(math.pi / 4)

    def leg(net, r, y):
        return [Track(net, F, 0.127, (5.0, y), (20.0, y)),
                Track(net, F, 0.127, (20.0, y), (20.0 + r, 15.5),
                      (20.0 + r * c45, 15.5 - r * c45)),
                Track(net, F, 0.127, (20.0 + r, 15.5), (20.0 + r, 25.0))]
    snap = dataclasses.replace(make_board(), tracks=leg("USB_P", 0.08, 15.42) +
                               leg("USB_N", 0.335, 15.165))
    ch = build_channel(SIBoard(snap), [["USB_P", "USB_N"]], BuildOptions())
    assert not [s for s in ch.sections if len(s.segs) == 1]
    arc_p, arc_n = math.pi / 2 * 0.08, math.pi / 2 * 0.335
    assert ch.net_length("USB_P") == pytest.approx(15 + arc_p + 9.5, rel=0.002)
    assert ch.net_length("USB_N") == pytest.approx(15 + arc_n + 9.5, rel=0.002)


def test_pair_channel(board):
    ch = build_channel(board, [["USB_P", "USB_N"]], BuildOptions())
    zd = [s.zdiff for s in ch.sections if s.zdiff]
    assert zd and 85 < np.median(zd) < 100
    assert set(ch.vias) == {"vp", "vn"}
    assert ch.net_length("USB_N") - ch.net_length("USB_P") == pytest.approx(math.sqrt(2) - 1,
                                                                            abs=1e-6)
    c = ch.circuit
    for k in ("jp", "jn", "u3p", "u3n"):
        c.add_port(k, ("pad", k))
    f = np.array([1e8, 1e9, 5e9])
    mm = sparams.mixed_mode(c.solve_s(f), [(0, 1), (2, 3)])
    sdd21 = sparams.db(mm[:, 1, 0])
    assert sdd21[0] > -0.2 and sdd21[0] > sdd21[1] > sdd21[2]
    # Scd21: little conversion low down; the legs' vias 1 mm apart convert more by 5 GHz.
    assert sparams.db(mm[:2, 3, 0]).max() < -20 and sparams.db(mm[2, 3, 0]) < -10


def test_via_sees_stitched_planes(board):
    vs = via_stack(board, board.snap.vias[0])
    assert [l.is_plane for l in vs.layers] == [False, True, True, False]
    assert vs.layers[1].antipad_r_mm == pytest.approx(0.45, abs=0.02)
    z = vs.cavity[(1, 2)](np.array([1e9]))
    assert 0 < z.imag[0] < 10                                      # a few hundred pH
    open_board = SIBoard(make_board(stitch=False))
    vs2 = via_stack(open_board, open_board.snap.vias[0])
    # Unstitched, the return crosses as displacement current: the plane pair's
    # capacitance at low frequency, where the stitched return is an inductance.
    assert vs2.cavity[(1, 2)](np.array([1e7]))[0].imag < 0
    assert vs.cavity[(1, 2)](np.array([1e7]))[0].imag > 0


def test_quick_scan(board):
    r = scan(board.snap, settings=ScanSettings(), board=board)
    by = r.by_net()
    assert set(by) == {"CLK", "CLK_R", "USB_P", "USB_N"}
    clk = by["CLK"]
    assert clk.status == PASS and clk.target == 50
    assert by["USB_P"].partner == "USB_N" and by["USB_P"].target == 90
    assert 85 < by["USB_P"].zdiff_mean < 100
    skew = [f for f in by["USB_P"].findings if f.check == "skew"][0]
    assert skew.status == WARN and by["USB_P"].skew_mm == pytest.approx(math.sqrt(2) - 1, abs=1e-6)
    assert r.cached > 0
    tight = scan(board.snap, ["CLK"], ScanSettings(z_target_ohm=60, z_tol_pct=5), board=board)
    assert tight.nets[0].status == FAIL


def test_plane_pair_matrix():
    from kipi.si.cavity import cavity_impedance, plane_pair_z, shorted, stitched_impedance
    from kipi.si.materials import Dielectric
    m = Dielectric(4.2, 0.02)
    f = np.array([1e7, 1e9, 5e9])
    sig = [(30.0, 14.6, 0.15), (31.0, 15.75, 0.15)]
    gnd = [(30.0, 13.8, 0.15), (31.0, 16.55, 0.15)]
    z, c0 = plane_pair_z(f, 1.0, sig + gnd, m, (60.0, 30.0))
    # One post, nothing shorted: the single-via cavity model.
    one = cavity_impedance(f, 60, 30, 1.0, sig[0][:2], 0.15, m, modes=600)
    assert np.allclose(z[:, 0, 0] + c0, one, rtol=0.01)
    # Stitching vias shorted: an inductance, as the stitched model has it at low frequency.
    zs = shorted(z[:, [0, 2, 3]][:, :, [0, 2, 3]], c0, 1)[:, 0, 0]
    ref = stitched_impedance(f, 1.0, sig[0][:2], 0.15, gnd)
    assert zs[0].imag > 0 and zs[0].imag == pytest.approx(ref[0].imag, rel=0.15)
    # The pair's two vias share the planes: coupled.
    zp = shorted(z, c0, 2)
    assert 0.1 < zp[1, 0, 1].imag / zp[1, 0, 0].imag < 0.5
    # Planes going on for ever: the same inductance low down, and radiation loss.
    zi, none = plane_pair_z(f, 1.0, sig + gnd, m, None)
    assert none is None
    zo = shorted(zi[:, [0, 2, 3]][:, :, [0, 2, 3]], None, 1)[:, 0, 0]
    assert zo[0].imag == pytest.approx(zs[0].imag, rel=0.15)
    assert zo[2].real > 10 * ref[2].real


def test_a_decoupling_capacitor_ends_a_post_in_its_impedance():
    # Planes of two nets (GND and a supply) joined only by a decoupling capacitor: two
    # signal vias share the floating pair's capacitance, 1/(jωC), coupled ever more
    # strongly as the frequency falls, until the capacitor ties the planes together.
    from kipi.si.cavity import plane_pair_z, shorted
    from kipi.si.materials import Dielectric
    m = Dielectric(4.2, 0.02)
    f = np.array([1e7, 1e8, 1e9])
    sig = [(30.0, 14.6, 0.15), (32.0, 14.6, 0.15)]
    cap = [(40.0, 15.0, 0.15)]
    z, c0 = plane_pair_z(f, 0.1, sig + cap, m, (60.0, 30.0))
    w = 2 * np.pi * f
    zl = (0.01 + 1j * w * 1e-9 + 1 / (1j * w * 10e-6))[:, None]
    floating = z[:, :2, :2] + c0[:, None, None]
    assert np.allclose(shorted(z, c0, 2, np.full((3, 1), 1e12 + 0j)), floating, rtol=1e-6)
    assert np.allclose(shorted(z, c0, 2, np.zeros((3, 1), complex)), shorted(z, c0, 2))
    held = shorted(z, c0, 2, zl)
    assert abs(held[0, 0, 1]) < abs(floating[0, 0, 1]) / 50          # 10 MHz: tied
    assert abs(held[2, 0, 1]) == pytest.approx(abs(shorted(z, c0, 2)[2, 0, 1]), rel=0.5)


def test_plane_pairs_of_two_nets_take_their_decoupling_capacitors():
    # The capacitors joining GND and a supply, nearest first, values read, the build's
    # not-fitted ones and other parts left out.
    import dataclasses

    from kipi.core.model.board import Footprint, Pad
    from kipi.si.extract import decaps_between
    snap = make_board()
    extra_fp, extra_pads = [], []
    for ref, value, x, nets in (("C90", "100n", 10.0, ("GND", "VCC")),
                                ("C91", "22u", 40.0, ("VCC", "GND")),
                                ("C92", "?", 12.0, ("GND", "VCC")),
                                ("R90", "0", 11.0, ("GND", "VCC")),
                                ("C93", "1u", 13.0, ("GND", "CLK"))):
        pads = [Pad(f"{ref}.{k}", n, ref, str(k + 1), (x, 5.0 + k), "smd", (0.0, 0.0), {})
                for k, n in enumerate(nets)]
        extra_pads += pads
        extra_fp.append(Footprint(ref, ref, value, (x, 5.5), "top", [p.id for p in pads]))
    snap = dataclasses.replace(snap, pads=snap.pads + extra_pads,
                               footprints=snap.footprints + extra_fp)
    b = SIBoard(snap)
    got = decaps_between(b, "VCC", "GND", (0.0, 5.0))
    assert [(d[0], d[3]) for d in got] == [(10.0, pytest.approx(100e-9)),
                                           (40.0, pytest.approx(22e-6))]
    assert [d[0] for d in decaps_between(b, "GND", "VCC", (0.0, 5.0), ["C90"])] == [40.0]


def test_channel_vias_are_coupled(board):
    opts = BuildOptions(0.5)
    c = build_channel(board, [["USB_P", "USB_N"]], opts).circuit
    assert any(b.label == "via barrels" and len(b.nodes) == 4 for b in c.blocks)


def test_slow_control_nets_by_name():
    from kipi.si.scan import is_slow
    for n in ("/ACQ and FE Voltage Regs/FE_EN", "nRESET", "PGOOD", "/Power/3V3_PG", "LED2",
              "I2C_SDA", "SCL", "JTAG_TCK", "FPGA_INIT_B", "/IO/GPIO12", "~{RST}"[2:-1]):
        assert is_slow(n), n
    for n in ("/ADC Driver 1/OUT_P", "CLK", "USB_N", "DDR_DQ3", "PCIE_TX0_P", "ADC_D7",
              "GENERATOR", "SENSE_OUT", "LVDS_CLK_N", "OPEN_DRAIN"):
        assert not is_slow(n), n


def test_cuts_alike_share_a_solve():
    from kipi.si.board import dielectric_bands
    from kipi.si.xsection import Cache, Conductor
    bands = dielectric_bands(make_board().stackup)
    box = (-1.5, 1.5, -2.0, 3.6)

    def cut(name, gap, plane=(-5.0, 9.0), m=1.0):
        x = sorted((m * (0.1 + gap), m * (0.3 + gap)))
        return [Conductor.rect(-0.1, 0.1, 1.53, 1.565, name),
                Conductor.rect(x[0], x[1], 1.53, 1.565, name + "_N"),
                Conductor.rect(plane[0], plane[1], 1.295, 1.33)]
    c = Cache(ground_grid_mm=0.025)
    a = c.key(cut("A", 0.15), bands, box)
    assert c.key(cut("B", 0.15), bands, box) == a                    # another net's name
    assert c.key(cut("A", 0.15, (-3.0, 2.0)), bands, box) == a       # plane past the window
    assert c.key(cut("A", 0.15, m=-1.0), bands, box) == a            # the mirror image
    assert c.key(cut("A", 0.20), bands, box) != a                    # a different gap


def test_quick_scan_in_parallel_matches_serial(board):
    one = scan(board.snap, settings=ScanSettings(workers=1), board=SIBoard(board.snap))
    import kipi.si.scan as sc
    old = sc.PARALLEL_MIN
    sc.PARALLEL_MIN = 1
    try:
        many = scan(board.snap, settings=ScanSettings(workers=2), board=SIBoard(board.snap))
    finally:
        sc.PARALLEL_MIN = old
    assert many.solves == one.solves
    for a, b in zip(one.segments, many.segments):
        assert a.z0 == pytest.approx(b.z0, rel=1e-9)
    # The board keeps the scan's cuts and solves: a second scan solves nothing.
    scan(board.snap, settings=ScanSettings(), board=board)
    again = scan(board.snap, settings=ScanSettings(), board=board)
    assert again.solves == 0


def test_screened_cut_keeps_what_bears_on_the_trace():
    from kipi.si.scan import screened
    from kipi.si.xsection import GROUND, Conductor
    # A trace 0.1 mm over a plane, another plane 1 mm above, a far trace and copper on a
    # layer beyond the lower plane.
    conds = [Conductor.rect(-0.1, 0.1, 1.0, 1.035, "S"),
             Conductor.rect(-1.5, 1.5, 0.865, 0.9),                     # plane below, h 0.1
             Conductor.rect(-1.5, 1.5, 2.0, 2.035),                     # plane above
             Conductor.rect(0.3, 0.5, 1.0, 1.035, "NEAR"),
             Conductor.rect(1.2, 1.4, 1.0, 1.035, "FAR"),
             Conductor.rect(-0.5, 0.5, 0.2, 0.235)]                     # behind the plane
    out, box, found = screened(conds, (-1.5, 1.5, -1.0, 3.0), ["NEAR", "FAR"], 5.0)
    assert box[1] == pytest.approx(0.1 + 5 * 0.1, abs=0.01)
    assert found == ["NEAR"]
    assert all(c.bounds[2] > 0.5 for c in out)                          # nothing behind
    assert sum(c.name == GROUND for c in out) == 2
    assert screened(conds, (-1.5, 1.5, -1.0, 3.0), ["NEAR", "FAR"], 0.0)[0] is conds


def test_partner_missing_from_the_cut_goes_on_alone(monkeypatch):
    # A pair's leg the reference's cross-section does not cross (it has turned into a
    # pad there, as at a BGA's escape) is solved on its own, not dropped: dropped, its
    # net would end short of the part (ThunderScope's MGT_TX2_N at the FPGA).
    b = SIBoard(make_board())
    whole = build_channel(b, [["USB_P", "USB_N"]], BuildOptions())
    section = SIBoard.section

    def no_partners(self, *a, **k):
        conds, box, _ = section(self, *a, **k)
        return conds, box, []
    monkeypatch.setattr(SIBoard, "section", no_partners)
    b = SIBoard(make_board())
    ch = build_channel(b, [["USB_P", "USB_N"]], BuildOptions())
    assert ch.net_length("USB_N") == pytest.approx(whole.net_length("USB_N"), rel=1e-6)
    assert {k for k in ch.circuit.nodes if k[0] == "pad"} == \
        {k for k in whole.circuit.nodes if k[0] == "pad"}


def test_the_builds_dnp_parts_are_not_followed():
    # KiCad's DNP (or the chosen assembly variant's) leaves R1 off for every channel,
    # with the channel's own Not fitted list; a variant the board lacks: its own DNP.
    from kipi.si.model import SISettings, new_channel, not_fitted
    snap = make_board()
    r1 = next(f for f in snap.footprints if f.reference == "R1")
    ch = new_channel("CLK")
    ch.not_fitted = ["C9"]
    assert not_fitted(snap, SISettings(), ch) == ["C9"]
    r1.variant_dnp = {"lite": True}
    snap.variants = {"lite": ""}
    assert not_fitted(snap, SISettings(variant="lite"), ch) == ["C9", "R1"]
    assert not_fitted(snap, SISettings(variant="gone"), ch) == ["C9"]
    r1.dnp = True
    assert not_fitted(snap, SISettings(), ch) == ["C9", "R1"]
