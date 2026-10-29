"""Vector fitting: a known rational function is recovered, a channel's S-parameters fit
passively, and the SPICE netlist written is the model."""

import numpy as np

from kipi.si import vectfit as vf


def test_recovers_a_rational_function():
    f = np.linspace(1e7, 20e9, 600)
    s = 2j * np.pi * f
    poles = np.array([-2e9 + 3e10j, -2e9 - 3e10j, -5e9])
    res = np.array([4e9 + 1e9j, 4e9 - 1e9j, 2e9])
    h = 0.1 + sum(r / (s - p) for r, p in zip(res, poles))
    m = vf.fit(f, h[:, None, None], n_poles=4)
    assert m.rms_error < 1e-6 * np.abs(h).max()
    found = sorted(m.poles, key=lambda z: (z.real, z.imag))
    for p in poles:
        assert min(abs(np.array(found) - p)) / abs(p) < 1e-3


def test_channel_model_and_netlist(tmp_path):
    from helpers.si_boards import make_board
    from kipi.dc.pi.model import PISetup
    from kipi.si.model import new_channel
    from kipi.si.report import export_spice
    from kipi.si.solver import solve_channel
    snap = make_board()
    setup = PISetup()
    setup.si_channels = [new_channel("USB_P", "USB_N")]
    r = solve_channel(snap, setup, "USB_P/USB_N")
    m = vf.fit_sparams(r.freqs, r.s)
    assert m.rms_error < 2e-3
    assert vf.passivity(m, vf._check_grid(r.freqs)) <= 1.0
    assert np.all(m.poles.real < 0)
    path, notes = export_spice(tmp_path / "ch.lib", r)
    text = path.read_text()
    assert text.count(".subckt") == 1 and ".ends" in text
    f = r.freqs[1::40]
    s_net = vf.spice_sparams(text, f)
    assert np.max(np.abs(s_net - m(f))) < 1e-6
    assert np.max(np.abs(s_net - r.s[1::40])) < 0.02
