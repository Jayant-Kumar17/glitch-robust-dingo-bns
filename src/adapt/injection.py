"""Synthetic signal injection into real detector noise.

This implements the "Option A" simulation approach used throughout the
GW machine learning literature, and previews the distributed
simulation-generation loop described in Section 4.3 of the report:
sample a clean synthetic waveform for known parameters, and superimpose
it onto real background noise to manufacture a labeled training/test
example.

This is intentionally a simplified, "toy" injection: it uses only the
plus polarization, with no detector antenna response, time-of-arrival
delay, or distance/SNR scaling. It exists to verify the routing
pipeline end-to-end against real noise, not to produce physically
realistic detector data.
"""

import numpy as np
from pycbc.waveform import get_td_waveform

from adapt.physics import chirp_mass, effective_spin, mass_ratio, total_mass


def generate_injection(
    m1: float,
    m2: float,
    noise: np.ndarray,
    sample_rate: float,
    spin1z: float = 0.0,
    spin2z: float = 0.0,
    f_lower: float = 25.0,
    approximant: str = "IMRPhenomD",
    merger_offset: float = 2.0,
):
    """Inject a synthetic CBC waveform into a real noise segment.

    Parameters
    ----------
    m1, m2 : float
        Component masses (solar masses) of the synthetic signal.
    noise : np.ndarray
        Real background noise strain to inject the signal into.
    sample_rate : float
        Sample rate (Hz) of `noise`; the waveform is generated at the
        same rate so it can be added directly.
    spin1z, spin2z : float
        Dimensionless aligned-spin components of each component, passed
        directly into the waveform generator so the spins you set here
        actually shape the generated signal.
    f_lower : float
        Starting frequency (Hz) for the waveform generation. Higher
        values shorten long BNS inspirals so they fit into shorter
        noise segments.
    approximant : str
        Waveform approximant name passed to PyCBC/LALSimulation.
    merger_offset : float
        How many seconds before the end of `noise` to place the merger.

    Returns
    -------
    injected : np.ndarray
        `noise` with the synthetic waveform added in, same length as `noise`.
    true_params : dict
        The known parameters used to generate the injection (m1, m2, mc,
        mtot, q, chi_eff) -- i.e. what a real matched-filter trigger would
        have recovered, used here to mock the trigger for the router. All
        derived quantities are computed from m1/m2/spin1z/spin2z, so they
        trace directly back to the parameters passed in.
    """
    hp, _ = get_td_waveform(
        approximant=approximant,
        mass1=m1,
        mass2=m2,
        spin1z=spin1z,
        spin2z=spin2z,
        delta_t=1.0 / sample_rate,
        f_lower=f_lower,
    )
    signal = hp.numpy()

    injected = np.array(noise, dtype=np.float64, copy=True)
    n_noise = len(injected)

    # The waveform's last sample is the merger; place it `merger_offset`
    # seconds before the end of the noise segment.
    merger_index = n_noise - int(merger_offset * sample_rate)
    start = merger_index - len(signal)

    if start < 0:
        # Waveform is longer than the available pre-merger noise (common
        # for low-mass/long BNS inspirals) -- trim the quiet early inspiral.
        signal = signal[-start:]
        start = 0

    end = start + len(signal)
    if end > n_noise:
        signal = signal[: n_noise - start]
        end = n_noise

    injected[start:end] += signal

    true_params = {
        "m1": m1,
        "m2": m2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "mc": chirp_mass(m1, m2),
        "mtot": total_mass(m1, m2),
        "q": mass_ratio(m1, m2),
        "chi_eff": effective_spin(m1, m2, spin1z, spin2z),
    }
    return injected, true_params
