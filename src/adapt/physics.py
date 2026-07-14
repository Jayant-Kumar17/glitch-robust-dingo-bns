"""Basic compact-binary mass relations used across the routing and injection pipeline.

These are the physical quantities extracted from (or attributed to) a
matched-filter trigger: chirp mass controls the leading-order inspiral
phasing, total mass sets the overall scale of the system, and mass ratio
describes how symmetric the two components are.
"""


def chirp_mass(m1: float, m2: float) -> float:
    """Chirp mass (in the same mass units as m1/m2, e.g. solar masses)."""
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def total_mass(m1: float, m2: float) -> float:
    """Total mass M = m1 + m2."""
    return m1 + m2


def mass_ratio(m1: float, m2: float) -> float:
    """Mass ratio q = min(m1, m2) / max(m1, m2), so q is always in (0, 1]."""
    return min(m1, m2) / max(m1, m2)
