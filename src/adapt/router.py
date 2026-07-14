"""Matched-filter routing heuristic (ADAPT report, Section 3.1).

Rather than running a redundant neural classifier post-detection, ADAPT
reuses the chirp mass already extracted from the best-matching template
during the standard matched-filtering stage to decide which global
physics pathway a candidate should be routed to:

- ``"heavy"``: the high-resolution BBH/NSBH pathway (``W_g,heavy``).
- ``"light"``: the aggressive-compression BNS pathway (``W_g,light``).

This is currently a BBH/BNS-only threshold heuristic. NSBH mass-ratio
routing (Section 3.2) is intentionally out of scope for now.
"""


class MatchedFilterRouter:
    """Routes a confirmed candidate to a global pathway using its trigger chirp mass."""

    HEAVY = "heavy"
    LIGHT = "light"

    def __init__(self, threshold_msun: float = 2.0):
        if threshold_msun <= 0:
            raise ValueError("threshold_msun must be positive")
        self.threshold_msun = threshold_msun

    def route_event(self, chirp_mass_msun: float) -> str:
        """Return the pathway ("heavy" or "light") for a given trigger chirp mass.

        Parameters
        ----------
        chirp_mass_msun : float
            Chirp mass (in solar masses) extracted from the matched-filter
            trigger template, ``M_trigger``.
        """
        if chirp_mass_msun <= 0:
            raise ValueError("chirp_mass_msun must be positive")
        return self.HEAVY if chirp_mass_msun >= self.threshold_msun else self.LIGHT
