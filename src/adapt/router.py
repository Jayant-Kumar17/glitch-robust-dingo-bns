"""Component-mass boundary router for the ADAPT dual-pathway architecture.

ADAPT targets two source classes for which mature, pre-trained deep-learning
parameter-estimation networks exist: binary black holes (BBH) and binary
neutron stars (BNS). There is no widely accepted NSBH-specific PE network
(no "Dingo-NSBH" equivalent), so NSBH systems -- and anything else that
does not cleanly fit BBH or BNS -- are deliberately not forced into an ML
pathway.

The router therefore classifies strictly on component masses:

1. BNS  -- both objects are light enough to be neutron stars.
2. BBH  -- both objects are heavy enough to be black holes.
3. AMBIGUOUS -- everything else (asymmetric NSBH systems, lower-mass-gap
   objects), routed to traditional, non-machine-learning offline analysis
   rather than guessed at.
"""


class MatchedFilterRouter:
    """Routes a trigger to a BNS or BBH pathway, or safely to AMBIGUOUS.

    Anything that is not cleanly a pair of neutron stars (both component
    masses <= ns_max) or a pair of black holes (both component masses
    >= bh_min) -- e.g. an NSBH system or a lower-mass-gap object -- is
    routed to AMBIGUOUS for offline analysis.
    """

    BNS = "BNS"
    BBH = "BBH"
    AMBIGUOUS = "AMBIGUOUS"

    def __init__(self, ns_max: float = 2.2, bh_min: float = 5.0):
        """
        Args:
            ns_max: Maximum plausible neutron-star mass (solar masses).
            bh_min: Minimum plausible black-hole mass (solar masses).
        """
        self.ns_max = ns_max
        self.bh_min = bh_min

    def route_event(self, m1: float, m2: float, chi_eff: float = 0.0) -> dict:
        """Classify a trigger from its component masses.

        Parameters
        ----------
        m1, m2 : float
            Component masses (solar masses). Order does not matter; they
            are sorted internally.
        chi_eff : float, optional
            Effective aligned spin. Accepted for interface compatibility
            with the wider pipeline but not used by the boundary rules.

        Returns
        -------
        dict with keys "route" (BNS/BBH/AMBIGUOUS) and "confidence"
        (1.0 for a clean BNS/BBH classification, 0.5 for AMBIGUOUS).
        """
        m_hi = max(m1, m2)
        m_lo = min(m1, m2)

        # Both objects light enough to be neutron stars -> BNS.
        if m_hi <= self.ns_max:
            return {"route": self.BNS, "confidence": 1.0}

        # Both objects heavy enough to be black holes -> BBH.
        if m_lo >= self.bh_min:
            return {"route": self.BBH, "confidence": 1.0}

        # NSBH, lower mass gap, or otherwise unclear -> AMBIGUOUS.
        return {"route": self.AMBIGUOUS, "confidence": 0.5}
