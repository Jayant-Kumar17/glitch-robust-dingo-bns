"""Matched-filter routing heuristic (ADAPT report, Section 3.1).

Rather than running a redundant neural classifier post-detection, ADAPT
reuses the physical parameters already recovered from the best-matching
template during the standard matched-filtering stage to decide which
global physics pathway a candidate should be routed to.

This is a hierarchical, multi-parameter decision process rather than a
single threshold lookup, because chirp mass alone is degenerate (many
different (m1, m2) combinations share the same chirp mass):

1. Hard chirp-mass gate: chirp mass deep in the BNS or BBH regime is
   classified immediately, with very high confidence.
2. Soft chirp-mass band: a looser split for the remaining, less extreme
   cases.
3. Mass-structure refinement: total mass and mass ratio are used as a
   sanity check on the chirp-mass guess. If they disagree with the
   soft-band guess, the event is downgraded to "AMBIGUOUS" rather than
   forced into a hard label.
4. Spin confidence modifier: if an aligned/effective spin estimate is
   available, it nudges confidence up but never overrides the mass-based
   decision, since low-latency spin estimates are noisy.

NSBH-specific routing (Section 3.2) is intentionally out of scope for now.
"""

from adapt.physics import chirp_mass, mass_ratio, total_mass


class MatchedFilterRouter:
    """Routes a confirmed candidate to a global pathway (BNS/BBH/AMBIGUOUS)."""

    BNS = "BNS"
    BBH = "BBH"
    AMBIGUOUS = "AMBIGUOUS"

    def __init__(
        self,
        hard_bns_max_msun: float = 0.87,
        hard_bbh_min_msun: float = 39.17,
        soft_bns_max_msun: float = 1.6,
        soft_bbh_min_msun: float = 2.5,
        bns_mtot_max: float = 4.0,
        bbh_mtot_min: float = 6.0,
        bns_q_min: float = 0.7,
        bbh_q_max: float = 0.5,
        bbh_spin_min: float = 0.3,
        bns_spin_max: float = 0.1,
    ):
        self.hard_bns_max_msun = hard_bns_max_msun
        self.hard_bbh_min_msun = hard_bbh_min_msun
        self.soft_bns_max_msun = soft_bns_max_msun
        self.soft_bbh_min_msun = soft_bbh_min_msun
        self.bns_mtot_max = bns_mtot_max
        self.bbh_mtot_min = bbh_mtot_min
        self.bns_q_min = bns_q_min
        self.bbh_q_max = bbh_q_max
        self.bbh_spin_min = bbh_spin_min
        self.bns_spin_max = bns_spin_max

    def route_event(self, m1: float, m2: float, chi_eff: float = None) -> dict:
        """Return a routing decision for a candidate with component masses m1, m2.

        Parameters
        ----------
        m1, m2 : float
            Component masses (in solar masses) recovered from the
            matched-filter trigger template.
        chi_eff : float, optional
            Effective aligned-spin estimate, used only as a confidence
            modifier on top of the mass-based decision.

        Returns
        -------
        dict with keys "mc", "mtot", "q", "route", "confidence".
        """
        mc = chirp_mass(m1, m2)
        mtot = total_mass(m1, m2)
        q = mass_ratio(m1, m2)

        # Stage 1: hard chirp-mass gating.
        if mc < self.hard_bns_max_msun:
            return self._result(mc, mtot, q, self.BNS, "very_high")
        if mc > self.hard_bbh_min_msun:
            return self._result(mc, mtot, q, self.BBH, "very_high")

        # Stage 2: soft chirp-mass band.
        if mc <= self.soft_bns_max_msun:
            base_route, base_confidence = self.BNS, "high"
        elif mc >= self.soft_bbh_min_msun:
            base_route, base_confidence = self.BBH, "high"
        else:
            base_route, base_confidence = self.AMBIGUOUS, "low"

        # Stage 3: mass-structure refinement (total mass + mass ratio sanity check).
        if base_route == self.BNS:
            if mtot < self.bns_mtot_max and q > self.bns_q_min:
                route, confidence = self.BNS, "high"
            else:
                route, confidence = self.AMBIGUOUS, "medium"
        elif base_route == self.BBH:
            if mtot > self.bbh_mtot_min or q < self.bbh_q_max:
                route, confidence = self.BBH, "high"
            else:
                route, confidence = self.AMBIGUOUS, "medium"
        else:
            if mtot < self.bns_mtot_max and q > self.bns_q_min:
                route, confidence = self.BNS, "medium"
            elif mtot > self.bbh_mtot_min or q < self.bbh_q_max:
                route, confidence = self.BBH, "medium"
            else:
                route, confidence = self.AMBIGUOUS, "low"

        # Stage 4: spin as a confidence modifier only, never a hard override.
        if chi_eff is not None:
            if abs(chi_eff) > self.bbh_spin_min and route == self.BBH:
                confidence = "higher"
            elif abs(chi_eff) < self.bns_spin_max and route == self.BNS:
                confidence = "higher"

        return self._result(mc, mtot, q, route, confidence)

    @staticmethod
    def _result(mc: float, mtot: float, q: float, route: str, confidence: str) -> dict:
        return {"mc": mc, "mtot": mtot, "q": q, "route": route, "confidence": confidence}
