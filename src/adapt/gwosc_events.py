"""Fetch published parameter estimates for real confirmed events from GWOSC.

This provides an external, peer-reviewed ground truth to validate the
router against, distinct from the self-consistency checks done with
synthetic injections in `adapt.injection`. Nothing here is hardcoded --
every value is queried live from GWOSC's event API.
"""

import requests

GWOSC_EVENT_API_TEMPLATE = "{host}/eventapi/json/{catalog}/{event_name}/{version_segment}"


def fetch_published_parameters(
    event_name: str,
    catalog: str,
    version: int = None,
    host: str = "https://gwosc.org",
    timeout: float = 30.0,
) -> dict:
    """Query GWOSC's event API for the published parameter estimate of a confirmed event.

    Parameters
    ----------
    event_name : str
        GWOSC event name, e.g. "GW150914".
    catalog : str
        Catalog short name that hosts this event, e.g. "GWTC-1-confident".
    version : int, optional
        Specific data release version. If omitted, GWOSC returns its
        default (typically the latest) version.
    host : str
        GWOSC host, defaults to the public archive.

    Returns
    -------
    dict with keys: name, gps, m1, m2, chirp_mass_published, chi_eff,
    distance_mpc -- all taken directly from the live API response.
    """
    version_segment = f"v{version}/" if version is not None else ""
    url = GWOSC_EVENT_API_TEMPLATE.format(host=host, catalog=catalog, event_name=event_name, version_segment=version_segment)

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    event = next(iter(data["events"].values()))

    return {
        "name": event["commonName"],
        "gps": event["GPS"],
        "m1": event["mass_1_source"],
        "m2": event["mass_2_source"],
        "chirp_mass_published": event["chirp_mass_source"],
        "chi_eff": event["chi_eff"],
        "distance_mpc": event["luminosity_distance"],
    }
