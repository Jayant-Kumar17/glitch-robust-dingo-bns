"""Fetch published parameters and raw strain for real confirmed events from GWOSC.

This provides an external, peer-reviewed ground truth to validate the
router against, distinct from the self-consistency checks done with
synthetic injections in `adapt.injection`. Nothing here is hardcoded --
every value/file is queried live from GWOSC's event API.

Confirmed events also have a small, pre-cut 32-second strain file
available (in addition to the ~4096-second continuous archive files),
which is what `fetch_event_strain` downloads -- a few MB instead of
~500MB, so it's fast even on a throttled connection.
"""

import sys

import requests

GWOSC_EVENT_API_TEMPLATE = "{host}/eventapi/json/{catalog}/{event_name}/{version_segment}"


def _fetch_event_json(event_name: str, catalog: str, version: int = None, host: str = "https://gwosc.org", timeout: float = 30.0) -> dict:
    version_segment = f"v{version}/" if version is not None else ""
    url = GWOSC_EVENT_API_TEMPLATE.format(host=host, catalog=catalog, event_name=event_name, version_segment=version_segment)

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    return next(iter(data["events"].values()))


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
    event = _fetch_event_json(event_name, catalog, version=version, host=host, timeout=timeout)

    return {
        "name": event["commonName"],
        "gps": event["GPS"],
        "m1": event["mass_1_source"],
        "m2": event["mass_2_source"],
        "chirp_mass_published": event["chirp_mass_source"],
        "chi_eff": event["chi_eff"],
        "distance_mpc": event["luminosity_distance"],
    }


def fetch_confident_catalog_events(
    catalogs=("GWTC-1-confident", "GWTC-2.1-confident", "GWTC-3-confident"),
    host: str = "https://gwosc.org",
    timeout: float = 30.0,
) -> dict:
    """Fetch published parameters for every confident event across several catalogs.

    Unlike `fetch_published_parameters` (one event at a time), this hits
    each catalog's list endpoint once and gets every event's parameters
    back in that single response -- e.g. all 11 GWTC-1-confident events
    come back from one request, not 11.

    Later catalogs in `catalogs` take precedence for events that appear
    in more than one (e.g. GWTC-2.1-confident reanalyzed most GWTC-1
    black-hole events with updated parameters; GWTC-1's own BNS event,
    GW170817, isn't reanalyzed there and so is only pulled from GWTC-1).

    Returns
    -------
    dict mapping commonName (e.g. "GW150914") -> parameter dict with keys
    name, gps, m1, m2, chirp_mass_published, chi_eff, distance_mpc, catalog.
    """
    by_name = {}
    for catalog in catalogs:
        url = f"{host}/eventapi/json/{catalog}/"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        for event in data["events"].values():
            name = event["commonName"]
            by_name[name] = {
                "name": name,
                "gps": event["GPS"],
                "m1": event["mass_1_source"],
                "m2": event["mass_2_source"],
                "chirp_mass_published": event["chirp_mass_source"],
                "chi_eff": event["chi_eff"],
                "distance_mpc": event["luminosity_distance"],
                "catalog": catalog,
            }

    return by_name


def fetch_event_strain(
    event_name: str,
    catalog: str,
    detector: str,
    version: int = None,
    sample_rate: int = 4096,
    duration: int = 32,
    dest_path: str = None,
    host: str = "https://gwosc.org",
    timeout: float = 60.0,
    show_progress: bool = True,
):
    """Download the small, pre-cut real strain segment for a confirmed event.

    Unlike the continuous archive (which always serves ~4096-second,
    ~500MB files regardless of the requested window), GWOSC's event API
    also serves a short segment (typically 32s) around confirmed events,
    which is what this downloads.

    Parameters
    ----------
    event_name, catalog, version : see `fetch_published_parameters`.
    detector : str
        Detector name, e.g. "H1", "L1", "V1".
    sample_rate : int
        Sample rate (Hz) of the file to fetch; GWOSC typically offers
        4096 and 16384.
    duration : int
        Duration (s) of the file to fetch; GWOSC typically offers the
        short (e.g. 32s) event segment and the full 4096s archive chunk.
    dest_path : str, optional
        Where to save the downloaded file. Defaults to a temp path
        derived from the event/detector/sample_rate.
    show_progress : bool
        If True, print byte-level download progress as it happens.

    Returns
    -------
    dest_path : str
        Path to the downloaded HDF5 file.
    """
    event = _fetch_event_json(event_name, catalog, version=version, host=host, timeout=timeout)

    matches = [
        s
        for s in event["strain"]
        if s["detector"] == detector and s["sampling_rate"] == sample_rate and s["duration"] == duration and s["format"] == "hdf5"
    ]
    if not matches:
        available = [(s["detector"], s["sampling_rate"], s["duration"], s["format"]) for s in event["strain"]]
        raise ValueError(f"No strain file found for detector={detector}, sample_rate={sample_rate}, duration={duration}. Available: {available}")

    url = matches[0]["url"]
    if dest_path is None:
        dest_path = f"/tmp/{event_name}_{detector}_{duration}s_{sample_rate}Hz.hdf5"

    if show_progress:
        print(f"Downloading real strain for {event_name} ({detector}, {duration}s @ {sample_rate}Hz):", flush=True)
        print(f"  {url}", flush=True)

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0))

    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if show_progress and total:
                pct = downloaded / total * 100
                sys.stdout.write(f"\r  {downloaded}/{total} bytes ({pct:.1f}%)")
                sys.stdout.flush()

    if show_progress:
        print(f"\n  Done -> {dest_path}", flush=True)

    return dest_path
