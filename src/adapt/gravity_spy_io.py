"""Gravity Spy O3 catalogue access and real-glitch excerpt extraction.

Data source: Zenodo record 5649212, *Gravity Spy Machine Learning
Classifications of LIGO Glitches from Observing Runs O1, O2, O3a, and O3b*
(Glanzer et al. 2023, Class. Quantum Grav. 40 065004).

The raw per-run CSVs are large (H1_O3a ~86 MB, H1_O3b ~106 MB) and are
downloaded once into ``data/gravity_spy/raw/`` (git-ignored). The filtered
selection used by the paper is cached as a small committed CSV.

Column names in the Zenodo tables have varied across releases, so
:func:`normalise_columns` maps known aliases onto a canonical schema and
raises if a required column is absent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

logger = logging.getLogger(__name__)

ZENODO_RECORD = 5649212
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ZENODO_FILES = {
    # file name -> md5 published in the record (verified 2026-09-18)
    "H1_O3a.csv": "29aea278b622cd97496971f7c07f7d6a",
    "H1_O3b.csv": "590290fe3c7ddf8cd9fa85c3a09cd5e4",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "gravity_spy" / "raw"
DEFAULT_SELECTED_CSV = REPO_ROOT / "data" / "gravity_spy" / "selected_h1_o3.csv"

# Canonical schema -> accepted aliases (case-insensitive match on lower()).
_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "event_time": ("event_time", "eventtime", "peak_time_gps", "gps"),
    "peak_time": ("peak_time", "peaktime"),
    "peak_time_ns": ("peak_time_ns",),
    "ifo": ("ifo", "detector"),
    "duration": ("duration",),
    "peak_frequency": ("peak_frequency", "peakfrequency", "peak_freq"),
    "snr": ("snr",),
    "ml_label": ("ml_label", "label", "mllabel", "ml_class"),
    "ml_confidence": ("ml_confidence", "confidence", "mlconfidence", "ml_conf"),
    "gravityspy_id": ("gravityspy_id", "id", "gid"),
}
REQUIRED = ("event_time", "ifo", "duration", "snr", "ml_label", "ml_confidence")

DEFAULT_LABEL_QUOTA: Dict[str, int] = {
    "Blip": 10,
    "Scattered_Light": 5,
    "Whistle": 5,
    "Koi_Fish": 5,
}


# ---------------------------------------------------------------------------
# Download / verification
# ---------------------------------------------------------------------------


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_zenodo_record(timeout: float = 60.0) -> Dict[str, Dict[str, str]]:
    """Query the Zenodo API and return ``{file_name: {md5, url, size}}``.

    Raises ``RuntimeError`` if the expected H1 O3 files are missing from the
    record or their checksums differ from the ones this module was written
    against (in which case the column-alias table should be re-checked).
    """
    with urllib.request.urlopen(ZENODO_API, timeout=timeout) as resp:
        rec = json.loads(resp.read().decode("utf-8"))
    files: Dict[str, Dict[str, str]] = {}
    for f in rec.get("files", []):
        files[f["key"]] = {
            "md5": str(f.get("checksum", "")).replace("md5:", ""),
            "url": f["links"]["self"],
            "size": str(f.get("size")),
        }
    missing = [k for k in ZENODO_FILES if k not in files]
    if missing:
        raise RuntimeError(f"Zenodo record {ZENODO_RECORD} lacks files: {missing}")
    for k, expected in ZENODO_FILES.items():
        got = files[k]["md5"]
        if got != expected:
            logger.warning(
                "Zenodo md5 for %s changed (%s -> %s); re-check column aliases.",
                k,
                expected,
                got,
            )
    logger.info(
        "Zenodo record %d verified: title=%r version=%s",
        ZENODO_RECORD,
        rec.get("metadata", {}).get("title"),
        rec.get("metadata", {}).get("version"),
    )
    return files


def download_h1_o3_tables(
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    verify: bool = True,
    timeout: float = 1800.0,
) -> List[Path]:
    """Download ``H1_O3a.csv`` and ``H1_O3b.csv`` if absent; verify MD5."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = verify_zenodo_record() if verify else {
        k: {"url": f"{ZENODO_API}/files/{k}/content", "md5": v}
        for k, v in ZENODO_FILES.items()
    }
    out: List[Path] = []
    for name in ZENODO_FILES:
        dest = raw_dir / name
        want = files[name]["md5"]
        if dest.is_file() and _md5(dest) == want:
            logger.info("Cached %s (md5 ok)", dest)
            out.append(dest)
            continue
        logger.info("Downloading %s -> %s", files[name]["url"], dest)
        urllib.request.urlretrieve(files[name]["url"], dest)  # noqa: S310
        got = _md5(dest)
        if got != want:
            raise RuntimeError(f"MD5 mismatch for {name}: {got} != {want}")
        out.append(dest)
    return out


# ---------------------------------------------------------------------------
# Column normalisation and filtering
# ---------------------------------------------------------------------------


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to the canonical schema; raise on missing columns."""
    lower = {c.lower(): c for c in df.columns}
    rename: Dict[str, str] = {}
    for canon, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            if a.lower() in lower and lower[a.lower()] != canon:
                rename[lower[a.lower()]] = canon
                break
    out = df.rename(columns=rename)
    # If event_time absent but peak_time present, synthesise it.
    if "event_time" not in out.columns and "peak_time" in out.columns:
        ns = out["peak_time_ns"].astype(float) if "peak_time_ns" in out.columns else 0.0
        out["event_time"] = out["peak_time"].astype(float) + ns * 1e-9
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise KeyError(
            f"Gravity Spy table missing required columns {missing}; "
            f"available: {sorted(out.columns)[:40]} ..."
        )
    return out


def read_gravity_spy_csv(path: Path, usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read a Gravity Spy CSV keeping only the canonical columns."""
    header = pd.read_csv(path, nrows=0)
    header = normalise_columns(header)
    # Reverse-map canonical -> original name for usecols.
    orig = pd.read_csv(path, nrows=0).columns
    lower = {c.lower(): c for c in orig}
    keep_orig: List[str] = []
    for canon, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            if a.lower() in lower:
                keep_orig.append(lower[a.lower()])
                break
    df = pd.read_csv(path, usecols=sorted(set(keep_orig)), low_memory=False)
    return normalise_columns(df)


@dataclass(frozen=True)
class SelectionCriteria:
    ifo: str = "H1"
    min_confidence: float = 0.95
    snr_min: float = 8.0
    snr_max: float = 30.0
    max_duration_s: float = 3.0
    # Only glitches whose Omicron peak frequency lies inside the DINGO-BNS
    # analysis band (f_max ~ 1535 Hz) can affect inference; excerpts are
    # band-passed to 1500 Hz, so out-of-band triggers would be filtered away.
    max_peak_frequency_hz: float = 1500.0
    label_quota: Mapping[str, int] = None  # type: ignore[assignment]

    def quota(self) -> Dict[str, int]:
        return dict(self.label_quota or DEFAULT_LABEL_QUOTA)


def select_glitches(
    df: pd.DataFrame,
    criteria: SelectionCriteria = SelectionCriteria(),
    *,
    min_separation_s: float = 60.0,
) -> pd.DataFrame:
    """Apply the paper's filter and take the top-confidence rows per label.

    ``min_separation_s`` avoids picking two triggers from the same noisy
    minute so the panel is not dominated by a single instrumental episode.
    """
    d = df.copy()
    d = d[d["ifo"].astype(str).str.upper() == criteria.ifo.upper()]
    d = d[d["ml_confidence"].astype(float) >= float(criteria.min_confidence)]
    d = d[(d["snr"].astype(float) >= criteria.snr_min) & (d["snr"].astype(float) <= criteria.snr_max)]
    d = d[d["duration"].astype(float) <= float(criteria.max_duration_s)]
    if "peak_frequency" in d.columns and criteria.max_peak_frequency_hz:
        d = d[d["peak_frequency"].astype(float) <= float(criteria.max_peak_frequency_hz)]
    parts: List[pd.DataFrame] = []
    for label, n in criteria.quota().items():
        sub = d[d["ml_label"].astype(str) == label].sort_values(
            ["ml_confidence", "snr"], ascending=[False, False]
        )
        chosen: List[int] = []
        times: List[float] = []
        for idx, row in sub.iterrows():
            t = float(row["event_time"])
            if all(abs(t - u) >= min_separation_s for u in times):
                chosen.append(idx)
                times.append(t)
            if len(chosen) >= n:
                break
        if len(chosen) < n:
            logger.warning("Only %d/%d rows available for label %s", len(chosen), n, label)
        parts.append(sub.loc[chosen])
    out = pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0]
    cols = [c for c in ("gravityspy_id", "event_time", "ifo", "ml_label", "ml_confidence",
                        "snr", "duration", "peak_frequency") if c in out.columns]
    return out[cols].reset_index(drop=True)


DEFAULT_SNR_BINS: Sequence[Tuple[float, float]] = ((8.0, 30.0), (30.0, 100.0), (100.0, 300.0), (300.0, float("inf")))
DEFAULT_WIDE_LABELS: Sequence[str] = ("Blip", "Koi_Fish", "Scattered_Light", "Whistle", "Tomte")


def snr_bin_label(snr: float, bins: Sequence[Tuple[float, float]] = DEFAULT_SNR_BINS) -> str:
    for lo, hi in bins:
        if lo <= float(snr) < hi:
            return f"[{lo:g},{'inf' if not np.isfinite(hi) else f'{hi:g}'})"
    return "out_of_range"


def select_glitches_binned(
    df: pd.DataFrame,
    *,
    labels: Sequence[str] = DEFAULT_WIDE_LABELS,
    bins: Sequence[Tuple[float, float]] = DEFAULT_SNR_BINS,
    n_per_bin: int = 5,
    ifo: str = "H1",
    min_confidence: float = 0.95,
    max_duration_s: float = 3.0,
    max_peak_frequency_hz: Optional[float] = 1500.0,
    min_separation_s: float = 60.0,
) -> pd.DataFrame:
    """Wide-SNR selection: ``n_per_bin`` top-confidence glitches per label and
    Omicron-SNR bin (no upper SNR cap). Adds an ``snr_bin`` column."""
    d = df.copy()
    d = d[d["ifo"].astype(str).str.upper() == ifo.upper()]
    d = d[d["ml_confidence"].astype(float) >= float(min_confidence)]
    d = d[d["duration"].astype(float) <= float(max_duration_s)]
    if max_peak_frequency_hz and "peak_frequency" in d.columns:
        d = d[d["peak_frequency"].astype(float) <= float(max_peak_frequency_hz)]
    parts: List[pd.DataFrame] = []
    for label in labels:
        dl = d[d["ml_label"].astype(str) == label]
        for lo, hi in bins:
            sub = dl[(dl["snr"].astype(float) >= lo) & (dl["snr"].astype(float) < hi)]
            sub = sub.sort_values(["ml_confidence", "snr"], ascending=[False, False])
            chosen: List[int] = []
            times: List[float] = []
            for idx, row in sub.iterrows():
                t = float(row["event_time"])
                if all(abs(t - u) >= min_separation_s for u in times):
                    chosen.append(idx); times.append(t)
                if len(chosen) >= n_per_bin:
                    break
            if len(chosen) < n_per_bin:
                logger.warning("label %s bin [%g,%g): only %d/%d available", label, lo, hi, len(chosen), n_per_bin)
            p = sub.loc[chosen].copy()
            p["snr_bin"] = snr_bin_label(lo, bins)
            parts.append(p)
    out = pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0]
    cols = [c for c in ("gravityspy_id", "event_time", "ifo", "ml_label", "ml_confidence", "snr", "snr_bin",
                        "duration", "peak_frequency") if c in out.columns]
    return out[cols].reset_index(drop=True)


def build_selected_catalogue(
    raw_paths: Sequence[Path],
    out_csv: Path = DEFAULT_SELECTED_CSV,
    criteria: SelectionCriteria = SelectionCriteria(),
) -> pd.DataFrame:
    """Read raw tables, filter, and write the committed selection CSV."""
    frames = [read_gravity_spy_csv(Path(p)) for p in raw_paths]
    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d Gravity Spy rows from %d files", len(df), len(frames))
    sel = select_glitches(df, criteria)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out_csv, index=False)
    logger.info("Wrote %d selected glitches -> %s", len(sel), out_csv)
    return sel


def load_selected_catalogue(path: Path = DEFAULT_SELECTED_CSV) -> pd.DataFrame:
    return normalise_columns(pd.read_csv(path))


def pick_noise_only_times(
    df_all: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    n: int,
    guard_s: float = 10.0,
    rng: np.random.Generator,
    max_tries: int = 20_000,
) -> List[float]:
    """Return GPS times on the same days as ``selected`` with no Omicron
    trigger of any class within ``±guard_s`` in the full H1 table."""
    all_t = np.sort(df_all["event_time"].astype(float).to_numpy())
    days = sorted({int(t // 86400) for t in selected["event_time"].astype(float)})
    out: List[float] = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        day = days[int(rng.integers(0, len(days)))]
        t = float(day * 86400 + rng.uniform(0.0, 86400.0))
        i = np.searchsorted(all_t, t)
        near = []
        if i > 0:
            near.append(all_t[i - 1])
        if i < all_t.size:
            near.append(all_t[i])
        if all(abs(t - u) > guard_s for u in near) and all(abs(t - u) > guard_s for u in out):
            out.append(t)
    if len(out) < n:
        raise RuntimeError(f"Could only find {len(out)}/{n} trigger-free times")
    return out


# ---------------------------------------------------------------------------
# Strain excerpts
# ---------------------------------------------------------------------------


class HttpRangeFile:
    """Minimal read-only file-like object backed by HTTP range requests.

    Used to let ``h5py`` read only the chunks it needs from a large GWOSC
    strain file instead of downloading the whole 4096 s (~130 MB) file.
    Blocks are cached in memory.
    """

    def __init__(self, url: str, *, block_size: int = 1 << 18, timeout: float = 120.0, retries: int = 6):
        self.url = url
        self.block_size = int(block_size)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self._pos = 0
        self._cache: Dict[int, bytes] = {}
        self.size = self._head_size()
        self.bytes_fetched = 0

    # -- HTTP -------------------------------------------------------------
    def _head_size(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return int(r.headers["Content-Length"])

    def _fetch_block(self, b: int) -> bytes:
        if b in self._cache:
            return self._cache[b]
        a = b * self.block_size
        e = min(self.size, a + self.block_size) - 1
        last: Optional[Exception] = None
        for t in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{e}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                if len(data) != e - a + 1:
                    raise IOError(f"short range read {len(data)} != {e - a + 1}")
                self._cache[b] = data
                self.bytes_fetched += len(data)
                return data
            except Exception as exc:  # pragma: no cover - network
                last = exc
                import time as _t

                _t.sleep(1.5 * (t + 1))
        raise IOError(f"range fetch failed for {self.url} [{a}-{e}]: {last}")

    # -- file-like API ------------------------------------------------------
    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = int(offset)
        elif whence == 1:
            self._pos += int(offset)
        elif whence == 2:
            self._pos = self.size + int(offset)
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        n = max(0, min(int(n), self.size - self._pos))
        out = bytearray()
        pos = self._pos
        while len(out) < n:
            b = pos // self.block_size
            blk = self._fetch_block(b)
            off = pos - b * self.block_size
            take = min(n - len(out), len(blk) - off)
            out += blk[off : off + take]
            pos += take
        self._pos = pos
        return bytes(out)

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def flush(self) -> None:  # h5py may call these
        return None

    def close(self) -> None:
        return None


def gwosc_strain_url(ifo: str, t0: float, t1: float, *, sample_rate: float = 4096.0) -> str:
    """Resolve the GWOSC bulk HDF5 file containing ``[t0, t1]``."""
    from gwosc import locate

    urls = locate.get_urls(ifo, t0, t1, sample_rate=int(sample_rate))
    hdf = [u for u in urls if u.endswith(".hdf5")]
    if not hdf:
        raise FileNotFoundError(f"no GWOSC hdf5 file covers {ifo} [{t0}, {t1}]")
    if len(hdf) > 1:
        raise ValueError(f"interval [{t0}, {t1}] spans {len(hdf)} GWOSC files; choose a shorter span")
    return hdf[0]


def fetch_h1_excerpt_ranged(
    event_time: float,
    *,
    half_span_s: float = 4.0,
    sample_rate: float = 4096.0,
    ifo: str = "H1",
) -> np.ndarray:
    """Fetch only the needed HDF5 chunks of the GWOSC file via HTTP ranges."""
    import h5py

    t0 = float(event_time) - float(half_span_s)
    t1 = float(event_time) + float(half_span_s)
    url = gwosc_strain_url(ifo, t0, t1, sample_rate=sample_rate)
    fobj = HttpRangeFile(url)
    with h5py.File(fobj, "r") as h:
        ds = h["strain"]["Strain"]
        xstart = float(ds.attrs.get("Xstart", h["meta"]["GPSstart"][()]))
        dx = float(ds.attrs.get("Xspacing", 1.0 / sample_rate))
        fs = 1.0 / dx
        i0 = int(round((t0 - xstart) * fs))
        n = int(round(2.0 * half_span_s * fs))
        if i0 < 0 or i0 + n > ds.shape[0]:
            raise ValueError("requested span outside GWOSC file")
        x = np.asarray(ds[i0 : i0 + n], dtype=np.float64)
    logger.info("GWOSC ranged fetch: %d samples using %.1f MB from %s", x.size, fobj.bytes_fetched / 1e6, url.rsplit("/", 1)[-1])
    if abs(fs - sample_rate) > 1e-6:
        from scipy.signal import resample_poly
        from fractions import Fraction

        fr = Fraction(int(round(sample_rate)), int(round(fs)))
        x = resample_poly(x, fr.numerator, fr.denominator)
    return x


def fetch_h1_excerpt(
    event_time: float,
    *,
    half_span_s: float = 4.0,
    sample_rate: float = 4096.0,
    timeout: float = 120.0,
    prefer_ranged: bool = True,
) -> np.ndarray:
    """Fetch ``2*half_span_s`` of H1 open strain centred on ``event_time``.

    Tries a chunk-level HTTP range read of the GWOSC bulk HDF5 file first
    (fast; only the required ~10 s of data are transferred) and falls back to
    ``gwpy.timeseries.TimeSeries.fetch_open_data`` (whole-file download).
    Returns the raw strain at ``sample_rate``. Raises on data gaps.
    """
    n_expect = int(round(2.0 * half_span_s * sample_rate))
    x: Optional[np.ndarray] = None
    if prefer_ranged:
        try:
            x = fetch_h1_excerpt_ranged(event_time, half_span_s=half_span_s, sample_rate=sample_rate)
        except Exception as exc:
            logger.warning("ranged GWOSC fetch failed (%s); falling back to gwpy", exc)
    if x is None:
        from gwpy.timeseries import TimeSeries

        t0 = float(event_time) - float(half_span_s)
        t1 = float(event_time) + float(half_span_s)
        ts = TimeSeries.fetch_open_data("H1", t0, t1, sample_rate=int(sample_rate), cache=False)
        x = np.asarray(ts.value, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite samples in fetched strain (data gap)")
    zero_frac = float(np.mean(np.abs(x) < 1e-30))
    if zero_frac > 0.05:
        raise ValueError(f"dead strain: {zero_frac:.0%} samples ~0 (data gap)")
    if float(np.std(x)) < 1e-25:
        raise ValueError("dead strain: RMS too low (data gap)")
    if abs(x.size - n_expect) > 2:
        raise ValueError(f"fetched {x.size} samples, expected {n_expect}")
    return x[:n_expect] if x.size >= n_expect else np.pad(x, (0, n_expect - x.size))


def bandpass(
    x: np.ndarray,
    sample_rate: float,
    *,
    f_lo: float = 20.0,
    f_hi: float = 1500.0,
    order: int = 8,
) -> np.ndarray:
    nyq = 0.5 * float(sample_rate)
    f_hi = min(float(f_hi), 0.98 * nyq)
    sos = sp_signal.butter(order, [float(f_lo) / nyq, f_hi / nyq], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


@dataclass
class GlitchExcerpt:
    """Windowed real-glitch excerpt ready for additive injection.

    ``waveform`` is the *whitened* (unit-variance background) excerpt on the
    window; ``denoised`` is the STFT-thresholded glitch-only estimate in the
    same units. ``excess_energy_w`` is the whitened excess energy in the window
    (sum of squares minus the expected noise contribution), i.e. an estimate
    of the glitch's matched-filter SNR squared.
    """

    waveform: np.ndarray
    denoised: np.ndarray
    window_s: float
    excess_energy_w: float
    background_rms_w: float
    sample_rate: float
    denoised_energy_w: float = 0.0
    # kept for backward compatibility with the raw-domain estimator
    excess_rms_inband: float = 0.0
    background_rms_inband: float = 0.0

    def native_energy_w(self) -> float:
        """Whitened energy used for native-loudness scaling.

        Prefer the taper-aware window excess; if that is non-positive (quiet
        window or a glitch that the off-window PSD partly absorbed) fall back
        to the STFT-thresholded estimate, which is still a real glitch when
        ``denoised_energy_w > 0``.
        """
        return float(max(self.excess_energy_w, self.denoised_energy_w, 0.0))

    def injection_waveform(self) -> np.ndarray:
        """Prefer the denoised estimate; fall back to the tapered window."""
        den = np.asarray(self.denoised, dtype=np.float64)
        if float(np.sum(den**2)) > 1e-12:
            return den
        return np.asarray(self.waveform, dtype=np.float64)

    def to_meta(self) -> Dict[str, float]:
        return {
            "excerpt_window_s": float(self.window_s),
            "excerpt_excess_energy_w": float(self.excess_energy_w),
            "excerpt_native_snr": float(np.sqrt(self.native_energy_w())),
            "excerpt_denoised_energy_w": float(self.denoised_energy_w),
            "excerpt_background_rms_w": float(self.background_rms_w),
        }


def inband_rms_local(x: np.ndarray, sample_rate: float, f_min: float, f_max: float) -> float:
    from adapt.stft_context import inband_rms

    return float(inband_rms(x, sample_rate, f_min=f_min, f_max=f_max))


def whiten_with_own_background(
    x: np.ndarray,
    sample_rate: float,
    *,
    i0: int,
    i1: int,
    f_lo: float = 20.0,
    f_hi: float = 1500.0,
    nperseg: int = 4096,
) -> np.ndarray:
    """Whiten an excerpt with a Welch PSD estimated on its *off-window* part.

    Returns a series whose off-window background has unit variance. Bins
    outside ``[f_lo, f_hi]`` are zeroed.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    bg = np.concatenate([x[:i0], x[i1:]])
    nperseg = int(min(nperseg, max(256, bg.size // 4)))
    f_w, psd = sp_signal.welch(bg, fs=float(sample_rate), nperseg=nperseg, noverlap=nperseg // 2, average="median")
    asd = np.sqrt(np.maximum(psd, 0.0))
    alpha = float(np.clip(2.0 * 0.1 * sample_rate / max(x.size, 1), 0.0, 1.0))
    X = np.fft.rfft(x * sp_signal.windows.tukey(x.size, alpha=alpha))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    asd_i = np.interp(freqs, f_w, asd, left=asd[0], right=asd[-1])
    band = (freqs >= float(f_lo)) & (freqs <= float(f_hi)) & (asd_i > 0)
    W = np.zeros_like(X)
    W[band] = X[band] / asd_i[band]
    w = np.fft.irfft(W, n=x.size)
    bg_w = np.concatenate([w[:i0], w[i1:]])
    s = float(np.std(bg_w)) or 1.0
    return w / s


def stft_denoise_window(
    w: np.ndarray,
    sample_rate: float,
    *,
    i0: int,
    i1: int,
    nperseg: int = 512,
    hop: int = 128,
    threshold: float = 8.0,
    support: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Keep only STFT pixels inside ``[i0, i1)`` whose power exceeds
    ``threshold`` times the per-frequency median off-window power.

    ``support`` optionally narrows the retained time columns further (e.g. to
    the Omicron duration of the trigger) so that isolated above-threshold
    noise pixels far from the glitch are not amplified by the severity
    scaling. Returns the inverse-STFT of the retained pixels (same length as
    ``w``), i.e. a glitch-only estimate of the whitened excerpt.
    """
    w = np.asarray(w, dtype=np.float64)
    f, t, Z = sp_signal.stft(w, fs=float(sample_rate), nperseg=int(nperseg), noverlap=int(nperseg) - int(hop),
                             boundary="zeros", padded=True)
    t_idx = np.round(t * float(sample_rate)).astype(int)
    in_win = (t_idx >= i0) & (t_idx < i1)
    if support is not None:
        in_sup = (t_idx >= int(support[0])) & (t_idx < int(support[1]))
    else:
        in_sup = in_win
    P = np.abs(Z) ** 2
    if np.any(~in_win):
        bg = np.median(P[:, ~in_win], axis=1, keepdims=True)
    else:
        bg = np.median(P, axis=1, keepdims=True)
    keep = in_sup[None, :] & (P > float(threshold) * np.maximum(bg, 1e-30))
    Zk = np.where(keep, Z, 0.0)
    _, wg = sp_signal.istft(Zk, fs=float(sample_rate), nperseg=int(nperseg), noverlap=int(nperseg) - int(hop),
                            boundary=True)
    wg = np.asarray(wg, dtype=np.float64)
    if wg.size < w.size:
        wg = np.pad(wg, (0, w.size - wg.size))
    return wg[: w.size]


def extract_glitch_excerpt(
    raw_8s: np.ndarray,
    *,
    sample_rate: float,
    duration_s: float,
    min_window_s: float = 0.5,
    tukey_alpha: float = 0.1,
    f_min: float = 23.0,
    f_max: float = 1535.0,
    denoise_threshold: float = 8.0,
    denoise: bool = True,
    support_margin_s: float = 0.15,
) -> GlitchExcerpt:
    """Cut ``event_time ± max(duration, min_window)`` from a centred excerpt.

    Steps: (1) whiten the 8 s excerpt with a PSD estimated on its own
    off-window part (unit-variance background); (2) cut the window and apply a
    Tukey taper; (3) STFT-threshold the window to obtain a glitch-only
    estimate; (4) record the whitened excess energy in the window
    (``sum((w*taper)^2) - sum(taper^2)``, an estimate of the glitch's SNR
    squared after Tukey-aware noise subtraction).

    Working in the whitened domain is essential: in raw strain the 20–40 Hz
    seismic wall dominates the in-band RMS and Gravity Spy glitches of
    Omicron SNR 8–30 are not measurable as an RMS excess.
    """
    x = np.asarray(raw_8s, dtype=np.float64)
    n = x.size
    centre = n // 2
    half_w = max(float(duration_s), float(min_window_s))
    half_n = int(round(half_w * sample_rate))
    i0 = max(0, centre - half_n)
    i1 = min(n, centre + half_n)
    w = whiten_with_own_background(x, sample_rate, i0=i0, i1=i1, f_lo=max(20.0, f_min), f_hi=min(1500.0, f_max))
    taper = sp_signal.windows.tukey(i1 - i0, alpha=float(tukey_alpha))
    taper_full = np.zeros(n)
    taper_full[i0:i1] = taper
    win_w = (w * taper_full)[i0:i1]
    bg_w = np.concatenate([w[:i0], w[i1:]])
    bg_rms = float(np.std(bg_w)) if bg_w.size > 64 else 1.0
    # Tukey-aware noise subtraction: the window is tapered, so the expected
    # noise energy is ``sum(taper^2) * var``, not ``n_win * var``. Using the
    # untapered length made excess_energy systematically negative for quiet
    # but valid excerpts (rho_w was then forced to 0 at native scale).
    excess_energy = float(np.sum(win_w**2) - float(np.sum(taper**2)) * bg_rms**2)
    if denoise:
        # Retain pixels only within the trigger's own duration (+ margin), so
        # that the scaled estimate has compact support like the trigger itself.
        sup_half = int(round((0.5 * float(duration_s) + float(support_margin_s)) * sample_rate))
        support = (max(i0, centre - sup_half), min(i1, centre + sup_half))
        wg_full = stft_denoise_window(
            w, sample_rate, i0=i0, i1=i1, threshold=float(denoise_threshold), support=support
        )
        den = (wg_full * taper_full)[i0:i1]
    else:
        den = win_w.copy()
    return GlitchExcerpt(
        waveform=win_w,
        denoised=den,
        window_s=2.0 * half_w,
        excess_energy_w=excess_energy,
        background_rms_w=bg_rms,
        sample_rate=float(sample_rate),
        denoised_energy_w=float(np.sum(den**2)),
    )


def smooth_asd_lines(asd: np.ndarray, delta_f: float, *, smooth_hz: float = 2.0) -> np.ndarray:
    """Running-median (in log space) of an ASD to remove narrow spectral lines
    while keeping the broadband colour. Used when transplanting glitches so the
    coloured waveform does not ring at line frequencies."""
    from scipy.ndimage import median_filter

    a = np.asarray(asd, dtype=np.float64).ravel()
    pos = a > 0
    la = np.log(np.where(pos, a, np.nan))
    fill = np.nanmedian(la) if np.any(pos) else 0.0
    la = np.where(np.isfinite(la), la, fill)
    k = int(max(3, round(float(smooth_hz) / float(delta_f))))
    k += 1 - (k % 2)
    return np.exp(median_filter(la, size=k, mode="nearest"))


def colour_with_asd(
    w: np.ndarray,
    sample_rate: float,
    *,
    asd: np.ndarray,
    delta_f: float,
    f_min: float,
    f_max: float,
    smooth_lines: bool = True,
    pad_factor: int = 4,
    retaper_alpha: float = 0.1,
) -> np.ndarray:
    """Map a whitened series into strain units with the target detector ASD
    (defined on the packaged grid ``k*delta_f``).

    The ASD is line-smoothed (running median, ~2 Hz) so that narrow lines do
    not induce long ringing; the series is zero-padded before the FFT to avoid
    circular wrap-around and re-tapered afterwards so the output has the same
    compact support as the input window. Bins outside ``[f_min, f_max]`` are
    zeroed. Absolute normalisation is arbitrary and is fixed downstream by the
    severity / SNR scaling rule."""
    w = np.asarray(w, dtype=np.float64)
    n = w.size
    n_pad = int(pad_factor) * n
    x = np.zeros(n_pad)
    s = (n_pad - n) // 2
    x[s : s + n] = w
    W = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n_pad, d=1.0 / float(sample_rate))
    asd_arr = np.asarray(asd, dtype=np.float64).ravel()
    if smooth_lines:
        asd_arr = smooth_asd_lines(asd_arr, delta_f)
    f_asd = np.arange(asd_arr.size) * float(delta_f)
    asd_i = np.interp(freqs, f_asd, asd_arr, left=asd_arr[0], right=asd_arr[-1])
    band = (freqs >= float(f_min)) & (freqs <= float(f_max))
    G = np.zeros_like(W)
    G[band] = W[band] * asd_i[band]
    g = np.fft.irfft(G, n=n_pad)[s : s + n]
    if retaper_alpha and retaper_alpha > 0:
        g = g * sp_signal.windows.tukey(n, alpha=float(retaper_alpha))
    return g


def place_series(
    w: np.ndarray, *, n_samples: int, sample_rate: float, t_peak: float
) -> np.ndarray:
    """Place ``w`` centred at ``t_peak`` in a zero series of ``n_samples``."""
    out = np.zeros(int(n_samples), dtype=np.float64)
    c = int(round(float(t_peak) * float(sample_rate)))
    i0 = c - w.size // 2
    j0 = max(0, i0)
    j1 = min(int(n_samples), i0 + w.size)
    if j1 > j0:
        out[j0:j1] = w[j0 - i0 : j1 - i0]
    return out
