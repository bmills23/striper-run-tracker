#!/usr/bin/env python3
"""Striped Bass Run Tracker.

Answers, at a glance and on any device, "where are the striped bass likely right
now, and is the run near me?" along the US East Coast (focused on the Gulf of
Maine), from free public data:

  1. A PROBABILITY (relative-likelihood) heatmap -- a weighted blend of water-
     temperature suitability, nearshore proximity, distance to the 50 deg F
     front, recent+seasonal sighting density, and chlorophyll (bait). This is
     the headline layer.
  2. The underlying sea-surface temperature (NOAA/JPL MUR SST), as a toggle.
  3. A single clean 50 deg F front line and live buoy water temps (NOAA NDBC,
     incl. NERACOOS Gulf of Maine moorings).
  4. A plain-English status panel (front location, your-area status, advance
     rate, ETA, best-odds spot) that collapses for mobile.

Output is a single self-contained interactive HTML map. This estimates the
AGGREGATE run, not individual fish -- there is no public live fish-telemetry
feed, and the "probability" is a habitat-suitability index, not a calibrated
probability.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import io
import json
import os
import sys
from dataclasses import dataclass
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from requests.adapters import HTTPAdapter  # noqa: E402
from urllib3.util.retry import Retry  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from scipy.ndimage import distance_transform_edt, gaussian_filter  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

# --- constants --------------------------------------------------------------

ERDDAP_SST = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csv"
# Global MODIS chlorophyll (covers the Atlantic). NOTE: this product is aging
# (reprocessing lags), so bait is an opt-in experimental factor, not a default.
ERDDAP_CHL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chlamday.csv"
INAT_BASE = "https://api.inaturalist.org/v1/observations"
GBIF_BASE = "https://api.gbif.org/v1/occurrence/search"
NDBC_REALTIME = "https://www.ndbc.noaa.gov/data/realtime2/{}.txt"
TAXON = "Morone saxatilis"
MUR_RES_DEG = 0.01

USER_AGENT = ("Mozilla/5.0 (compatible; striper-run-tracker/1.0; "
              "+https://github.com/bmills23/striper-run-tracker)")
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/csv, application/json, */*"}


def _make_session():
    """HTTP session with browser-like headers and retries+backoff. NOAA ERDDAP
    sits behind a WAF that intermittently 403s datacenter (CI) IPs; retrying
    almost always gets through, and this also rides out transient API hiccups."""
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(total=5, connect=3, read=3, status=5,
                  status_forcelist=(403, 408, 425, 429, 500, 502, 503, 504),
                  backoff_factor=1.5, raise_on_status=False,
                  allowed_methods=frozenset(["GET"]))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = _make_session()

F50_C = (50 - 32) * 5 / 9
F55_C = (55 - 32) * 5 / 9
MI_PER_DEG_LAT = 69.0
KM_PER_DEG = 111.0

DEFAULT_BBOX = (35.0, 45.0, -77.0, -66.0)
REGIONS = {"maine": ([43.6, -68.8], 7), "newengland": ([42.0, -70.0], 7),
           "full": (None, 6)}

# Probability blend weights (renormalized over whichever factors are available).
PROB_WEIGHTS = {"temp": 0.40, "near": 0.25, "front": 0.20, "sight": 0.15, "bait": 0.10}
PROB_LABELS = {"temp": "temperature", "near": "nearshore", "front": "front",
               "sight": "sightings", "bait": "bait/chl"}

BUOYS = {
    "44027": ("Jonesport, ME (Down East)", 44.273, -67.314),
    "44034": ("Eastern Maine Shelf", 44.105, -68.108),
    "44032": ("Central Maine Shelf", 43.716, -69.355),
    "44007": ("Portland, ME", 43.525, -70.141),
    "44098": ("Jeffreys Ledge", 42.798, -70.168),
    "44013": ("Boston, MA", 42.346, -70.651),
    "44029": ("Massachusetts Bay", 42.318, -70.566),
    "44090": ("Cape Cod Bay, MA", 41.840, -70.329),
    "44008": ("Nantucket Shoals", 40.504, -69.248),
    "44017": ("Montauk Point, NY", 40.694, -72.048),
    "44025": ("Long Island (Islip), NY", 40.251, -73.164),
    "44065": ("New York Harbor", 40.369, -73.703),
    "44066": ("Texas Tower, NJ", 39.584, -72.594),
    "44009": ("Delaware Bay mouth", 38.457, -74.702),
    "44099": ("Cape Henry, VA", 36.915, -75.719),
    "44014": ("Virginia Beach, VA", 36.611, -74.842),
    "44100": ("Duck, NC", 36.260, -75.594),
}


@dataclass
class Sighting:
    lat: float
    lon: float
    date: str
    place: str
    url: str
    quality: str
    observer: str = ""
    sst_c: float = None


@dataclass
class Record:
    lat: float
    lon: float
    year: int
    source: str
    url: str
    date: str = ""
    basis: str = ""
    recorder: str = ""
    state: str = ""
    uncertainty_m: float = None
    sst_c: float = None


@dataclass
class Buoy:
    bid: str
    name: str
    lat: float
    lon: float
    wtmp_c: float
    when: str


@dataclass
class RunStatus:
    front_text: str
    home_text: str
    advance_text: str
    eta_text: str
    series: list
    summary: str
    best_text: str = ""


def f_temp(c):
    return c * 9 / 5 + 32


def _cell_km(lats, lons):
    """(dlat_km, dlon_km) for one grid cell at the grid's mean latitude."""
    dlat = abs(lats[1] - lats[0]) * KM_PER_DEG
    dlon = abs(lons[1] - lons[0]) * KM_PER_DEG * np.cos(np.radians(np.mean(lats)))
    return dlat, dlon


# --- sea-surface temperature ------------------------------------------------


def fetch_sst(bbox, date=None, stride_deg=0.05, timeout=120):
    """Strided MUR SST grid (deg C). Returns (lats, lons, grid, date_str)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    stride = max(1, round(stride_deg / MUR_RES_DEG))
    time_sel = "[last]" if date is None else f"[({date}T09:00:00Z)]"
    selector = (f"analysed_sst{time_sel}"
                f"[({lat_min}):{stride}:({lat_max})][({lon_min}):{stride}:({lon_max})]")
    resp = SESSION.get(f"{ERDDAP_SST}?{quote(selector, safe='()[]:.,-')}",
                        headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    rows = list(csv.reader(io.StringIO(resp.text)))[2:]
    lat_vals, lon_vals, sst_vals, times = [], [], [], []
    for r in rows:
        times.append(r[0])
        lat_vals.append(float(r[1]))
        lon_vals.append(float(r[2]))
        sst_vals.append(np.nan if r[3] in ("", "NaN") else float(r[3]))
    lats = np.array(sorted(set(lat_vals)))
    lons = np.array(sorted(set(lon_vals)))
    li = {v: i for i, v in enumerate(lats)}
    lj = {v: i for i, v in enumerate(lons)}
    grid = np.full((len(lats), len(lons)), np.nan)
    for la, lo, v in zip(lat_vals, lon_vals, sst_vals):
        grid[li[la], lj[lo]] = v
    return lats, lons, grid, (times[0][:10] if times else (date or "unknown"))


def fetch_land_mask(bbox, stride_deg=0.03):
    """Finer land/water outline (mlats, mlons, mwater) for crisp coastline
    trimming of the heatmaps. Reuses MUR SST at a finer stride (NaN == land)."""
    mlats, mlons, mgrid, _ = fetch_sst(bbox, stride_deg=stride_deg)
    mwater = ~np.isnan(mgrid)
    print(f"[mask] {mwater.shape[0]}x{mwater.shape[1]} land mask "
          f"({stride_deg:.2f} deg)", file=sys.stderr)
    return mlats, mlons, mwater


def extract_front(lats, lons, grid, level_c, min_vertices=8):
    cs = plt.contour(lons, lats, grid, levels=[level_c])
    lines = [[(float(y), float(x)) for x, y in seg]
             for seg in cs.allsegs[0] if len(seg) >= min_vertices]
    plt.close("all")
    return lines


def _path_len_deg(seg):
    a = np.asarray(seg, float)
    if len(a) < 2:
        return 0.0
    d = np.diff(a, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def keep_longest(lines, n=2):
    return sorted(lines, key=_path_len_deg, reverse=True)[:n]


def front_leading_latitude(lines, lon_center, halfwidth=1.0):
    in_band = [lat for line in lines for (lat, lon) in line
               if abs(lon - lon_center) <= halfwidth]
    return max(in_band) if in_band else None


def fetch_sst_history(bbox, days, end_date, stride_deg=0.1):
    base = dt.date.fromisoformat(end_date)
    out = []
    for d in range(days, 0, -1):
        day = (base - dt.timedelta(days=d)).isoformat()
        try:
            la, lo, g, ds = fetch_sst(bbox, date=day, stride_deg=stride_deg)
            out.append((ds, extract_front(la, lo, g, F50_C)))
        except Exception as e:
            print(f"[history] skip {day}: {e}", file=sys.stderr)
    print(f"[history] {len(out)}/{days} prior days", file=sys.stderr)
    return out


def render_sst_png(grid, lats, lons, mask):
    """SST heatmap (diverging colormap centered 50F, finer-mask coastline).
    Returns (uri, bounds, vmin_f, vmax_f)."""
    valid = grid[~np.isnan(grid)]
    vmin = min(float(np.floor(valid.min())) if valid.size else 0.0, F50_C - 1)
    vmax = max(float(np.ceil(valid.max())) if valid.size else 25.0, F50_C + 1)
    norm = TwoSlopeNorm(vcenter=F50_C, vmin=vmin, vmax=vmax)
    uri, bounds = _render_field(grid, lats, lons, mask,
                                matplotlib.colormaps["RdYlBu_r"], norm,
                                lambda f: np.ones(f.shape))
    return uri, bounds, f_temp(vmin), f_temp(vmax)


# --- chlorophyll (bait) -----------------------------------------------------


def fetch_chl(bbox, sst_lats, sst_lons, timeout=120, stride=4):
    """Monthly chlorophyll-a (mg/m^3) regridded onto the SST grid, or None.
    Best-effort: any failure drops the bait factor."""
    lat_min, lat_max, lon_min, lon_max = bbox
    sel = (f"chlorophyll[last]"
           f"[({lat_min}):{stride}:({lat_max})][({lon_min}):{stride}:({lon_max})]")
    try:
        resp = SESSION.get(f"{ERDDAP_CHL}?{quote(sel, safe='()[]:.,-')}",
                            headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        rows = list(csv.reader(io.StringIO(resp.text)))[2:]
        la, lo, va = [], [], []
        chl_date = rows[0][0][:10] if rows else "?"
        for r in rows:                       # time, lat, lon, chlorophyll
            la.append(float(r[1]))
            lo.append(float(r[2]))
            va.append(np.nan if r[3] in ("", "NaN") else float(r[3]))
        clats = np.array(sorted(set(la)))
        clons = np.array(sorted(set(lo)))
        if len(clats) < 2 or len(clons) < 2:
            raise ValueError("chl grid too small")
        li = {v: i for i, v in enumerate(clats)}
        lj = {v: i for i, v in enumerate(clons)}
        cg = np.full((len(clats), len(clons)), np.nan)
        for a, b, v in zip(la, lo, va):
            cg[li[a], lj[b]] = v
        med = float(np.nanmedian(cg)) if np.any(~np.isnan(cg)) else 0.5
        cg = np.where(np.isnan(cg), med, cg)
        rgi = RegularGridInterpolator((clats, clons), cg, bounds_error=False,
                                      fill_value=med)
        lon2d, lat2d = np.meshgrid(sst_lons, sst_lats)
        out = rgi(np.column_stack([lat2d.ravel(), lon2d.ravel()])).reshape(lat2d.shape)
        print(f"[chl] {cg.shape[0]}x{cg.shape[1]} chlorophyll grid from {chl_date} "
              f"(median {med:.2f} mg/m3)", file=sys.stderr)
        return out
    except Exception as e:
        print(f"[chl] unavailable, dropping bait factor: {e}", file=sys.stderr)
        return None


# --- buoys ------------------------------------------------------------------


def _latest_wtmp(body):
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) >= 15 and f[14] != "MM":
            try:
                return float(f[14]), f"{f[0]}-{f[1]}-{f[2]} {f[3]}:{f[4]}Z"
            except ValueError:
                continue
    return None, None


def fetch_buoys(bbox, timeout=15):
    lat_min, lat_max, lon_min, lon_max = bbox
    out = []
    for bid, (name, lat, lon) in BUOYS.items():
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            continue
        try:
            body = SESSION.get(NDBC_REALTIME.format(bid), headers=HEADERS,
                                timeout=timeout).text
        except requests.RequestException:
            continue
        wtmp, when = _latest_wtmp(body)
        if wtmp is not None:
            out.append(Buoy(bid, name, lat, lon, wtmp, when))
    print(f"[buoys] {len(out)} live water-temp buoys", file=sys.stderr)
    return out


# --- sightings & records ----------------------------------------------------


def fetch_sightings(bbox, days=30, timeout=60, max_results=200):
    lat_min, lat_max, lon_min, lon_max = bbox
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    params = {"taxon_name": TAXON, "geo": "true", "verifiable": "true",
              "per_page": str(max_results), "order_by": "observed_on",
              "order": "desc", "d1": since, "swlat": lat_min, "swlng": lon_min,
              "nelat": lat_max, "nelng": lon_max}
    resp = SESSION.get(INAT_BASE, params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", []):
        coords = (r.get("geojson") or {}).get("coordinates")
        if coords:
            lon, lat = float(coords[0]), float(coords[1])
        elif r.get("location"):
            lat, lon = (float(x) for x in r["location"].split(","))
        else:
            continue
        out.append(Sighting(lat, lon,
                            r.get("observed_on") or (r.get("time_observed_at") or "")[:10],
                            r.get("place_guess") or "", r.get("uri") or "",
                            r.get("quality_grade") or "",
                            observer=(r.get("user") or {}).get("login") or ""))
    print(f"[sightings] {len(out)} recent iNaturalist (last {days}d)", file=sys.stderr)
    return out


def fetch_gbif_seasonal(bbox, month_pad=1, timeout=60, max_records=1500):
    """Striped bass records for the current season across ALL years (dense,
    seasonally relevant) -- powers both the density factor and the display layer."""
    lat_min, lat_max, lon_min, lon_max = bbox
    m = dt.date.today().month
    lo, hi = max(1, m - month_pad), min(12, m + month_pad)
    out, offset = [], 0
    while len(out) < max_records:
        params = {"scientificName": TAXON, "hasCoordinate": "true",
                  "decimalLatitude": f"{lat_min},{lat_max}",
                  "decimalLongitude": f"{lon_min},{lon_max}",
                  "month": f"{lo},{hi}", "limit": "300", "offset": str(offset)}
        resp = SESSION.get(GBIF_BASE, params=params, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        for r in data.get("results", []):
            lat, lon = r.get("decimalLatitude"), r.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            key = r.get("key")
            out.append(Record(
                float(lat), float(lon), r.get("year") or 0,
                r.get("datasetName") or r.get("institutionCode") or "GBIF",
                r.get("references") or (f"https://www.gbif.org/occurrence/{key}" if key else ""),
                date=(r.get("eventDate") or "")[:10],
                basis=r.get("basisOfRecord") or "",
                recorder=r.get("recordedBy") or "",
                state=r.get("stateProvince") or "",
                uncertainty_m=r.get("coordinateUncertaintyInMeters")))
        if data.get("endOfRecords") or not data.get("results"):
            break
        offset += 300
    print(f"[gbif] {len(out)} seasonal records (months {lo}-{hi}, all years)",
          file=sys.stderr)
    return out


# --- probability model ------------------------------------------------------


def temp_suitability(grid):
    """0-1 from SST: peaks in the 55-68F feeding band, off below ~48F / above ~75F."""
    s = np.interp(f_temp(grid), [40, 48, 55, 68, 75, 85], [0, 0, 1, 1, 0.12, 0.0])
    s[np.isnan(grid)] = np.nan
    return s


def nearshore_factor(grid, lats, lons, scale_km=30.0):
    """(0-1 score, distance-to-shore km): bass hug the coast/estuaries."""
    water = ~np.isnan(grid)
    dist = distance_transform_edt(water, sampling=list(_cell_km(lats, lons)))
    n = np.exp(-dist / scale_km)
    n[~water] = np.nan
    return n, dist


def front_factor(grid, lats, lons, front_lines, scale_km=30.0):
    """(0-1 score, distance-to-front km), or None: boost near the 50F edge."""
    verts = [(la, lo) for line in front_lines for (la, lo) in line]
    if not verts:
        return None
    lat0 = float(np.mean(lats))
    kx = KM_PER_DEG * np.cos(np.radians(lat0))
    tree = cKDTree([(la * KM_PER_DEG, lo * kx) for la, lo in verts])
    lon2d, lat2d = np.meshgrid(lons, lats)
    dist, _ = tree.query(np.column_stack([(lat2d * KM_PER_DEG).ravel(),
                                          (lon2d * kx).ravel()]))
    dist = dist.reshape(grid.shape)
    f = np.exp(-dist / scale_km)
    f[np.isnan(grid)] = np.nan
    return f, dist


def _edges(c):
    c = np.asarray(c, float)
    m = (c[1:] + c[:-1]) / 2
    return np.concatenate([[2 * c[0] - m[0]], m, [2 * c[-1] - m[-1]]])


def sighting_density(grid, lats, lons, pts, scale_km=35.0):
    """0-1 Gaussian-smoothed density of sighting points on the grid."""
    if not pts:
        return None
    h, _, _ = np.histogram2d([p[0] for p in pts], [p[1] for p in pts],
                             bins=[_edges(lats), _edges(lons)])
    dlat, dlon = _cell_km(lats, lons)
    d = gaussian_filter(h, sigma=[scale_km / dlat, scale_km / dlon])
    mx = d.max()
    if mx > 0:
        d = d / mx
    d[np.isnan(grid)] = np.nan
    return d


def chl_suitability(chl, grid):
    """0-1 from chlorophyll: more bait -> better, log-scaled (0.3 -> 5 mg/m3)."""
    s = (np.log10(np.clip(chl, 0.05, 30)) - np.log10(0.3)) / (np.log10(5) - np.log10(0.3))
    s = np.clip(s, 0, 1)
    s[np.isnan(grid)] = np.nan
    return s


def compute_probability(grid, lats, lons, front_lines, sight_pts, chl_grid, weights,
                        near=None, sight=None, normalize=True):
    """Weighted blend of available 0-1 factors -> 0-1 grid (NaN land). Returns
    (prob, used, detail, wnorm).

    ``near`` (a (score, dist_km) tuple) and ``sight`` (a score grid) may be
    precomputed and passed in to reuse across days. ``normalize=False`` returns
    the raw blend (no per-day rescale) so a whole timeline can share one color
    scale.
    """
    water = ~np.isnan(grid)
    scores, detail = {}, {}
    scores["temp"] = temp_suitability(grid)
    detail["temp"] = {"score": scores["temp"]}
    if near is None:
        near = nearshore_factor(grid, lats, lons)
    scores["near"] = near[0]
    detail["near"] = {"score": near[0], "dist_km": near[1]}
    fr = front_factor(grid, lats, lons, front_lines)
    if fr is not None:
        scores["front"] = fr[0]
        detail["front"] = {"score": fr[0], "dist_km": fr[1]}
    if sight is None:
        sight = sighting_density(grid, lats, lons, sight_pts)
    if sight is not None:
        scores["sight"] = sight
        detail["sight"] = {"score": sight}
    if chl_grid is not None:
        scores["bait"] = chl_suitability(chl_grid, grid)
        detail["bait"] = {"score": scores["bait"]}

    used = [k for k in scores if k in weights]
    tw = sum(weights[k] for k in used)
    wnorm = {k: weights[k] / tw for k in used}
    prob = np.zeros(grid.shape)
    for k in used:
        prob += wnorm[k] * np.nan_to_num(scores[k], nan=0.0)
    prob = gaussian_filter(prob, sigma=1.0)
    prob[~water] = np.nan
    if normalize:
        mx = np.nanmax(prob)
        if mx and mx > 0:
            prob = prob / mx
    return prob, used, detail, wnorm


def build_inspection_cells(lats, lons, grid, prob, detail, mask, stride=2, thresh=0.03):
    """Downsample to ~0.1deg clickable cells (over water only). Returns (rows,
    half_deg); each row is [lat, lon, prob%, tScore, sstF, nScore, shoreKm,
    fScore, frontKm, sScore]."""
    mlats, mlons, mwater = mask

    def is_water(la, lo):
        return bool(mwater[int(np.argmin(np.abs(mlats - la))),
                           int(np.argmin(np.abs(mlons - lo)))])

    def r2(x, nd=2):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)

    tS = detail["temp"]["score"]
    nS, nK = detail["near"]["score"], detail["near"]["dist_km"]
    fS = detail.get("front", {}).get("score")
    fK = detail.get("front", {}).get("dist_km")
    sS = detail.get("sight", {}).get("score")
    rows = []
    for i in range(0, len(lats), stride):
        for j in range(0, len(lons), stride):
            p = prob[i, j]
            if np.isnan(p) or p < thresh or not is_water(lats[i], lons[j]):
                continue
            rows.append([
                round(float(lats[i]), 3), round(float(lons[j]), 3), round(float(p) * 100),
                r2(tS[i, j]), r2(f_temp(grid[i, j]), 0),
                r2(nS[i, j]), r2(nK[i, j], 0),
                r2(fS[i, j]) if fS is not None else None,
                r2(fK[i, j], 0) if fK is not None else None,
                r2(sS[i, j]) if sS is not None else None,
            ])
    half = round(stride * float(lats[1] - lats[0]) / 2, 4)
    return rows, half


def best_odds_text(prob, lats, lons, buoys, home, radius_deg=2.2):
    """Human label for the peak-suitability spot near home."""
    home_lat, home_lon, home_name = home
    lon2d, lat2d = np.meshgrid(lons, lats)
    near = (np.abs(lat2d - home_lat) <= radius_deg) & (np.abs(lon2d - home_lon) <= radius_deg)
    p = np.where(near & ~np.isnan(prob), prob, -1.0)
    if p.max() <= 0:
        return ""
    i, j = np.unravel_index(np.argmax(p), p.shape)
    blat, blon = lats[i], lons[j]
    if buoys:
        nb = min(buoys, key=lambda b: (b.lat - blat) ** 2 + (b.lon - blon) ** 2)
        where = nb.name
    else:
        where = f"{blat:.2f}, {blon:.2f}"
    return f"Best odds near you: around {where}."


# --- status -----------------------------------------------------------------


def _sample_grid(lats, lons, grid, lat, lon):
    i = int(np.argmin(np.abs(lats - lat)))
    j = int(np.argmin(np.abs(lons - lon)))
    if not np.isnan(grid[i, j]):
        return float(grid[i, j])
    for r in range(1, 8):
        sub = grid[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
        if np.any(~np.isnan(sub)):
            return float(np.nanmean(sub))
    return None


def compute_status(buoys, history, front_50, grid, lats, lons, home, date_str):
    home_lat, home_lon, home_name = home
    home_sst = _sample_grid(lats, lons, grid, home_lat, home_lon)

    warm = [b for b in buoys if b.wtmp_c >= F50_C]
    cold = [b for b in buoys if b.wtmp_c < F50_C]
    if not buoys:
        front_text = "Front: no live buoys available right now."
    elif not cold:
        front_text = "Run-temperature water (≥50°F) at every buoy in view."
    elif not warm:
        front_text = "Front is south of the area — all buoys still <50°F."
    else:
        north_warm = max(warm, key=lambda b: b.lat)
        colder_north = [b for b in cold if b.lat > north_warm.lat]
        if colder_north:
            sc = min(colder_north, key=lambda b: b.lat)
            front_text = (f"Front between {north_warm.name} "
                          f"({f_temp(north_warm.wtmp_c):.0f}°F) and {sc.name} "
                          f"({f_temp(sc.wtmp_c):.0f}°F).")
        else:
            front_text = (f"Leading 50°F edge near {north_warm.name} "
                          f"({f_temp(north_warm.wtmp_c):.0f}°F).")

    if home_sst is None:
        home_text = f"{home_name}: no temperature reading."
    elif home_sst >= F50_C:
        home_text = f"{home_name}: {f_temp(home_sst):.0f}°F — run water present ✓"
    else:
        home_text = f"{home_name}: {f_temp(home_sst):.0f}°F — not there yet."

    series = []
    for ds, lines in history:
        la = front_leading_latitude(lines, home_lon)
        if la is not None:
            series.append((ds, la))
    front_lat_now = front_leading_latitude(front_50, home_lon)
    if front_lat_now is not None:
        series.append((date_str, front_lat_now))

    rate_mi = None
    if len(series) >= 2:
        xs = np.array([dt.date.fromisoformat(d).toordinal() for d, _ in series], float)
        ys = np.array([la for _, la in series], float)
        rate_mi = float(np.polyfit(xs, ys, 1)[0]) * MI_PER_DEG_LAT

    if rate_mi is None:
        advance_text = "Advance rate: not enough data."
    elif rate_mi > 0.5:
        advance_text = f"Front advancing ~{rate_mi:.0f} mi/day north."
    elif rate_mi < -0.5:
        advance_text = f"Front sliding ~{abs(rate_mi):.0f} mi/day south."
    else:
        advance_text = "Front roughly stationary."

    if home_sst is not None and home_sst >= F50_C:
        if front_lat_now is not None and front_lat_now > home_lat:
            d = (front_lat_now - home_lat) * MI_PER_DEG_LAT
            eta_text = f"Arrived at {home_name}; leading edge now ~{d:.0f} mi north."
        else:
            eta_text = f"Run water has arrived at {home_name}."
    elif (rate_mi and rate_mi > 0.5 and front_lat_now is not None
          and front_lat_now < home_lat):
        miles = (home_lat - front_lat_now) * MI_PER_DEG_LAT
        eta_text = (f"~{miles:.0f} mi south; est. arrival at {home_name} in "
                    f"~{miles / rate_mi:.0f} days.")
    else:
        eta_text = f"No clear northward advance toward {home_name} right now."

    summary = " | ".join([front_text, home_text, advance_text, eta_text])
    return RunStatus(front_text, home_text, advance_text, eta_text, series, summary)


# --- rendering --------------------------------------------------------------


def temp_color(c):
    if c < F50_C:
        return "#2c7fb8"
    if c < F55_C:
        return "#fdae61"
    return "#d7191c"


def _png_overlay(lats, lons, rgba, px_step=1):
    """RGBA grid (row i = lats[i], ascending) -> (data-uri, bounds).

    Two corrections so the overlay lands exactly on the CartoDB (Web Mercator)
    basemap: (1) expand bounds by half a cell — pixels are areas, not points;
    (2) resample rows to be uniform in Mercator y, since Leaflet stretches the
    image linearly in projected space while the data is equirectangular.
    """
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])
    south, north = float(lats.min()) - dlat / 2, float(lats.max()) + dlat / 2
    west, east = float(lons.min()) - dlon / 2, float(lons.max()) + dlon / 2

    img = rgba[::-1, :, :]  # north-first
    h = img.shape[0]
    merc = lambda d: np.log(np.tan(np.pi / 4 + np.radians(d) / 2))
    ys = np.linspace(merc(north), merc(south), h)
    out_lat = np.degrees(2 * np.arctan(np.exp(ys)) - np.pi / 2)
    asc_idx = np.interp(out_lat, lats, np.arange(len(lats)))
    ri = np.clip(np.round(len(lats) - 1 - asc_idx).astype(int), 0, h - 1)
    out = img[ri, :, :]
    if px_step > 1:  # shrink pixels to cut bytes; bounds are geographic, unchanged
        out = out[::px_step, ::px_step, :]

    buf = io.BytesIO()
    mimg.imsave(buf, out, format="png")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return uri, [[south, west], [north, east]]


def _fill_nearest(a):
    """Replace NaNs with the nearest non-NaN value (so coastal interpolation
    isn't dragged toward land/zero)."""
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _render_field(field, lats, lons, mask, cmap, norm, alpha_fn, px_step=1):
    """Color ``field`` (on lats/lons) onto the finer land mask grid and return a
    masked PNG overlay. ``mask`` is (mlats, mlons, mwater)."""
    mlats, mlons, mwater = mask
    rgi = RegularGridInterpolator((lats, lons), _fill_nearest(field),
                                  bounds_error=False, fill_value=np.nan)
    lon2d, lat2d = np.meshgrid(mlons, mlats)
    fine = rgi(np.column_stack([lat2d.ravel(), lon2d.ravel()])).reshape(lat2d.shape)
    rgba = cmap(norm(np.ma.masked_invalid(fine)))
    a = alpha_fn(fine)
    a[~mwater] = 0.0
    a[np.isnan(fine)] = 0.0
    rgba[..., 3] = a
    return _png_overlay(mlats, mlons, rgba, px_step=px_step)


def render_prob_png(prob, lats, lons, mask, px_step=1):
    """Probability heatmap (viridis, value-ramped alpha, finer-mask coastline)."""
    def alpha(f):
        a = np.clip(np.nan_to_num(f, nan=0.0), 0, 1)
        return 0.85 * a ** 0.7
    return _render_field(prob, lats, lons, mask, matplotlib.colormaps["viridis"],
                         Normalize(0, 1), alpha, px_step=px_step)


def render_sparkline(series):
    if len(series) < 2:
        return ""
    xs = [dt.date.fromisoformat(d).toordinal() for d, _ in series]
    ys = [la for _, la in series]
    fig, ax = plt.subplots(figsize=(2.9, 0.7), dpi=120)
    ax.plot(xs, ys, color="#08519c", lw=1.6, marker="o", ms=2.5)
    ax.fill_between(xs, ys, min(ys), color="#08519c", alpha=0.10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(0.06)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


MOBILE_CSS = """
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=3">
<style>
#srt-panel{position:fixed;top:12px;left:12px;z-index:9999;
  background:rgba(255,255,255,.96);padding:11px 13px;border-radius:10px;
  box-shadow:0 2px 9px rgba(0,0,0,.28);font-family:system-ui,sans-serif;
  font-size:13px;line-height:1.45;max-width:322px;}
#srt-panel .srt-hd{display:flex;justify-content:space-between;align-items:center;
  gap:8px;cursor:pointer;}
#srt-toggle{border:none;background:#eef2f8;border-radius:6px;padding:1px 9px;
  cursor:pointer;font-size:14px;line-height:1.2;color:#333;}
#srt-panel img{max-width:100%;}
@media (max-width:600px){
  #srt-panel{max-width:74vw;font-size:11px;padding:8px 10px;top:8px;left:8px;
    border-radius:8px;}
}
</style>
"""

TOGGLE_JS = """
<script>
function srtToggle(){var b=document.getElementById('srt-body'),
 t=document.getElementById('srt-toggle');
 if(b.style.display==='none'){b.style.display='block';t.textContent='–';}
 else{b.style.display='none';t.textContent='+';}}
</script>
"""

# Click-to-inspect: snap to the nearest embedded cell, show the factor breakdown,
# and outline that cell. Placeholders are filled by str.replace in build_map.
INSPECT_JS_TMPL = """
<script>
document.addEventListener('DOMContentLoaded', function(){
  var CELLS = __CELLS__, W = __W__, HALF = __HALF__;
  var LBL = {"temp":"Temperature","near":"Nearshore","front":"Front","sight":"Sightings"};
  var COL = {"temp":[3,4,"°F"],"near":[5,6," km to shore"],"front":[7,8," km to front"],"sight":[9,null,""]};
  var map = __MAP__, hl = null;
  map.on('click', function(e){
    var la=e.latlng.lat, lo=e.latlng.lng, best=null, bd=1e9, i, c, d;
    for(i=0;i<CELLS.length;i++){c=CELLS[i];d=Math.abs(c[0]-la)+Math.abs(c[1]-lo);if(d<bd){bd=d;best=c;}}
    if(!best || Math.abs(best[0]-la)>HALF*2 || Math.abs(best[1]-lo)>HALF*2){return;}
    var keys=Object.keys(W), tot=0, contrib={}, k, kk, sc;
    for(k=0;k<keys.length;k++){kk=keys[k];sc=best[COL[kk][0]];if(sc==null)continue;contrib[kk]=W[kk]*sc;tot+=contrib[kk];}
    var html='<div style="font-family:system-ui,sans-serif;font-size:12.5px;min-width:188px;">'
      +'<b>Fish probability: '+best[2]+'%</b>'
      +'<div style="color:#888;font-size:10.5px;margin:2px 0 5px;">~7-mi cell &middot; '+best[0].toFixed(2)+', '+best[1].toFixed(2)+'</div>'
      +'<div style="font-size:11px;color:#555;margin-bottom:2px;">what it is based on:</div>';
    for(k=0;k<keys.length;k++){kk=keys[k];sc=best[COL[kk][0]];if(sc==null)continue;
      var share=tot>0?Math.round(100*contrib[kk]/tot):0;
      var rawi=COL[kk][1], raw=(rawi!=null&&best[rawi]!=null)?(' &middot; '+best[rawi]+COL[kk][2]):'';
      html+='<div style="display:flex;justify-content:space-between;gap:12px;">'
        +'<span>'+LBL[kk]+' <span style="color:#aaa;">('+sc.toFixed(2)+raw+')</span></span>'
        +'<b>'+share+'%</b></div>';}
    html+='<div style="font-size:9.5px;color:#aaa;margin-top:5px;">share of weighted blend &middot; relative suitability, not calibrated probability</div></div>';
    if(hl){map.removeLayer(hl);}
    hl=L.rectangle([[best[0]-HALF,best[1]-HALF],[best[0]+HALF,best[1]+HALF]],{color:'#111',weight:1,fill:false,interactive:false}).addTo(map);
    L.popup({maxWidth:280}).setLatLng([best[0],best[1]]).setContent(html).openOn(map);
  });
});
</script>
"""

# Show the color key for whichever heatmap is currently visible.
KEY_JS_TMPL = """
<script>
document.addEventListener('DOMContentLoaded', function(){
  var map = __MAP__;
  function show(id,on){var el=document.getElementById(id); if(el){el.style.display=on?'block':'none';}}
  map.on('overlayadd', function(e){
    if(/SST/.test(e.name)) show('key-sst',true);
    if(/probability/i.test(e.name)) show('key-prob',true);
  });
  map.on('overlayremove', function(e){
    if(/SST/.test(e.name)) show('key-sst',false);
    if(/probability/i.test(e.name)) show('key-prob',false);
  });
});
</script>
"""


def _front_geojson(front_lines):
    """50F front lines -> GeoJSON FeatureCollection (coords are [lon, lat])."""
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "LineString",
                      "coordinates": [[lon, lat] for lat, lon in line]}}
        for line in front_lines]}


# Bottom time-slider: scrub the probability overlay + 50F front across N days.
# The front layer is created here (in JS) so re-styling survives day swaps.
SLIDER_JS_TMPL = """
<div id="srt-slider" style="position:fixed;bottom:14px;left:50%;transform:translateX(-50%);
  z-index:9999;background:rgba(255,255,255,.95);padding:7px 12px;border-radius:10px;
  box-shadow:0 2px 9px rgba(0,0,0,.28);font-family:system-ui,sans-serif;font-size:12px;
  display:flex;align-items:center;gap:9px;max-width:92vw;">
  <button id="srt-play" style="border:none;background:#08519c;color:#fff;border-radius:6px;
    padding:3px 10px;cursor:pointer;font-size:12px;white-space:nowrap;">&#9654; play</button>
  <input id="srt-range" type="range" min="0" max="1" value="1" step="1"
    style="width:44vw;max-width:360px;">
  <span id="srt-date" style="font-weight:700;min-width:88px;text-align:right;">&nbsp;</span>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var URIS=__URIS__, DATES=__DATES__, FRONTS=__FRONTS__, ov=__OVERLAY__, map=__MAP__;
  var n=URIS.length, timer=null;
  var fl=L.geoJSON(FRONTS[n-1], {style:function(){return {color:'#08306b',weight:4,opacity:0.9};}}).addTo(map);
  var rng=document.getElementById('srt-range'), lab=document.getElementById('srt-date'),
      btn=document.getElementById('srt-play');
  rng.max=n-1; rng.value=n-1;
  function setFrame(i){
    i=Math.max(0,Math.min(n-1,i)); rng.value=i;
    if(ov&&ov.setUrl){ov.setUrl(URIS[i]);}
    fl.clearLayers(); fl.addData(FRONTS[i]); lab.textContent=DATES[i];
  }
  function stop(){ if(timer){clearInterval(timer);timer=null;} btn.innerHTML='&#9654; play'; }
  function play(){
    if(timer){stop();return;}
    btn.innerHTML='&#10073;&#10073; pause';
    var i=parseInt(rng.value,10); if(i>=n-1){i=-1;}
    timer=setInterval(function(){ i++; if(i>n-1){stop();return;} setFrame(i); }, 650);
  }
  rng.addEventListener('input', function(){ stop(); setFrame(parseInt(rng.value,10)); });
  btn.addEventListener('click', play);
  setFrame(n-1);
});
</script>
"""


def _sighting_popup_html(s):
    parts = ["<b>Striped bass</b> (iNaturalist)"]
    if s.date:
        parts.append(f"&#128197; {s.date}")
    if s.sst_c is not None:
        parts.append(f"&#127777;&#65039; water here now: {f_temp(s.sst_c):.0f}&deg;F")
    if s.place:
        parts.append(f"&#128205; {s.place}")
    if s.quality:
        parts.append(f"grade: {s.quality.replace('_', ' ')}")
    if s.observer:
        parts.append(f"by: {s.observer}")
    if s.url:
        parts.append(f"<a href='{s.url}' target='_blank'>view observation</a>")
    return "<br>".join(parts)


def _record_popup_html(r):
    parts = ["<b>Striped bass</b> (record)",
             f"&#128197; {r.date or r.year or 'date n/a'}"]
    if r.sst_c is not None:
        parts.append(f"&#127777;&#65039; water here now: {f_temp(r.sst_c):.0f}&deg;F")
    if r.state:
        parts.append(f"&#128205; {r.state}")
    if r.basis:
        parts.append(f"type: {r.basis.replace('_', ' ').lower()}")
    if r.recorder:
        parts.append(f"by: {r.recorder}")
    if r.source:
        parts.append(f"source: {r.source}")
    if r.uncertainty_m:
        parts.append(f"&plusmn;{r.uncertainty_m:.0f} m precision")
    if r.url:
        parts.append(f"<a href='{r.url}' target='_blank'>source record</a>")
    return "<br>".join(parts)


def _panel_html(status, date_str, sparkline_uri, used, n_inat, n_gbif,
                vmin_f, vmax_f):
    spark = (f'<img src="{sparkline_uri}" style="margin:5px 0 1px;">'
             f'<div style="color:#888;font-size:10px;">50°F leading-edge latitude '
             f'(up = moving north)</div>' if sparkline_uri else "")
    best = (f'<div style="margin-top:4px;">&#11088; {status.best_text}</div>'
            if status.best_text else "")
    factor_list = ", ".join(PROB_LABELS[k] for k in used)
    # Two keys; JS shows the one matching the visible heatmap (prob on by default).
    key_prob = (
        '<div id="key-prob">'
        '<div style="font-size:11px;">Fish probability'
        '<span style="display:inline-block;width:118px;height:10px;border-radius:3px;'
        'background:linear-gradient(to right,#440154,#31688e,#35b779,#fde725);'
        'vertical-align:middle;margin-left:4px;"></span></div>'
        '<div style="font-size:10px;color:#888;width:150px;display:flex;'
        'justify-content:space-between;margin-left:58px;"><span>0%</span>'
        '<span>100%</span></div>'
        '<div style="font-size:9.5px;color:#aaa;">relative likelihood, vs today\'s peak</div>'
        '</div>')
    key_sst = (
        '<div id="key-sst" style="display:none;">'
        '<div style="font-size:11px;">Water temp'
        '<span style="display:inline-block;width:118px;height:10px;border-radius:3px;'
        'background:linear-gradient(to right,#2c7bb6,#ffffbf,#d7191c);'
        'vertical-align:middle;margin-left:4px;"></span></div>'
        f'<div style="font-size:10px;color:#888;width:162px;display:flex;'
        f'justify-content:space-between;margin-left:46px;"><span>{vmin_f:.0f}°F</span>'
        f'<span>50°F</span><span>{vmax_f:.0f}°F</span></div></div>')
    return f"""
    <div id="srt-panel">
      <div class="srt-hd" onclick="srtToggle()">
        <span style="font-weight:700;font-size:14.5px;">&#127907; Striped Bass Run</span>
        <button id="srt-toggle" onclick="event.stopPropagation();srtToggle()">&#8211;</button>
      </div>
      <div id="srt-body">
        <div style="color:#666;font-size:11px;margin:2px 0 6px;">SST {date_str}</div>
        <div>&#128205; {status.front_text}</div>
        <div style="margin-top:3px;">&#127968; {status.home_text}</div>
        <div style="margin-top:3px;">&#10145;&#65039; {status.advance_text}</div>
        <div style="margin-top:5px;padding:6px 8px;background:#eef5fb;border-radius:6px;
          font-weight:600;">&#9203; {status.eta_text}</div>
        {best}
        {spark}
        <hr style="border:none;border-top:1px solid #eee;margin:8px 0 7px;">
        {key_prob}
        {key_sst}
        <div style="font-size:10px;color:#888;margin-top:3px;">factors: {factor_list}</div>
        <div style="font-size:11px;margin-top:5px;">Buoys
          <span style="color:#2c7fb8;">&#9679;</span>&lt;50
          <span style="color:#fdae61;">&#9679;</span>50&ndash;55
          <span style="color:#d7191c;">&#9679;</span>55+°F</div>
        <div style="font-size:10.5px;color:#888;margin-top:4px;">Toggle SST, sightings
          ({n_inat}) &amp; records ({n_gbif}) at top-right.</div>
        <div style="font-size:10px;color:#aaa;margin-top:5px;">Relative likelihood
          (habitat suitability) &mdash; <i>not</i> calibrated probability or live
          fish telemetry.</div>
      </div>
    </div>"""


def build_map(*, prob_uri, prob_bounds, sst_uri, sst_bounds, sst_vmin_f, sst_vmax_f,
              frame_uris, frame_dates, front_frames, buoys, sightings, records,
              status, sparkline_uri, used, cells, insp_w, half, bbox, date_str,
              region, out_path):
    import folium
    from folium.plugins import MarkerCluster
    from folium.raster_layers import ImageOverlay

    lat_min, lat_max, lon_min, lon_max = bbox
    center, zoom = REGIONS.get(region, REGIONS["full"])
    if center is None:
        center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
    m.get_root().header.add_child(folium.Element(MOBILE_CSS))

    # SST first (toggle), probability on top (headline). The probability overlay
    # is named so the time-slider can swap its image per day via setUrl.
    ImageOverlay(image=sst_uri, bounds=sst_bounds, opacity=0.65,
                 name="SST heatmap", show=False).add_to(m)
    prob_layer = ImageOverlay(image=prob_uri, bounds=prob_bounds, opacity=1.0,
                              name="Fish probability", show=True)
    prob_layer.add_to(m)
    # The 50F front is drawn and scrubbed by the time-slider JS (added below).

    fg_buoy = folium.FeatureGroup(name=f"Buoy water temp ({len(buoys)})", show=True)
    for b in buoys:
        f = f_temp(b.wtmp_c)
        popup = folium.Popup(f"<b>Buoy {b.bid}</b> &mdash; {b.name}<br>"
                             f"<b>{b.wtmp_c:.1f}&deg;C / {f:.0f}&deg;F</b><br>{b.when}",
                             max_width=250)
        folium.CircleMarker([b.lat, b.lon], radius=9, color="#222", weight=1,
                            fill=True, fill_color=temp_color(b.wtmp_c),
                            fill_opacity=0.95, popup=popup,
                            tooltip=f"{f:.0f}°F").add_to(fg_buoy)
    fg_buoy.add_to(m)

    fg_inat = folium.FeatureGroup(name=f"Recent sightings ({len(sightings)})", show=False)
    cl_i = MarkerCluster().add_to(fg_inat)
    for s in sightings:
        folium.CircleMarker([s.lat, s.lon], radius=5, color="#2ca25f", weight=1,
                            fill=True, fill_color="#2ca25f", fill_opacity=0.85,
                            popup=folium.Popup(_sighting_popup_html(s), max_width=260)
                            ).add_to(cl_i)
    fg_inat.add_to(m)

    fg_gbif = folium.FeatureGroup(name=f"Seasonal records ({len(records)})", show=False)
    cl_g = MarkerCluster().add_to(fg_gbif)
    for r in records:
        folium.CircleMarker([r.lat, r.lon], radius=3, color="#888", weight=0,
                            fill=True, fill_color="#888", fill_opacity=0.5,
                            popup=folium.Popup(_record_popup_html(r), max_width=260)
                            ).add_to(cl_g)
    fg_gbif.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    m.get_root().html.add_child(folium.Element(_panel_html(
        status, date_str, sparkline_uri, used, len(sightings), len(records),
        sst_vmin_f, sst_vmax_f)))
    m.get_root().html.add_child(folium.Element(TOGGLE_JS))
    inspect_js = (INSPECT_JS_TMPL
                  .replace("__CELLS__", json.dumps(cells))
                  .replace("__W__", json.dumps(insp_w))
                  .replace("__HALF__", str(half))
                  .replace("__MAP__", m.get_name()))
    m.get_root().html.add_child(folium.Element(inspect_js))
    m.get_root().html.add_child(folium.Element(
        KEY_JS_TMPL.replace("__MAP__", m.get_name())))
    slider_js = (SLIDER_JS_TMPL
                 .replace("__URIS__", json.dumps(frame_uris))
                 .replace("__DATES__", json.dumps(frame_dates))
                 .replace("__FRONTS__", json.dumps(front_frames))
                 .replace("__OVERLAY__", prob_layer.get_name())
                 .replace("__MAP__", m.get_name()))
    m.get_root().html.add_child(folium.Element(slider_js))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    m.save(out_path)
    print(f"[map] wrote {out_path}", file=sys.stderr)
    return out_path


# --- cli --------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description="Track the striped bass coastal run.")
    p.add_argument("--date", help="SST date YYYY-MM-DD (default: latest available)")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"),
                   default=list(DEFAULT_BBOX), help="data bounding box")
    p.add_argument("--region", choices=sorted(REGIONS), default="maine",
                   help="initial map viewport (default: maine)")
    p.add_argument("--home-lat", type=float, default=43.6)
    p.add_argument("--home-lon", type=float, default=-70.1)
    p.add_argument("--home-name", default="Portland, ME")
    p.add_argument("--days", type=int, default=30, help="iNaturalist lookback (days)")
    p.add_argument("--timeline-days", type=int, default=14,
                   help="days in the scrub-back timeline (incl. latest)")
    p.add_argument("--stride", type=float, default=0.05, help="SST spacing (deg)")
    p.add_argument("--bait", action="store_true",
                   help="add experimental chlorophyll factor (data source is aging)")
    p.add_argument("--out", help="output HTML path")
    args = p.parse_args(argv)

    bbox = tuple(args.bbox)
    print("[sst] fetching latest SST grid...", file=sys.stderr)
    lats, lons, grid, date_str = fetch_sst(bbox, date=args.date, stride_deg=args.stride)
    mask = fetch_land_mask(bbox)
    front_50 = keep_longest(extract_front(lats, lons, grid, F50_C), n=2)
    sst_uri, sst_bounds, sst_vmin_f, sst_vmax_f = render_sst_png(grid, lats, lons, mask)
    buoys = fetch_buoys(bbox)
    sightings = fetch_sightings(bbox, days=args.days)
    records = fetch_gbif_seasonal(bbox)
    chl_grid = fetch_chl(bbox, lats, lons) if args.bait else None

    # Enrich each point with the current SST sampled at its location.
    for r in records:
        r.sst_c = _sample_grid(lats, lons, grid, r.lat, r.lon)
    for s in sightings:
        s.sst_c = _sample_grid(lats, lons, grid, s.lat, s.lon)
    sight_pts = [(r.lat, r.lon) for r in records] + [(s.lat, s.lon) for s in sightings]

    # Static factors (constant day to day) computed once and reused per frame.
    near = nearshore_factor(grid, lats, lons)
    sight = sighting_density(grid, lats, lons, sight_pts)

    # Per-day blended probability + 50F front for the last N days (incl. latest),
    # then normalize all days to one shared peak so colors compare across time.
    base = dt.date.fromisoformat(date_str)
    raw, used, wnorm = [], [], {}
    for d in range(args.timeline_days - 1, -1, -1):
        if d == 0:
            g, ds, fr = grid, date_str, front_50
        else:
            day = (base - dt.timedelta(days=d)).isoformat()
            try:
                _, _, g, ds = fetch_sst(bbox, date=day, stride_deg=args.stride)
            except Exception as e:
                print(f"[timeline] skip {day}: {e}", file=sys.stderr)
                continue
            fr = keep_longest(extract_front(lats, lons, g, F50_C), n=2)
        blended, used, _detail, wnorm = compute_probability(
            g, lats, lons, fr, sight_pts, chl_grid if d == 0 else None,
            PROB_WEIGHTS, near=near, sight=sight, normalize=False)
        raw.append((ds, blended, fr, _detail))
    print(f"[timeline] {len(raw)} daily frames", file=sys.stderr)

    gmax = max((np.nanmax(b) for _, b, _, _ in raw if np.isfinite(np.nanmax(b))),
               default=1.0) or 1.0
    frames = [(ds, np.clip(b / gmax, 0, 1), fr, det) for ds, b, fr, det in raw]
    # Latest frame full-res (it's the default view + drives click detail); past
    # frames at half pixel resolution to keep the file light. Bounds are identical.
    rendered = [render_prob_png(pf, lats, lons, mask,
                                px_step=(1 if i == len(frames) - 1 else 2))
                for i, (_, pf, _, _) in enumerate(frames)]
    frame_uris = [u for u, _ in rendered]
    prob_bounds = rendered[-1][1]
    frame_dates = [ds for ds, _, _, _ in frames]
    front_frames = [_front_geojson(fr) for _, _, fr, _ in frames]

    _, prob_latest, front_latest, detail_latest = frames[-1]
    cells, half = build_inspection_cells(lats, lons, grid, prob_latest, detail_latest, mask)
    insp_w = {k: round(wnorm[k], 3) for k in ("temp", "near", "front", "sight")
              if k in wnorm}
    print(f"[inspect] {len(cells)} clickable cells | {len(frames)} timeline frames",
          file=sys.stderr)

    home = (args.home_lat, args.home_lon, args.home_name)
    history = [(ds, fr) for ds, _, fr, _ in frames]
    status = compute_status(buoys, history, front_latest, grid, lats, lons, home, date_str)
    status.best_text = best_odds_text(prob_latest, lats, lons, buoys, home)
    print("[status] " + status.summary + " | " + status.best_text, file=sys.stderr)
    sparkline_uri = render_sparkline(status.series)

    out_path = args.out or f"striper_run_{date_str}.html"
    build_map(prob_uri=frame_uris[-1], prob_bounds=prob_bounds, sst_uri=sst_uri,
              sst_bounds=sst_bounds, sst_vmin_f=sst_vmin_f, sst_vmax_f=sst_vmax_f,
              frame_uris=frame_uris, frame_dates=frame_dates, front_frames=front_frames,
              buoys=buoys, sightings=sightings, records=records, status=status,
              sparkline_uri=sparkline_uri, used=used, cells=cells, insp_w=insp_w,
              half=half, bbox=bbox, date_str=date_str, region=args.region,
              out_path=out_path)
    print(out_path)


if __name__ == "__main__":
    main()
