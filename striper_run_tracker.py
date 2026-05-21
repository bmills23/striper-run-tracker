#!/usr/bin/env python3
"""Striped Bass Run Tracker.

Answers, at a glance, "is the striped bass run near me, and when does it arrive?"
along the US East Coast (focused on the Gulf of Maine), from free public data:

  1. Sea-surface temperature (NOAA/JPL MUR SST via ERDDAP), drawn as a smooth
     heatmap centered on 50 deg F -- the temperature the run tracks -- so the
     leading edge reads as a color break. A single clean 50 deg F line marks it.
  2. Buoy water temperatures (NOAA NDBC, incl. NERACOOS Gulf of Maine moorings)
     -- real, daily, point-truth where satellite SST is coarse nearshore.
  3. A plain-English status panel: where the front is, whether run-temp water has
     reached your area, how fast it is moving, and a rough arrival estimate
     (extrapolated from the 50 deg F line's recent day-by-day advance).
  4. Optional detail layers (recent iNaturalist sightings, broader GBIF records),
     off by default.

Output is a single self-contained interactive HTML map. This maps the AGGREGATE
run front, not individual fish -- there is no public live fish-telemetry feed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import io
import os
import sys
from dataclasses import dataclass
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")  # headless: contouring + image rendering, never a window
import matplotlib.image as mimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

# --- constants --------------------------------------------------------------

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csv"
INAT_BASE = "https://api.inaturalist.org/v1/observations"
GBIF_BASE = "https://api.gbif.org/v1/occurrence/search"
NDBC_REALTIME = "https://www.ndbc.noaa.gov/data/realtime2/{}.txt"
TAXON = "Morone saxatilis"
MUR_RES_DEG = 0.01

USER_AGENT = "striper-run-tracker (+https://github.com/bmills23/striper-run-tracker)"
HEADERS = {"User-Agent": USER_AGENT}

F50_C = (50 - 32) * 5 / 9  # 10.0 C  -- leading edge of the run
F55_C = (55 - 32) * 5 / 9  # ~12.78 C -- core comfort band
MI_PER_DEG_LAT = 69.0

DEFAULT_BBOX = (35.0, 45.0, -77.0, -66.0)  # lat_min, lat_max, lon_min, lon_max

REGIONS = {  # initial viewport only; data is always coast-wide
    "maine": ([43.6, -68.8], 7),
    "newengland": ([42.0, -70.0], 7),
    "full": (None, 6),
}

# Curated NOAA NDBC buoys with water-temp sensors along the run, north->south.
# Dead/MM buoys are skipped at runtime. NERACOOS Gulf of Maine moorings appear
# here under their NDBC ids (E01=44032, I01=44034, A01=44029, etc.).
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


@dataclass
class Record:
    lat: float
    lon: float
    year: int
    source: str
    url: str


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
    series: list  # [(date_str, leading_latitude), ...]
    summary: str


def f_temp(c):
    """Celsius -> Fahrenheit."""
    return c * 9 / 5 + 32


# --- sea-surface temperature ------------------------------------------------


def fetch_sst(bbox, date=None, stride_deg=0.05, timeout=120):
    """Fetch a strided MUR SST grid (deg C). Returns (lats, lons, grid, date_str);
    grid[i, j] is SST at lats[i], lons[j] (NaN over land). date None -> latest."""
    lat_min, lat_max, lon_min, lon_max = bbox
    stride = max(1, round(stride_deg / MUR_RES_DEG))
    time_sel = "[last]" if date is None else f"[({date}T09:00:00Z)]"
    selector = (
        f"analysed_sst{time_sel}"
        f"[({lat_min}):{stride}:({lat_max})][({lon_min}):{stride}:({lon_max})]"
    )
    url = f"{ERDDAP_BASE}?{quote(selector, safe='()[]:.,-')}"
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    rows = list(csv.reader(io.StringIO(resp.text)))[2:]  # drop name + unit rows
    lat_vals, lon_vals, sst_vals, times = [], [], [], []
    for r in rows:
        times.append(r[0])
        lat_vals.append(float(r[1]))
        lon_vals.append(float(r[2]))
        sst_vals.append(np.nan if r[3] in ("", "NaN") else float(r[3]))

    lats = np.array(sorted(set(lat_vals)))
    lons = np.array(sorted(set(lon_vals)))
    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}
    grid = np.full((len(lats), len(lons)), np.nan)
    for la, lo, sst in zip(lat_vals, lon_vals, sst_vals):
        grid[lat_idx[la], lon_idx[lo]] = sst

    date_str = times[0][:10] if times else (date or "unknown")
    return lats, lons, grid, date_str


def extract_front(lats, lons, grid, level_c, min_vertices=8):
    """Return the ``level_c`` isotherm as a list of [(lat, lon), ...] lines."""
    cs = plt.contour(lons, lats, grid, levels=[level_c])
    lines = [
        [(float(y), float(x)) for x, y in seg]
        for seg in cs.allsegs[0]
        if len(seg) >= min_vertices
    ]
    plt.close("all")
    return lines


def _path_len_deg(seg):
    a = np.asarray(seg, float)
    if len(a) < 2:
        return 0.0
    d = np.diff(a, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def keep_longest(lines, n=2):
    """Keep the n longest segments (the coastal front), drop offshore eddy loops."""
    return sorted(lines, key=_path_len_deg, reverse=True)[:n]


def front_leading_latitude(lines, lon_center, halfwidth=1.0):
    """Northern extent of the isotherm within lon_center +/- halfwidth (the
    coastal leading edge), or None."""
    in_band = [lat for line in lines for (lat, lon) in line
               if abs(lon - lon_center) <= halfwidth]
    return max(in_band) if in_band else None


def fetch_sst_history(bbox, days, end_date, stride_deg=0.1):
    """[(date_str, 50F lines), ...] for the ``days`` days before ``end_date``."""
    base = dt.date.fromisoformat(end_date)
    out = []
    for d in range(days, 0, -1):
        day = (base - dt.timedelta(days=d)).isoformat()
        try:
            la, lo, g, ds = fetch_sst(bbox, date=day, stride_deg=stride_deg)
            out.append((ds, extract_front(la, lo, g, F50_C)))
        except Exception as e:  # missing day / transient error -> skip
            print(f"[history] skip {day}: {e}", file=sys.stderr)
    print(f"[history] {len(out)}/{days} prior days", file=sys.stderr)
    return out


def render_sst_png(lats, lons, grid):
    """SST heatmap as a base64 PNG data URI + map bounds.

    Diverging colormap centered on 50 deg F so the run's leading edge is the
    natural blue->yellow->red color break. Land/no-data is transparent.
    """
    valid = grid[~np.isnan(grid)]
    vmin = float(np.floor(valid.min())) if valid.size else 0.0
    vmax = float(np.ceil(valid.max())) if valid.size else 25.0
    vmin = min(vmin, F50_C - 1)
    vmax = max(vmax, F50_C + 1)
    norm = TwoSlopeNorm(vcenter=F50_C, vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps["RdYlBu_r"]

    rgba = cmap(norm(np.ma.masked_invalid(grid)))
    rgba[np.isnan(grid), 3] = 0.0     # transparent land / no data
    rgba = rgba[::-1, :, :]           # row 0 -> north, for ImageOverlay

    buf = io.BytesIO()
    mimg.imsave(buf, rgba, format="png")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    bounds = [[float(lats.min()), float(lons.min())],
              [float(lats.max()), float(lons.max())]]
    return uri, bounds


# --- buoys ------------------------------------------------------------------


def _latest_wtmp(body):
    """(water_temp_C, 'YYYY-MM-DD HH:MMZ') for the most recent valid NDBC reading."""
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
    """Live NDBC water-temp buoys within the bounding box."""
    lat_min, lat_max, lon_min, lon_max = bbox
    out = []
    for bid, (name, lat, lon) in BUOYS.items():
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            continue
        try:
            body = requests.get(NDBC_REALTIME.format(bid), headers=HEADERS,
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
    """Recent geotagged striped bass observations from iNaturalist."""
    lat_min, lat_max, lon_min, lon_max = bbox
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    params = {
        "taxon_name": TAXON, "geo": "true", "verifiable": "true",
        "per_page": str(max_results), "order_by": "observed_on", "order": "desc",
        "d1": since, "swlat": lat_min, "swlng": lon_min,
        "nelat": lat_max, "nelng": lon_max,
    }
    resp = requests.get(INAT_BASE, params=params, headers=HEADERS, timeout=timeout)
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
        out.append(Sighting(
            lat=lat, lon=lon,
            date=r.get("observed_on") or (r.get("time_observed_at") or "")[:10],
            place=r.get("place_guess") or "", url=r.get("uri") or "",
            quality=r.get("quality_grade") or "",
        ))
    print(f"[sightings] {len(out)} recent iNaturalist (last {days}d)", file=sys.stderr)
    return out


def fetch_gbif(bbox, years=4, timeout=60, max_records=900):
    """Broader historical records from GBIF (aggregates iNaturalist research-grade,
    museum, and survey/marine datasets)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    this_year = dt.date.today().year
    out, offset = [], 0
    while len(out) < max_records:
        params = {
            "scientificName": TAXON, "hasCoordinate": "true",
            "decimalLatitude": f"{lat_min},{lat_max}",
            "decimalLongitude": f"{lon_min},{lon_max}",
            "year": f"{this_year - years + 1},{this_year}",
            "limit": "300", "offset": str(offset),
        }
        resp = requests.get(GBIF_BASE, params=params, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        for r in data.get("results", []):
            lat, lon = r.get("decimalLatitude"), r.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            key = r.get("key")
            out.append(Record(
                lat=float(lat), lon=float(lon), year=r.get("year") or 0,
                source=r.get("datasetName") or r.get("institutionCode") or "GBIF",
                url=f"https://www.gbif.org/occurrence/{key}" if key else "",
            ))
        if data.get("endOfRecords") or not data.get("results"):
            break
        offset += 300
    print(f"[gbif] {len(out)} historical records (last {years}y)", file=sys.stderr)
    return out


# --- status computation -----------------------------------------------------


def _sample_grid(lats, lons, grid, lat, lon):
    """Nearest non-NaN SST (deg C) to (lat, lon), searching a small neighborhood
    so a coastal/land cell still resolves to nearby water. None if nothing near."""
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
    """Plain-English run status from buoys (front location, home) and the 50F
    isotherm's day-by-day advance (rate + ETA)."""
    home_lat, home_lon, home_name = home
    home_sst = _sample_grid(lats, lons, grid, home_lat, home_lon)

    # Front location, from buoys (most reliable point-truth).
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

    # Home status.
    if home_sst is None:
        home_text = f"{home_name}: no temperature reading."
    elif home_sst >= F50_C:
        home_text = f"{home_name}: {f_temp(home_sst):.0f}°F — run water present ✓"
    else:
        home_text = f"{home_name}: {f_temp(home_sst):.0f}°F — not there yet."

    # Leading-latitude series (history + today) along the home's coastal band.
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

    # ETA to home.
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
    """Buoy color by how the run feels it: cold / arriving / prime."""
    if c < F50_C:
        return "#2c7fb8"
    if c < F55_C:
        return "#fdae61"
    return "#d7191c"


def render_sparkline(series):
    """Tiny base64 PNG of leading-edge latitude over time (up = moving north)."""
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


def _status_panel_html(status, date_str, sparkline_uri, n_inat, n_gbif):
    spark = (
        f'<img src="{sparkline_uri}" style="width:100%;margin:5px 0 1px;">'
        f'<div style="color:#888;font-size:10px;">50°F leading-edge latitude '
        f'(up = moving north)</div>' if sparkline_uri else "")
    return f"""
    <div style="position: fixed; top: 12px; left: 12px; z-index: 9999;
        background: rgba(255,255,255,.96); padding: 11px 14px; border-radius: 10px;
        box-shadow: 0 2px 9px rgba(0,0,0,.28); font-family: system-ui, sans-serif;
        font-size: 13px; max-width: 322px; line-height: 1.45;">
      <div style="font-weight:700;font-size:14.5px;">&#127907; Striped Bass Run</div>
      <div style="color:#666;font-size:11px;margin-bottom:7px;">SST {date_str}</div>
      <div>&#128205; {status.front_text}</div>
      <div style="margin-top:3px;">&#127968; {status.home_text}</div>
      <div style="margin-top:3px;">&#10145;&#65039; {status.advance_text}</div>
      <div style="margin-top:5px;padding:6px 8px;background:#eef5fb;border-radius:6px;
        font-weight:600;">&#9203; {status.eta_text}</div>
      {spark}
      <hr style="border:none;border-top:1px solid #eee;margin:8px 0 7px;">
      <div style="font-size:11px;">Water temp
        <span style="display:inline-block;width:130px;height:10px;border-radius:3px;
          background:linear-gradient(to right,#2c7bb6,#ffffbf,#d7191c);
          vertical-align:middle;margin-left:4px;"></span></div>
      <div style="font-size:10px;color:#888;width:160px;display:flex;
        justify-content:space-between;margin-left:62px;">
        <span>cold</span><span>50°F</span><span>warm</span></div>
      <div style="font-size:11px;margin-top:5px;">Buoys
        <span style="color:#2c7fb8;">&#9679;</span>&lt;50
        <span style="color:#fdae61;">&#9679;</span>50&ndash;55
        <span style="color:#d7191c;">&#9679;</span>55+°F</div>
      <div style="font-size:10.5px;color:#888;margin-top:4px;">Toggle sightings
        ({n_inat}) &amp; historical records ({n_gbif}) at top-right.</div>
      <div style="font-size:10px;color:#aaa;margin-top:5px;">Temperature-proxy
        estimate &mdash; <i>not</i> live fish telemetry.</div>
    </div>"""


def build_map(*, sst_uri, sst_bounds, front_50, history, buoys, sightings,
              records, status, sparkline_uri, bbox, date_str, region, out_path):
    import folium
    from folium.plugins import MarkerCluster
    from folium.raster_layers import ImageOverlay

    lat_min, lat_max, lon_min, lon_max = bbox
    center, zoom = REGIONS.get(region, REGIONS["full"])
    if center is None:
        center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    ImageOverlay(image=sst_uri, bounds=sst_bounds, opacity=0.65,
                 name="SST heatmap", show=True).add_to(m)

    fg50 = folium.FeatureGroup(name="50°F front (leading edge)", show=True)
    for line in front_50:
        folium.PolyLine(line, color="#08306b", weight=4, opacity=0.9).add_to(fg50)
    fg50.add_to(m)

    fg_buoy = folium.FeatureGroup(name=f"Buoy water temp ({len(buoys)})", show=True)
    for b in buoys:
        f = f_temp(b.wtmp_c)
        popup = folium.Popup(
            f"<b>Buoy {b.bid}</b> &mdash; {b.name}<br>"
            f"<b>{b.wtmp_c:.1f}&deg;C / {f:.0f}&deg;F</b><br>{b.when}", max_width=250)
        folium.CircleMarker([b.lat, b.lon], radius=9, color="#222", weight=1,
                            fill=True, fill_color=temp_color(b.wtmp_c),
                            fill_opacity=0.95, popup=popup,
                            tooltip=f"{f:.0f}°F").add_to(fg_buoy)
    fg_buoy.add_to(m)

    # --- detail layers, off by default ---
    fg_trail = folium.FeatureGroup(name=f"50°F trail, raw ({len(history)}d)", show=False)
    n = len(history)
    for i, (_, lines) in enumerate(history):
        op = 0.10 + 0.35 * (i / max(1, n - 1))
        for line in lines:
            folium.PolyLine(line, color="#08519c", weight=1.5, opacity=op).add_to(fg_trail)
    fg_trail.add_to(m)

    fg_inat = folium.FeatureGroup(name=f"Recent sightings ({len(sightings)})", show=False)
    cl_i = MarkerCluster().add_to(fg_inat)
    for s in sightings:
        popup = folium.Popup(
            f"<b>Striped bass</b><br>{s.date}<br>{s.place}<br><i>{s.quality}</i>"
            f"<br><a href='{s.url}' target='_blank'>iNaturalist</a>", max_width=250)
        folium.CircleMarker([s.lat, s.lon], radius=5, color="#2ca25f", weight=1,
                            fill=True, fill_color="#2ca25f", fill_opacity=0.85,
                            popup=popup).add_to(cl_i)
    fg_inat.add_to(m)

    fg_gbif = folium.FeatureGroup(name=f"Historical records ({len(records)})", show=False)
    cl_g = MarkerCluster().add_to(fg_gbif)
    for r in records:
        popup = folium.Popup(
            (f"<b>Striped bass</b> ({r.year})<br>{r.source}<br>"
             f"<a href='{r.url}' target='_blank'>GBIF</a>") if r.url else "GBIF record",
            max_width=240)
        folium.CircleMarker([r.lat, r.lon], radius=3, color="#888", weight=0,
                            fill=True, fill_color="#888", fill_opacity=0.55,
                            popup=popup).add_to(cl_g)
    fg_gbif.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_status_panel_html(
        status, date_str, sparkline_uri, len(sightings), len(records))))

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
    p.add_argument("--home-lat", type=float, default=43.6, help="your-area latitude")
    p.add_argument("--home-lon", type=float, default=-70.1, help="your-area longitude")
    p.add_argument("--home-name", default="Portland, ME", help="your-area label")
    p.add_argument("--days", type=int, default=30, help="iNaturalist lookback (days)")
    p.add_argument("--history-days", type=int, default=10,
                   help="prior days used for advance rate / sparkline")
    p.add_argument("--gbif-years", type=int, default=4, help="GBIF window (years)")
    p.add_argument("--stride", type=float, default=0.05, help="SST spacing (deg)")
    p.add_argument("--out", help="output HTML path")
    args = p.parse_args(argv)

    bbox = tuple(args.bbox)
    print("[sst] fetching latest SST grid...", file=sys.stderr)
    lats, lons, grid, date_str = fetch_sst(bbox, date=args.date, stride_deg=args.stride)
    front_50 = keep_longest(extract_front(lats, lons, grid, F50_C), n=2)
    sst_uri, sst_bounds = render_sst_png(lats, lons, grid)
    history = (fetch_sst_history(bbox, args.history_days, date_str)
               if args.history_days > 0 else [])
    buoys = fetch_buoys(bbox)
    sightings = fetch_sightings(bbox, days=args.days)
    records = fetch_gbif(bbox, years=args.gbif_years) if args.gbif_years > 0 else []

    home = (args.home_lat, args.home_lon, args.home_name)
    status = compute_status(buoys, history, front_50, grid, lats, lons, home, date_str)
    print("[status] " + status.summary, file=sys.stderr)
    sparkline_uri = render_sparkline(status.series)

    out_path = args.out or f"striper_run_{date_str}.html"
    build_map(sst_uri=sst_uri, sst_bounds=sst_bounds, front_50=front_50,
              history=history, buoys=buoys, sightings=sightings, records=records,
              status=status, sparkline_uri=sparkline_uri, bbox=bbox,
              date_str=date_str, region=args.region, out_path=out_path)
    print(out_path)


if __name__ == "__main__":
    main()
