#!/usr/bin/env python3
"""Striped Bass Run Tracker.

Estimates where the annual striped bass (Morone saxatilis) coastal migration
front is along the US East Coast, with a focus on the Gulf of Maine, by
combining free, public, near-real-time sources:

  1. Sea-surface temperature (NOAA/JPL MUR SST via ERDDAP). The run tracks the
     ~50 deg F water, so the 50 deg F (10 deg C) isotherm marks the leading edge.
     A multi-day trail of that isotherm shows the front advancing day by day.
  2. Buoy water temperatures (NOAA NDBC, incl. NERACOOS Gulf of Maine moorings)
     -- real, daily, point-truth where satellite SST is coarse nearshore.
  3. Recent citizen-science sightings (iNaturalist) and a broader historical
     record set (GBIF) to show where bass actually turn up.

Output is an interactive Folium/Leaflet HTML map. This maps the AGGREGATE run
front, not individual fish -- there is no public live fish-telemetry feed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import os
import sys
from dataclasses import dataclass
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")  # headless: we only use the contouring engine, never a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

# --- constants --------------------------------------------------------------

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csv"
INAT_BASE = "https://api.inaturalist.org/v1/observations"
GBIF_BASE = "https://api.gbif.org/v1/occurrence/search"
NDBC_REALTIME = "https://www.ndbc.noaa.gov/data/realtime2/{}.txt"
TAXON = "Morone saxatilis"
MUR_RES_DEG = 0.01  # native MUR grid resolution

# Identify ourselves to the public APIs (iNaturalist requests a User-Agent).
USER_AGENT = "striper-run-tracker (+https://github.com/bmills23/striper-run-tracker)"
HEADERS = {"User-Agent": USER_AGENT}

F50_C = (50 - 32) * 5 / 9  # 10.0 C  -- leading edge of the run
F55_C = (55 - 32) * 5 / 9  # ~12.78 C -- core comfort band

# Default coverage: Cape Hatteras -> Down East Maine (lat_max 45 includes the
# eastern Gulf of Maine buoys), the full span of the coastal run.
DEFAULT_BBOX = (35.0, 45.0, -77.0, -66.0)  # lat_min, lat_max, lon_min, lon_max

# Initial map viewport presets (data is always coast-wide; this is just zoom).
REGIONS = {
    "maine": ([43.6, -68.8], 7),
    "newengland": ([42.0, -70.0], 7),
    "full": (None, 6),  # None center -> use bbox center
}

# Curated NOAA NDBC buoys with water-temp sensors along the run, north->south.
# (name, lat, lon). Dead/MM buoys are skipped at runtime. NERACOOS Gulf of
# Maine moorings appear here under their NDBC ids (E01=44032, I01=44034, etc.).
BUOYS = {
    "44027": ("Jonesport, ME (Down East)", 44.273, -67.314),
    "44034": ("Eastern Maine Shelf (NERACOOS I01)", 44.105, -68.108),
    "44032": ("Central Maine Shelf (NERACOOS E01)", 43.716, -69.355),
    "44007": ("Portland, ME", 43.525, -70.141),
    "44098": ("Jeffreys Ledge", 42.798, -70.168),
    "44013": ("Boston, MA", 42.346, -70.651),
    "44029": ("Massachusetts Bay (NERACOOS A01)", 42.318, -70.566),
    "44090": ("Cape Cod Bay, MA", 41.840, -70.329),
    "44008": ("Nantucket Shoals", 40.504, -69.248),
    "44017": ("Montauk Point, NY", 40.694, -72.048),
    "44025": ("Long Island (off Islip), NY", 40.251, -73.164),
    "44065": ("New York Harbor Entrance", 40.369, -73.703),
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


# --- sea-surface temperature ------------------------------------------------


def fetch_sst(bbox, date=None, stride_deg=0.05, timeout=120):
    """Fetch a strided MUR SST grid (deg C) for the bounding box.

    Returns (lats, lons, grid, date_str) where grid[i, j] is the SST at
    lats[i], lons[j] (NaN over land). ``date`` None -> latest available day.
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    stride = max(1, round(stride_deg / MUR_RES_DEG))
    time_sel = "[last]" if date is None else f"[({date}T09:00:00Z)]"
    selector = (
        f"analysed_sst{time_sel}"
        f"[({lat_min}):{stride}:({lat_max})]"
        f"[({lon_min}):{stride}:({lon_max})]"
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


def fetch_sst_history(bbox, days, end_date, stride_deg=0.1):
    """Front trail: 50 deg F isotherm for each of the ``days`` days before
    ``end_date``. Returns [(date_str, lines), ...] oldest-first."""
    base = dt.date.fromisoformat(end_date)
    out = []
    for d in range(days, 0, -1):
        day = (base - dt.timedelta(days=d)).isoformat()
        try:
            lats, lons, grid, ds = fetch_sst(bbox, date=day, stride_deg=stride_deg)
            out.append((ds, extract_front(lats, lons, grid, F50_C)))
        except Exception as e:  # missing day, transient ERDDAP error -> skip
            print(f"[history] skip {day}: {e}", file=sys.stderr)
    print(f"[history] {len(out)}/{days} prior days of 50F front", file=sys.stderr)
    return out


# --- buoys ------------------------------------------------------------------


def _latest_wtmp(body):
    """Return (water_temp_C, 'YYYY-MM-DD HH:MMZ') for the most recent valid
    reading in an NDBC realtime2 file, or (None, None)."""
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
            body = requests.get(
                NDBC_REALTIME.format(bid), headers=HEADERS, timeout=timeout
            ).text
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
            place=r.get("place_guess") or "",
            url=r.get("uri") or "",
            quality=r.get("quality_grade") or "",
        ))
    print(f"[sightings] {len(out)} recent iNaturalist (last {days}d)", file=sys.stderr)
    return out


def fetch_gbif(bbox, years=4, timeout=60, max_records=900):
    """Broader historical occurrence records from GBIF (aggregates iNaturalist
    research-grade, museum, and survey/marine datasets incl. OBIS feeds)."""
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


# --- rendering --------------------------------------------------------------


def temp_color(c):
    """Buoy marker color by how the run feels it: cold / arriving / prime."""
    if c < F50_C:
        return "#2c7fb8"  # < 50F: too cold, run not here
    if c < F55_C:
        return "#fdae61"  # 50-55F: leading edge / arriving
    return "#d7191c"      # 55F+: prime


def _legend_html(date_str, days, n_inat, n_buoys, n_gbif, n_hist):
    return f"""
    <div style="position: fixed; top: 12px; left: 50px; z-index: 9999;
        background: white; padding: 10px 14px; border-radius: 8px;
        box-shadow: 0 1px 5px rgba(0,0,0,.3); font-family: sans-serif;
        font-size: 12.5px; max-width: 330px; line-height: 1.5;">
      <b>&#127907; Striped Bass Run Tracker</b> &mdash; SST {date_str}<br>
      <span style="color:#2c7fb8;font-weight:bold;">&#9473;&#9473;</span>
        &asymp;50&deg;F isotherm &mdash; leading edge<br>
      <span style="color:#41b6c4;font-weight:bold;">&#9473;&#9473;</span>
        &asymp;55&deg;F isotherm &mdash; core band<br>
      <span style="color:#08519c;">&#9482;&#9482;</span>
        50&deg;F front trail, last {n_hist}d (faint=older)<br>
      Buoys ({n_buoys}):
        <span style="color:#2c7fb8;">&#9679;</span>&lt;50
        <span style="color:#fdae61;">&#9679;</span>50&ndash;55
        <span style="color:#d7191c;">&#9679;</span>55+&deg;F<br>
      <span style="color:#2ca25f;font-weight:bold;">&#9679;</span>
        iNaturalist, last {days}d ({n_inat}) &nbsp;
      <span style="color:#888;font-weight:bold;">&#9679;</span>
        GBIF records ({n_gbif})<br>
      <small style="color:#777;">Temperature-proxy + sightings &mdash;
        <i>not</i> live fish telemetry.</small>
    </div>"""


def build_map(*, front_50, front_55, history, buoys, sightings, records,
              bbox, date_str, days, region, out_path):
    import folium
    from folium.plugins import MarkerCluster

    lat_min, lat_max, lon_min, lon_max = bbox
    center, zoom = REGIONS.get(region, REGIONS["full"])
    if center is None:
        center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    # Front trail (oldest faint -> newest darker), drawn first so it sits under.
    fg_hist = folium.FeatureGroup(name=f"50F front trail ({len(history)}d)", show=True)
    n = len(history)
    for i, (_, lines) in enumerate(history):
        op = 0.12 + 0.38 * (i / max(1, n - 1))
        for line in lines:
            folium.PolyLine(line, color="#08519c", weight=2, opacity=op).add_to(fg_hist)
    fg_hist.add_to(m)

    fg55 = folium.FeatureGroup(name="~55F isotherm (core)", show=True)
    for line in front_55:
        folium.PolyLine(line, color="#41b6c4", weight=3, opacity=0.85).add_to(fg55)
    fg55.add_to(m)

    fg50 = folium.FeatureGroup(name="~50F isotherm (leading edge)", show=True)
    for line in front_50:
        folium.PolyLine(line, color="#2c7fb8", weight=5, opacity=0.95).add_to(fg50)
    fg50.add_to(m)

    # Historical records (GBIF) -- clustered, on but visually subtle.
    fg_gbif = folium.FeatureGroup(name=f"Historical records ({len(records)})", show=True)
    cl_g = MarkerCluster().add_to(fg_gbif)
    for r in records:
        popup = folium.Popup(
            f"<b>Striped bass</b> ({r.year})<br>{r.source}<br>"
            f"<a href='{r.url}' target='_blank'>GBIF</a>" if r.url else "GBIF record",
            max_width=240)
        folium.CircleMarker([r.lat, r.lon], radius=3, color="#888", weight=0,
                            fill=True, fill_color="#888", fill_opacity=0.55,
                            popup=popup).add_to(cl_g)
    fg_gbif.add_to(m)

    # Recent iNaturalist sightings -- clustered, prominent.
    fg_inat = folium.FeatureGroup(name=f"Recent sightings ({len(sightings)})", show=True)
    cl_i = MarkerCluster().add_to(fg_inat)
    for s in sightings:
        popup = folium.Popup(
            f"<b>Striped bass</b><br>{s.date}<br>{s.place}<br><i>{s.quality}</i>"
            f"<br><a href='{s.url}' target='_blank'>iNaturalist</a>", max_width=250)
        folium.CircleMarker([s.lat, s.lon], radius=5, color="#2ca25f", weight=1,
                            fill=True, fill_color="#2ca25f", fill_opacity=0.85,
                            popup=popup).add_to(cl_i)
    fg_inat.add_to(m)

    # Live buoy water temps -- the real Gulf of Maine signal.
    fg_buoy = folium.FeatureGroup(name=f"Buoy water temp ({len(buoys)})", show=True)
    for b in buoys:
        f = b.wtmp_c * 9 / 5 + 32
        popup = folium.Popup(
            f"<b>Buoy {b.bid}</b> &mdash; {b.name}<br>"
            f"<b>{b.wtmp_c:.1f}&deg;C / {f:.0f}&deg;F</b><br>{b.when}", max_width=250)
        folium.CircleMarker([b.lat, b.lon], radius=9, color="#222", weight=1,
                            fill=True, fill_color=temp_color(b.wtmp_c),
                            fill_opacity=0.95, popup=popup,
                            tooltip=f"{f:.0f}°F").add_to(fg_buoy)
    fg_buoy.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html(
        date_str, days, len(sightings), len(buoys), len(records), len(history))))

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
    p.add_argument("--days", type=int, default=30, help="iNaturalist lookback (days)")
    p.add_argument("--history-days", type=int, default=10,
                   help="prior days of 50F front trail (0 to disable)")
    p.add_argument("--gbif-years", type=int, default=4,
                   help="historical GBIF record window (years; 0 to disable)")
    p.add_argument("--stride", type=float, default=0.05, help="SST spacing (deg)")
    p.add_argument("--out", help="output HTML path")
    args = p.parse_args(argv)

    bbox = tuple(args.bbox)
    print("[sst] fetching latest SST grid...", file=sys.stderr)
    lats, lons, grid, date_str = fetch_sst(bbox, date=args.date, stride_deg=args.stride)
    front_50 = extract_front(lats, lons, grid, F50_C)
    front_55 = extract_front(lats, lons, grid, F55_C)
    history = (fetch_sst_history(bbox, args.history_days, date_str)
               if args.history_days > 0 else [])
    buoys = fetch_buoys(bbox)
    sightings = fetch_sightings(bbox, days=args.days)
    records = fetch_gbif(bbox, years=args.gbif_years) if args.gbif_years > 0 else []

    out_path = args.out or f"striper_run_{date_str}.html"
    build_map(front_50=front_50, front_55=front_55, history=history, buoys=buoys,
              sightings=sightings, records=records, bbox=bbox, date_str=date_str,
              days=args.days, region=args.region, out_path=out_path)
    print(out_path)


if __name__ == "__main__":
    main()
