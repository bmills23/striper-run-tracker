#!/usr/bin/env python3
"""Striped Bass Run Tracker.

Estimates where the annual striped bass (Morone saxatilis) migration front is
along the US East Coast by combining two free, public, near-real-time sources:

  1. Sea-surface temperature (NOAA/JPL MUR SST via ERDDAP). The run tracks the
     ~50 deg F water as it moves north in spring and south in fall, so the 50 deg F
     (10 deg C) isotherm is a good proxy for the leading edge of the run.
  2. Citizen-science sightings (iNaturalist) to ground-truth the proxy.

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
INAT_TAXON = "Morone saxatilis"
MUR_RES_DEG = 0.01  # native MUR grid resolution

# Identify ourselves to the public APIs (iNaturalist requests a User-Agent).
USER_AGENT = "striper-run-tracker (+https://github.com/; public-data migration map)"
HEADERS = {"User-Agent": USER_AGENT}

F50_C = (50 - 32) * 5 / 9  # 10.0 C  -- leading edge of the run
F55_C = (55 - 32) * 5 / 9  # ~12.78 C -- core comfort band

# Default coverage: Cape Hatteras -> Maine, the heart of the coastal run.
DEFAULT_BBOX = (35.0, 44.0, -77.0, -66.0)  # lat_min, lat_max, lon_min, lon_max


@dataclass
class Sighting:
    lat: float
    lon: float
    date: str
    place: str
    url: str
    quality: str


# --- data fetching ----------------------------------------------------------


def fetch_sst(bbox, date=None, stride_deg=0.05, timeout=120):
    """Fetch a strided MUR SST grid (deg C) for the bounding box.

    Returns (lats, lons, grid, date_str) where grid[i, j] is the SST at
    lats[i], lons[j] (NaN over land). If ``date`` is None, the latest available
    day is used.
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
    print(f"[sst] GET {url}", file=sys.stderr)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    reader = csv.reader(io.StringIO(resp.text))
    rows = list(reader)
    # rows[0] = column names, rows[1] = units, rows[2:] = data
    data = rows[2:]
    lat_vals, lon_vals, sst_vals, times = [], [], [], []
    for r in data:
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
    print(f"[sst] {grid.shape[0]}x{grid.shape[1]} grid for {date_str} "
          f"(stride {stride*MUR_RES_DEG:.2f} deg)", file=sys.stderr)
    return lats, lons, grid, date_str


def extract_front(lats, lons, grid, level_c, min_vertices=8):
    """Return the ``level_c`` isotherm as a list of [(lat, lon), ...] lines."""
    cs = plt.contour(lons, lats, grid, levels=[level_c])
    lines = []
    for seg in cs.allsegs[0]:  # allsegs[level] -> list of (N, 2) [x=lon, y=lat]
        if len(seg) >= min_vertices:
            lines.append([(float(y), float(x)) for x, y in seg])
    plt.close("all")
    print(f"[front] {level_c:.1f} C ({level_c*9/5+32:.0f} F): "
          f"{len(lines)} segment(s)", file=sys.stderr)
    return lines


def fetch_sightings(bbox, days=30, timeout=60, max_results=200):
    """Fetch recent geotagged striped bass observations from iNaturalist."""
    lat_min, lat_max, lon_min, lon_max = bbox
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    params = {
        "taxon_name": INAT_TAXON,
        "geo": "true",
        "verifiable": "true",
        "per_page": str(max_results),
        "order_by": "observed_on",
        "order": "desc",
        "d1": since,
        "swlat": lat_min, "swlng": lon_min,
        "nelat": lat_max, "nelng": lon_max,
    }
    print(f"[sightings] iNaturalist since {since}", file=sys.stderr)
    resp = requests.get(INAT_BASE, params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", []):
        geo = r.get("geojson") or {}
        coords = geo.get("coordinates")
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
    print(f"[sightings] {len(out)} geotagged in box", file=sys.stderr)
    return out


# --- rendering --------------------------------------------------------------


def _legend_html(date_str, days, n_sightings):
    return f"""
    <div style="position: fixed; top: 12px; left: 50px; z-index: 9999;
        background: white; padding: 10px 14px; border-radius: 8px;
        box-shadow: 0 1px 5px rgba(0,0,0,.3); font-family: sans-serif;
        font-size: 13px; max-width: 340px; line-height: 1.5;">
      <b>&#127907; Striped Bass Run Tracker</b><br>
      SST date: <b>{date_str}</b><br>
      <span style="color:#2c7fb8;font-weight:bold;">&#9473;&#9473;</span>
        &asymp;50&deg;F (10&deg;C) &mdash; leading edge of run<br>
      <span style="color:#41b6c4;font-weight:bold;">&#9473;&#9473;</span>
        &asymp;55&deg;F (12.8&deg;C) &mdash; core comfort band<br>
      <span style="color:#2ca25f;font-weight:bold;">&#9679;</span>
        sightings, last {days}d ({n_sightings})<br>
      <small style="color:#777;">Temperature-proxy estimate + citizen
        sightings &mdash; <i>not</i> live fish telemetry.</small>
    </div>"""


def build_map(front_50, front_55, sightings, bbox, date_str, days, out_path):
    import folium
    from folium.plugins import MarkerCluster

    lat_min, lat_max, lon_min, lon_max = bbox
    center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    fg55 = folium.FeatureGroup(name="~55 deg F (core band)", show=True)
    for line in front_55:
        folium.PolyLine(line, color="#41b6c4", weight=3, opacity=0.8).add_to(fg55)
    fg55.add_to(m)

    fg50 = folium.FeatureGroup(name="~50 deg F (leading edge)", show=True)
    for line in front_50:
        folium.PolyLine(line, color="#2c7fb8", weight=5, opacity=0.9).add_to(fg50)
    fg50.add_to(m)

    fgs = folium.FeatureGroup(name=f"Sightings (last {days}d)", show=True)
    cluster = MarkerCluster().add_to(fgs)
    for s in sightings:
        popup = folium.Popup(
            f"<b>Striped bass</b><br>{s.date}<br>{s.place}<br>"
            f"<i>{s.quality}</i><br><a href='{s.url}' target='_blank'>iNaturalist</a>",
            max_width=250,
        )
        folium.CircleMarker(
            [s.lat, s.lon], radius=5, color="#2ca25f", fill=True,
            fill_color="#2ca25f", fill_opacity=0.8, popup=popup,
        ).add_to(cluster)
    fgs.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(
        folium.Element(_legend_html(date_str, days, len(sightings)))
    )
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
    p.add_argument("--bbox", nargs=4, type=float, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"),
                   default=list(DEFAULT_BBOX), help="bounding box (default: Hatteras->Maine)")
    p.add_argument("--days", type=int, default=30, help="sighting lookback window")
    p.add_argument("--stride", type=float, default=0.05, help="SST sample spacing in degrees")
    p.add_argument("--out", help="output HTML path (default: striper_run_<date>.html)")
    args = p.parse_args(argv)

    bbox = tuple(args.bbox)
    lats, lons, grid, date_str = fetch_sst(bbox, date=args.date, stride_deg=args.stride)
    front_50 = extract_front(lats, lons, grid, F50_C)
    front_55 = extract_front(lats, lons, grid, F55_C)
    sightings = fetch_sightings(bbox, days=args.days)
    out_path = args.out or f"striper_run_{date_str}.html"
    build_map(front_50, front_55, sightings, bbox, date_str, args.days, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
