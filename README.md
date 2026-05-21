# Striped Bass Run Tracker

Estimate where the annual striped bass (*Morone saxatilis*) migration front is
along the US East Coast, right now, using only free public data — and render it
as an interactive map.

## How it works

The coastal run tracks water temperature: bass follow the **~50 °F** water north
in spring and south in fall. So instead of (impossibly) tracking individual
fish, this tool reads that signal and answers, in plain English, **"is the run
near me, and when does it arrive?"**

The headline is a **probability (relative-likelihood) heatmap** — a weighted
blend of the conditions that concentrate bass — plus a **status panel** that
answers, in plain English, front location, whether run-temperature water has
reached your area, how fast it's moving, an arrival estimate, and the best-odds
spot near you. The whole thing is **mobile-friendly** (collapsible panel, touch
map).

| Element | Source | Shown by default |
|---------|--------|------------------|
| **Probability heatmap** (viridis) — weighted blend, normalized 0–1 | computed (see below) | yes |
| **Status panel** (front location, your-area status, advance rate, ETA, best-odds, sparkline) | computed from buoys + SST | yes |
| **50 °F front line** — single clean leading edge (longest segments only) | [NOAA/JPL MUR SST](https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.html) via ERDDAP (~1 km, sampled to ~3.5 mi) | yes |
| **Buoy water temps** — color-coded above/below 50 °F | [NOAA NDBC](https://www.ndbc.noaa.gov/) realtime (incl. NERACOOS Gulf of Maine moorings) | yes |
| SST heatmap (diverging color centered on 50 °F) | same MUR SST | toggle |
| Recent sightings | [iNaturalist API](https://api.inaturalist.org/v1/docs/), last 30 days | toggle |
| Seasonal records | [GBIF API](https://www.gbif.org/developer/occurrence) — this season, all years (aggregates museum/survey/iNat) | toggle |

### The probability model

A 0–1 suitability grid, weighted blend (renormalized over whichever factors are
available):

| Factor | Weight | Basis |
|--------|-------:|-------|
| Temperature suitability | 0.40 | SST: peaks 55–68 °F, off below ~48 °F / above ~75 °F |
| Nearshore proximity | 0.25 | distance to coast (from the SST land mask) |
| Front proximity | 0.20 | distance to the 50 °F leading edge |
| Sighting density | 0.15 | Gaussian-smoothed GBIF seasonal + recent iNaturalist points |
| Bait (chlorophyll) | 0.10 | **opt-in** `--bait`; the readily-available Atlantic chlorophyll product is aging, so it's experimental and off by default |

The advance rate and ETA are extrapolated from how far the 50 °F line moved over
the last `--history-days` days along your area's coastal band — rough, clearly
labeled, not a forecast.

### Inspecting the map

**Click (or tap) anywhere** to inspect the nearest ~7-mi probability cell: a
popup shows the cell's probability and *what it's based on* — each factor's score
(0–1), the raw input behind it (water temp °F, km to shore, km to the front), and
its share of the weighted blend. The clicked cell is outlined.

**Point popups** (toggle on "Recent sightings" / "Seasonal records") carry full
metadata: observation date, the **current** water temp at that spot, location,
record type, who recorded it, the source dataset, coordinate precision, and a
link. Note the temperature is *current SST at the location*, not the temp when
the fish was seen (GBIF records carry no water temperature).

The heatmaps are reprojected to the basemap and masked to a fine (~0.03°)
coastline so they hug the shore without glowing over land. The **color key is
dynamic**: it shows the probability scale (0–100%, relative to today's peak) by
default, and switches to the real SST °F range (min … 50 °F … max) when you
toggle the SST layer on.

**Honest scope:** "probability" here is a **habitat-suitability index, not a
calibrated probability**, and this maps the *aggregate* run, not individual fish.
There is no public live fish-telemetry feed (acoustic-tag data in ACT / MATOS /
OTN networks is researcher-owned, embargoed, and limited to fixed receiver
gates).

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python striper_run_tracker.py            # latest SST, Hatteras -> Maine
open striper_run_*.html
```

### Options

```
--date YYYY-MM-DD     specific SST day (default: latest available)
--bbox LATMIN LATMAX LONMIN LONMAX   default: 35 45 -77 -66
--region NAME         initial viewport: maine | newengland | full (default: maine)
--home-lat / --home-lon / --home-name   "your area" for the status panel ETA
                       (default: Portland, ME @ 43.6, -70.1)
--days N              iNaturalist sighting lookback (default: 30)
--history-days N      prior days used for advance rate / sparkline (default: 10)
--stride DEG          SST sample spacing in degrees (default: 0.05 ~ 3.5 mi)
--bait                add experimental chlorophyll factor (data source is aging)
--out PATH            output HTML path
```

Set `--home-*` to your launch spot to personalize the arrival readout, e.g. Down
East: `--home-lat 44.27 --home-lon -67.31 --home-name "Jonesport"`.

## Deployment

A daily [GitHub Actions workflow](.github/workflows/build.yml) rebuilds the map
and publishes it to GitHub Pages — **fully self-sustaining, no server, nothing
to run by hand**. The cron runs at 12:30 UTC daily (after the MUR SST product
posts). Note: GitHub pauses scheduled workflows after 60 days with no repo
activity, so push something occasionally in the off-season to keep it live.
