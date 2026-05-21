# Striped Bass Run Tracker

Estimate where the annual striped bass (*Morone saxatilis*) migration front is
along the US East Coast, right now, using only free public data — and render it
as an interactive map.

## How it works

The coastal run tracks water temperature: bass follow the **~50 °F** water north
in spring and south in fall. So instead of (impossibly) tracking individual
fish, this tool maps the **50 °F isotherm** as a proxy for the leading edge of
the run, draws a multi-day **trail** of that line so you can watch it advance,
and overlays real **buoy water temps** plus sightings to ground-truth it.

| Layer | Source | Notes |
|-------|--------|-------|
| Sea-surface temperature + 50 °F front | [NOAA/JPL MUR SST](https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.html) via ERDDAP | ~1 km native; sampled to ~3.5 mi |
| 50 °F front trail (last N days) | same | one isotherm per prior day, faded oldest→newest |
| Buoy water temps | [NOAA NDBC](https://www.ndbc.noaa.gov/) realtime (incl. NERACOOS Gulf of Maine moorings) | live, color-coded above/below 50 °F |
| Recent sightings | [iNaturalist API](https://api.inaturalist.org/v1/docs/) | geotagged, last 30 days |
| Historical records | [GBIF API](https://www.gbif.org/developer/occurrence) | last 4 yrs; aggregates museum/survey/iNat data |

The Gulf of Maine buoys are the most actionable layer for the Maine run: when
the Portland buoy crosses 50 °F while Down East is still cold, the front is
arriving — something iNaturalist's sparse Maine coverage can't show.

**Honest scope:** this maps the *aggregate* run front, not individual fish.
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
--days N              iNaturalist sighting lookback (default: 30)
--history-days N      prior days of 50 °F front trail (default: 10, 0 disables)
--gbif-years N        historical record window in years (default: 4, 0 disables)
--stride DEG          SST sample spacing in degrees (default: 0.05 ~ 3.5 mi)
--out PATH            output HTML path
```

## Deployment

A daily [GitHub Actions workflow](.github/workflows/build.yml) rebuilds the map
and publishes it to GitHub Pages — **fully self-sustaining, no server, nothing
to run by hand**. The cron runs at 12:30 UTC daily (after the MUR SST product
posts). Note: GitHub pauses scheduled workflows after 60 days with no repo
activity, so push something occasionally in the off-season to keep it live.
