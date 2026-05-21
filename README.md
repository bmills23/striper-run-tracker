# Striped Bass Run Tracker

Estimate where the annual striped bass (*Morone saxatilis*) migration front is
along the US East Coast, right now, using only free public data — and render it
as an interactive map.

## How it works

The coastal run tracks water temperature: bass follow the **~50 °F** water north
in spring and south in fall. So instead of (impossibly) tracking individual
fish, this tool reads that signal and answers, in plain English, **"is the run
near me, and when does it arrive?"**

A **status panel** does the interpreting for you — front location, whether
run-temperature water has reached your area, how fast it's moving, and a rough
arrival estimate — over a clean map you can read at a glance.

| Element | Source | Shown by default |
|---------|--------|------------------|
| **Status panel** (front location, your-area status, advance rate, ETA, trend sparkline) | computed from buoys + SST | yes |
| **SST heatmap** — diverging color centered on 50 °F, so the front is the color break | [NOAA/JPL MUR SST](https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.html) via ERDDAP (~1 km, sampled to ~3.5 mi) | yes |
| **50 °F front line** — single clean leading edge (longest segments only) | same SST | yes |
| **Buoy water temps** — color-coded above/below 50 °F | [NOAA NDBC](https://www.ndbc.noaa.gov/) realtime (incl. NERACOOS Gulf of Maine moorings) | yes |
| 50 °F trail (raw daily lines) | same SST | toggle |
| Recent sightings | [iNaturalist API](https://api.inaturalist.org/v1/docs/), last 30 days | toggle |
| Historical records | [GBIF API](https://www.gbif.org/developer/occurrence), last 4 yrs (aggregates museum/survey/iNat) | toggle |

The advance rate and ETA are extrapolated from how far the 50 °F line moved over
the last `--history-days` days along your area's coastal band — a rough estimate,
clearly labeled, not a forecast. The Gulf of Maine buoys are the most actionable
real signal for the Maine run (e.g. Portland crossing 50 °F while Down East is
still cold), which iNaturalist's sparse Maine coverage can't show.

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
--home-lat / --home-lon / --home-name   "your area" for the status panel ETA
                       (default: Portland, ME @ 43.6, -70.1)
--days N              iNaturalist sighting lookback (default: 30)
--history-days N      prior days used for advance rate / sparkline (default: 10)
--gbif-years N        historical record window in years (default: 4, 0 disables)
--stride DEG          SST sample spacing in degrees (default: 0.05 ~ 3.5 mi)
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
