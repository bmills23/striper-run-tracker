# Striped Bass Run Tracker

Estimate where the annual striped bass (*Morone saxatilis*) migration front is
along the US East Coast, right now, using only free public data — and render it
as an interactive map.

## How it works

The coastal run tracks water temperature: bass follow the **~50 °F** water north
in spring and south in fall. So instead of (impossibly) tracking individual
fish, this tool maps the **50 °F isotherm** as a proxy for the leading edge of
the run, then overlays recent **citizen-science sightings** to ground-truth it.

| Layer | Source | Notes |
|-------|--------|-------|
| Sea-surface temperature | [NOAA/JPL MUR SST](https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.html) via ERDDAP | ~1 km native; sampled to ~3.5 mi |
| Striped bass sightings | [iNaturalist API](https://api.inaturalist.org/v1/docs/) | geotagged, last 30 days |

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
--bbox LATMIN LATMAX LONMIN LONMAX   default: 35 44 -77 -66
--days N              sighting lookback window (default: 30)
--stride DEG          SST sample spacing in degrees (default: 0.05 ~ 3.5 mi)
--out PATH            output HTML path
```
