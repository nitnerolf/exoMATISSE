# CO₂ Model Updater

This Python script updates a CO₂ model stored in `nco2.json` using:

- **Scripps** `.csv` data files that can be found here : `https://scrippsco2.ucsd.edu/data/atmospheric_co2/bcs.html`
- and/or **NOAA** `.txt` flask data files that can be found here : `https://gml.noaa.gov/data/dataset.php?item=ush-co2-flask`

It computes the **linear trend** (slope and intercept) and **seasonal variations** using linear regression and spline smoothing. The results are written into `nco2.json`, which must be located in the **same directory as the script**.

---

## ▶️ Usage

```bash
python3 op_update_co2.py [OPTIONS]
```

### 📅 Options

| Option       | Type  | Default | Description                                                                     |
| ------------ | ----- | ------- | ------------------------------------------------------------------------------- |
| `--base_dir` | `str` | `.`     | Directory containing Scripps `.csv` files. Required for `scripps` or `both`.    |
| `--file`     | `str` | *None*  | NOAA `.txt` file (`*_ccgg_event.txt`). Required for `noaa` or `both`.           |
| `--source`   | `str` | `noaa`  | Data source: `scripps`, `noaa`, or `both`. If omitted, source is auto-detected. |
| `--year`     | `int` | `2015`  | Start year for regression and seasonal analysis.                                |
| `--year_max` | `int` | `2024`  | End year (exclusive) for regression and smoothing.                              |
| `--plot`     | flag  | `False` | If set, shows plots of regression and seasonal components.                      |

---

## 📂 Example Commands

### NOAA only:

```bash
python3 op_update_co2.py --file co2_ush_surface-flask_1_ccgg_event.txt --source noaa --plot
```

### Scripps only:

```bash
python3 op_update_co2.py --base_dir ./scripps_data --source scripps --plot
```

### Both NOAA and Scripps:

```bash
python3 op_update_co2.py --base_dir ./scripps_data --file co2_ush_surface-flask_1_ccgg_event.txt --source both --plot
```

### Autodetect source:

```bash
python3 op_update_co2.py --base_dir ./scripps_data --file co2_ush_surface-flask_1_ccgg_event.txt --plot
```

If `--source` is omitted, it will:

- use `scripps` if `.csv` files are found,
- use `noaa` if `--file` is provided,
- use `both` if both are present.

---

## 📄 Output

This script updates or creates the following fields in `nco2.json`:

- `slope`: CO₂ linear growth rate,
- `intercept`: Y-intercept of the trend,
- `variations`: seasonal residual signal, smoothed and averaged over each year.

This file is used as a compact and up-to-date CO₂ model for OPTRA.

---

## ⚠️ Notes

- `nco2.json` must be in the same directory as the script.
- NOAA input file must be a valid `_ccgg_event.txt` file and not `_ccgg_month.txt` file
- The seasonal variation is obtained after detrending and spline smoothing.
- All plots (if enabled) are shown together at the end.

