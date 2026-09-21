# Low-Cost Air-Quality Sensor Calibration
**低成本空氣品質感測器校正模型**

A machine-learning pipeline that calibrates raw low-cost air-quality sensor readings into reference-grade pollutant estimates, using the UCI Air Quality dataset. Built as a fully modular, reproducible pipeline that benchmarks seven regression models across four pollutants.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Low-cost gas sensors are cheap and easy to deploy, but their raw readings drift with temperature, humidity, and cross-sensitivity to other gases. This project learns a data-driven calibration that maps raw sensor channels — plus weather and temporal context — to reference-grade concentrations, and evaluates which model generalizes best for each pollutant.

- **Dataset:** UCI Air Quality — 9,471 hourly records from a multi-sensor device deployed in an Italian city.
- **Targets:** CO, C6H6 (benzene), NO2, NOx.
- **Models benchmarked:** Linear, Ridge, Lasso, Random Forest, SVR, XGBoost, LightGBM.
- **Evaluation:** time-ordered 80/20 split (no shuffling, to avoid look-ahead leakage), reported with MAE / RMSE / R².

## Key results

Best model per pollutant on the held-out test set:

| Pollutant | Best model | Test RMSE | Test R² |
|---|---|---|---|
| C6H6 (benzene) | LightGBM | 1.71 | 0.931 |
| CO | LightGBM | 0.53 | 0.833 |
| NO2 | XGBoost | 32.5 | 0.616 |
| NOx | SVR | 88.6 | 0.782 |

Gradient-boosted trees win where the sensor–target relationship is strongest (CO, benzene); NO2 / NOx are harder and are the clearest target for future work. Every run also emits diagnostic plots — calibration before/after, residuals, feature importance, and SHAP summaries — under `outputs/figures/`.

## Method

1. **Load & parse** the UCI CSV (`;`-separated, comma decimals, `-200` = missing).
2. **Feature engineering** — 5 sensor channels, 3 weather variables (T, RH, AH), 6 calendar features (hour, day-of-week, month, weekend, season, rush-hour), plus lag / rolling-mean features that capture sensor autocorrelation and response delay.
3. **Preprocessing** — drop rows missing the target, median-impute the remaining features.
4. **Train & evaluate** all seven models on a time-ordered split; select the best per pollutant.
5. **Persist** metrics, per-row calibrated result tables, diagnostic plots, and the best model.

## Project structure

```
project/
├── main.py                 # end-to-end pipeline
├── src/
│   ├── config.py           # target pollutant + feature config
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py           # the 7 regressors
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
├── data/AirQualityUCI.csv  # UCI dataset
├── outputs/                # figures, tables, trained models (per pollutant)
└── requirements.txt
```

## Getting started

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

To calibrate a different pollutant, set `TARGET_COL` in `src/config.py` (e.g. `"NOx(GT)"`) and re-run. Outputs are written to `outputs/{figures,tables,models}/<pollutant>/`.

## Dataset

UCI Machine Learning Repository — *Air Quality Data Set* (De Vito et al.). The CSV is included here for reproducibility; the original source and license are at <https://archive.ics.uci.edu/dataset/360/air+quality>.

## Author

**Wei-Pei Chen (陳暐培)** — Dept. of Management Information Systems, National Chung Hsing University. Data Mining course project.

## License

Released under the [MIT License](LICENSE).
