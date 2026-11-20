# Empirical Asset Pricing via Machine Learning — Replication

MSDM5054 Statistical Machine Learning (HKUST), Final Project, Topic 5.1.
Replication of Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning", *Review of Financial Studies* 33(5).

We predict next-month excess returns of individual U.S. stocks from 94 firm characteristics, 8 macroeconomic
predictors and 74 industry dummies (920 inputs). We estimate 14 specifications with the paper's recursive scheme:
annual refits, an expanding training window, a rolling 12-year validation window, and 30 out-of-sample years (1987–2016).
We then report out-of-sample R², Diebold–Mariano tests, model complexity and variable importance.

**Team:** Zhang Qi · Wei Fanjing · Tang Yunshu · Du Yunhe

## Main result: monthly out-of-sample R² (%), 1987–2016

| Model | All | Top 1000 | Bottom 1000 | GKX (All) |
|---|---:|---:|---:|---:|
| OLS | −22.49 | −103.35 | −9.43 | — |
| OLS+H | −23.48 | −110.65 | −9.94 | −3.46 |
| OLS-3 | 0.20 | 0.23 | 0.19 | — |
| OLS-3+H | 0.19 | 0.24 | 0.17 | 0.16 |
| PLS | 0.20 | −0.59 | 0.40 | 0.27 |
| PCR | 0.28 | −0.00 | 0.38 | 0.26 |
| ENet+H | 0.07 | −0.14 | 0.21 | 0.11 |
| RF | 0.08 | −0.08 | 0.21 | 0.33 |
| GBRT+H | 0.11 | 0.21 | 0.12 | 0.34 |
| NN1 | 0.34 | 0.36 | 0.52 | 0.33 |
| NN2 | 0.31 | 0.61 | 0.36 | 0.39 |
| NN3 | 0.21 | 0.36 | 0.38 | 0.40 |
| NN4 | 0.19 | 0.32 | 0.38 | 0.39 |
| NN5 | 0.28 | 0.26 | 0.44 | 0.36 |

R²_oos = 1 − Σ(r − r̂)² / Σr². It is pooled over all stock-months, and the benchmark is a zero forecast, as in GKX.
"Top/Bottom 1000" are the largest and smallest 1,000 stocks by market equity each month.

The remaining results are in [`results/`](results/):

| File | Content |
|---|---|
| `tables/tab_r2.tex` | R² table |
| `tables/r2_by_year.csv` | R² by year |
| `tables/tab_dm.tex` | Diebold–Mariano tests, 360 months |
| `tables/tab_vi_groups.tex`, `tables/tab_vi_macro.tex` | Variable importance by characteristic group and for the macro predictors |
| `tables/vi_chars.csv` | Variable importance of each characteristic |
| `tables/tab_portfolios.tex` | Decile long–short portfolios (optional section) |
| `tables/coverage.csv` | Test years covered by each model |
| `figures/` | Plots of the R², complexity and variable-importance results |

## Report

[`report/main.pdf`](report/main.pdf) is the final report (NeurIPS format). To rebuild it on Overleaf, upload `report/`
together with `neurips_2019.sty`; without the style file, `main.tex` falls back to a plain layout.

## Repository layout

```
src/
  config.py      paths, sample split, the 94 characteristics and 8 macro predictors
  data.py        builds the panel: return timing check, excess returns, rank-normalisation to [-1, 1],
                 Welch–Goyal macro predictors
  panel.py       in-memory panel; builds the 920 inputs (94 × (1 + 8 macro) + 74 sic2 dummies) on the fly
  linear.py      OLS, OLS-3, Huber (IRLS), ENet+H, PCR, PLS from streamed Gram matrices
  trees.py       RF, GBRT+H (XGBoost)
  nn.py          NN1–NN5 (PyTorch): ReLU, batch norm, L1 penalty, early stopping, ensembles
  recursive.py   annual refits; saves forecasts, hyper-parameters and variable importance per year
  metrics.py     R²_oos, Diebold–Mariano tests, variable importance, figures and LaTeX tables
notebooks/
  main.ipynb     documented pipeline: data → recursive estimation → results
  runs/          executed notebooks of each member's model runs (process record)
results/         final tables and figures
report/          report source (main.tex), figures and compiled PDF
```

## Data

The data is not included in this repository (several GB).

1. **Firm characteristics:** the authors' public dataset `GKX_20201231.csv` (June 2021 release; 1926–2020,
   4.3M stock-months). Download it from Dacheng Xiu's website (`datashare.zip`) and save it as `data/raw/datashare.csv`.
   Its `RET` column is the CRSP monthly holding return including dividends. The characteristics are already lagged:
   1 month for monthly, 4 months for quarterly and 6 months for annual characteristics.
2. **Macro predictors:** the Welch–Goyal monthly data (`PredictorData2025.xlsx`, "Updated data (up to 2025)" on Amit
   Goyal's website). Put it in `data/raw/`.

The panel (`data/panel.parquet`) and our saved forecasts (`outputs/<model>/pred_<year>.parquet`) are available
from the team on request.

## How to reproduce

```bash
pip install -r requirements.txt
jupyter lab notebooks/main.ipynb
```

1. **§1 — build the panel** (~10–30 min). The notebook runs an automatic timing check. In our data,
   corr(mom1m_{t+1}, ret_t) = 1.000 and corr(mom1m_t, ret_t) = 0.004, so each row's characteristics predict the same
   row's return. Excess return = RET − Rfree.
2. **§3–§5 — estimate models.** Set `MY_MODELS`, e.g. `["OLS", "OLS+H", "PLS"]`. Run `QUICK = True` first (two test
   years), then `QUICK = False`. A GPU is recommended for the trees and the neural networks. Every refit is saved, so an
   interrupted run resumes where it stopped.
3. **§6–§7 — results.** Leave `MY_MODELS = []`. Put all model folders into `outputs/`, or list the downloaded team
   folders in `SOURCES` (first cell of §6), which copies them in for you. This step needs no GPU, and the full panel is
   not loaded.

Two optional checks are in the notebook (flags default to `False`):

* **§3b — `RUN_OLS_CHECK`.** Re-estimates OLS and OLS+H for 1987, 1988 and 1994 on the CPU in float64 (~20 min, ~3 GB
  RAM) and compares them with the team's GPU run. The check prints a verdict: "same" means the large negative R² is
  genuine; "different" means you should re-run both models with `GPU_FLOAT64 = True` (§3c).
* **§4b — `RUN_TREE_VI_MEDIAN`.** Recomputes the tree variable importance by setting each characteristic to its
  monthly median instead of zero (see the note on variable importance below). Only `vi_*.csv` are rewritten.

## Implementation choices and deviations from GKX

| Topic | Our choice |
|---|---|
| Data | 2021 public release of the characteristics (not the original 2018 sample); sample 1957–2016 as in GKX |
| Linear models | Exact solutions from streamed Gram matrices Z'WZ. Huber threshold = 99.9% quantile of absolute residuals (IRLS). ENet+H applies the elastic-net path to the Huber-weighted Gram matrix (one-step approximation). |
| Trees | XGBoost. Inputs: 94 characteristics + sic2, not the 920 interaction terms. The macro predictors are **not** standalone tree inputs (see below). RF depth {2, 4, 6}, features per split {10, 30}. GBRT+H uses a pseudo-Huber loss and early stopping. |
| Neural networks | Ensembles of 5 networks (3 for NN2–NN4), not 10. NN2–NN4 are refit every two years; all other models are refit every year. |
| Hyper-parameter tuning | Every year for the linear models; every 5 refits for the trees and the networks. |
| Variable importance | Drop in R² when one input is set to zero (optionally its monthly median, for the trees), computed at 6 refits on 300,000 training observations. |
| Not estimated | GLM with splines. |

### Notes on the results

* **Tree models and macro predictors.** In a first specification we gave the trees the 8 macro predictors as
  standalone levels (103 inputs). The trees then forecast the market-wide component of returns, which is mostly
  noise, and R²_oos was negative: RF −0.66%, GBRT+H −0.11%. In GKX the macro predictors enter only through
  interactions with cross-sectionally centred characteristics, so they affect differences between stocks but not the
  overall level of the forecasts. Removing the standalone levels gives the positive results above. An independent
  920-input tree implementation (Tang Yunshu) gives RF +0.24% and GBRT+H +0.19%.
* **OLS / OLS+H with all 920 inputs.** Both are strongly negative, as in GKX, but larger in magnitude. The loss comes
  from a small set of explosive forecasts: about 15% of forecasts exceed ±10% per month. These arise from nearly
  collinear inputs: characteristics and their interactions with slowly moving macro variables, and near-duplicate pairs
  such as beta/betasq. Capping forecasts at ±5% would give −3.25%; this is a diagnostic only, not a reported result.
  The Huber loss does not help, because the problem is extrapolation in the inputs, not extreme returns.
* **Diebold–Mariano tests.** Every model except OLS is significantly better than OLS+H (DM ≈ 4.2); OLS and OLS+H are not significantly different (DM −1.26). The other pairwise
  differences are mostly insignificant, which is a weaker result than in GKX.
* **Variable importance.** The leading characteristics are mom1m, mom12m, mvel1, turn, chmom and retvol, in line with
  GKX. Two caveats apply:
  * For the trees, binary indicators (divi, divo, sin, convind) rank high partly because of how the importance is
    measured. Missing values are set to 0, and setting an input to 0 marks every stock as "missing".
    `RUN_TREE_VI_MEDIAN` (§4b) removes this artefact.
  * For the neural networks, the macro importance partly reflects input scale: the macro predictors enter the
    interaction terms in raw units, and the networks are trained with an L1 penalty.

## Use of AI tools

The code framework was drafted with the help of an AI assistant (Claude, Anthropic). All code was run, checked and
debugged by the team, and all numbers come from our own runs. See the report for each member's contribution and AI use.
