"""Evaluation and tables/figures for the report.

R2_oos = 1 - sum (r - r_hat)^2 / sum r^2   (pooled over stocks and test months; the
denominator is NOT demeaned, paper eq. 19). Diebold-Mariano tests compare models through the
cross-sectional average squared-error difference each month (paper eq. 20), Newey-West s.e.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Paper Table 1, monthly R2_oos in % (all stocks). Check against the published table before using.
PAPER_R2_ALL = {"OLS+H": -3.46, "OLS-3+H": 0.16, "PLS": 0.27, "PCR": 0.26, "ENet+H": 0.11, "GLM+H": 0.19,
                "RF": 0.33, "GBRT+H": 0.34, "NN1": 0.33, "NN2": 0.39, "NN3": 0.40, "NN4": 0.39, "NN5": 0.36}
ORDER = ["OLS", "OLS+H", "OLS-3", "OLS-3+H", "PLS", "PCR", "ENet+H", "RF", "GBRT+H",
         "NN1", "NN2", "NN3", "NN4", "NN5"]


def load_preds(out_dir, name) -> pd.DataFrame:
    files = sorted((Path(out_dir) / name).glob("pred_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def available_models(out_dir):
    out = Path(out_dir)
    names = [p.name for p in out.iterdir() if p.is_dir() and any(p.glob("pred_*.parquet"))] if out.exists() else []
    if not names:
        nested = sorted({str(f.parent) for f in out.rglob("pred_*.parquet")}) if out.exists() else []
        hint = (f" Found forecast files in sub-folders instead, e.g. {nested[:3]} -- move each model folder "
                f"directly under {out}/, or use collect_outputs().") if nested else ""
        raise FileNotFoundError(f"No model folders with pred_*.parquet directly in '{out}'.{hint}")
    return [m for m in ORDER if m in names] + sorted(set(names) - set(ORDER))


def collect_outputs(sources, dest="outputs", models=None, dry_run=False) -> pd.DataFrame:
    """Gather model folders (any folder holding pred_*.parquet) from downloaded/unzipped team folders
    into dest/<model>/. Sources are searched recursively in the given order; if a model appears in
    several sources the LAST one wins (list the newest upload last). Folders whose name ends in
    '_macro' or that sit under an 'outputs_old' folder are ignored (superseded runs).
    An existing dest/<model> folder is replaced completely, so old and new files never mix."""
    import shutil
    dest = Path(dest).resolve()
    found = {}
    for src in sources:
        for f in sorted(Path(src).rglob("pred_*.parquet")):
            d = f.parent.resolve()
            if d.parent == dest or "outputs_old" in d.parts or d.name.endswith("_macro"):
                continue
            if models and d.name not in models:
                continue
            if d.name in found and found[d.name] != d:
                print(f"  {d.name}: {found[d.name]}  ->  replaced by {d}")
            found[d.name] = d
    rows = {}
    for name, d in found.items():
        rows[name] = {"source": str(d), "pred files": len(list(d.glob("pred_*.parquet"))),
                      "vi files": len(list(d.glob("vi_*.csv")))}
        if not dry_run:
            target = dest / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(d, target)
    order = [m for m in ORDER if m in rows] + sorted(set(rows) - set(ORDER))
    return pd.DataFrame(rows).T.loc[order] if rows else pd.DataFrame()


def r2_oos(y, yhat):
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)


def size_groups(df: pd.DataFrame, n=1000) -> pd.DataFrame:
    """Flags for the 1,000 largest and 1,000 smallest stocks each month (by lagged market equity)."""
    r = df.groupby("date")["size"].rank(ascending=False, method="first")
    cnt = df.groupby("date")["size"].transform("count")
    return df.assign(top=r <= n, bottom=r > cnt - n)


def r2_table(out_dir, models=None) -> pd.DataFrame:
    models = models or available_models(out_dir)
    rows = {}
    for m in models:
        d = size_groups(load_preds(out_dir, m))
        rows[m] = {"All": r2_oos(d.y, d.pred),
                   "Top 1000": r2_oos(d.y[d.top], d.pred[d.top]),
                   "Bottom 1000": r2_oos(d.y[d.bottom], d.pred[d.bottom]),
                   "years": f"{pd.to_datetime(d.date).dt.year.min()}-{pd.to_datetime(d.date).dt.year.max()}"}
    t = pd.DataFrame(rows).T
    for c in ["All", "Top 1000", "Bottom 1000"]:
        t[c] = 100 * t[c].astype(float)
    t["Paper (All)"] = [PAPER_R2_ALL.get(m, np.nan) for m in t.index]
    return t


def r2_by_year(out_dir, models=None) -> pd.DataFrame:
    models = models or available_models(out_dir)
    res = {}
    for m in models:
        d = load_preds(out_dir, m)
        d["year"] = pd.to_datetime(d.date).dt.year
        res[m] = d.groupby("year").apply(lambda g: 100 * r2_oos(g.y, g.pred))
    return pd.DataFrame(res)


def r2_decomposition(out_dir, models=None) -> pd.DataFrame:
    """Exact split of the pooled R2_oos (zero benchmark) into a time-series ("timing") part and a
    cross-sectional ("stock picking") part, plus the mean monthly rank IC.
    With rbar_t, pbar_t the cross-sectional means of returns and forecasts in month t:
      timing  = sum_t N_t [rbar_t^2 - (rbar_t - pbar_t)^2]                         / sum r^2
      picking = sum_it [(r - rbar_t)^2 - ((r - rbar_t) - (p - pbar_t))^2]          / sum r^2
    and timing + picking = R2_oos. Rank IC = mean over months of the Spearman correlation between
    forecasts and returns (months with constant forecasts are skipped)."""
    models = models or available_models(out_dir)
    rows = {}
    for m in models:
        d = load_preds(out_dir, m)[["date", "y", "pred"]].copy()
        g = d.groupby("date")
        rb, pb = g.y.transform("mean"), g.pred.transform("mean")
        sst = np.sum(d.y ** 2)
        timing = np.sum(rb ** 2 - (rb - pb) ** 2) / sst
        picking = np.sum((d.y - rb) ** 2 - ((d.y - rb) - (d.pred - pb)) ** 2) / sst
        d["ry"], d["rp"] = g.y.rank(), g.pred.rank()
        ic = d.groupby("date")[["ry", "rp"]].apply(
            lambda x: np.nan if x.rp.nunique() < 2 else np.corrcoef(x.ry, x.rp)[0, 1])
        rows[m] = {"R2": 100 * r2_oos(d.y, d.pred), "timing": 100 * timing, "picking": 100 * picking,
                   "rank IC": ic.mean(), "IC t-stat": ic.mean() / ic.std() * np.sqrt(ic.count()),
                   "IC > 0 (%)": 100 * (ic > 0).sum() / ic.count()}
    return pd.DataFrame(rows).T


def newey_west_se(x: np.ndarray, lags: int = 6) -> float:
    x = x - x.mean()
    T = len(x)
    v = x @ x / T
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * (x[L:] @ x[:-L]) / T
    return np.sqrt(v / T)


def _keys(d: pd.DataFrame) -> np.ndarray:
    """(month, permno) as one int64 key, e.g. 198701 * 1e6 + permno."""
    ym = d["date"].dt.year.to_numpy().astype(np.int64) * 100 + d["date"].dt.month.to_numpy()
    return ym * 1_000_000 + d["permno"].to_numpy().astype(np.int64)


def dm_matrix(out_dir, models=None, lags=6) -> pd.DataFrame:
    """DM statistic of row model vs column model; positive = column model is more accurate.
    Uses only stock-months forecast by every model (common sample), then the monthly cross-sectional
    mean of squared errors (GKX eq. 20) with Newey-West standard errors. Memory-light: one model in
    memory at a time plus int64 keys and float32 errors."""
    models = models or available_models(out_dir)
    keys, errs = {}, {}
    for m in models:
        d = load_preds(out_dir, m)
        keys[m] = _keys(d)
        errs[m] = ((d["y"].to_numpy(np.float64) - d["pred"].to_numpy(np.float64)) ** 2).astype(np.float32)
        del d
    common = None
    for k in keys.values():
        common = np.unique(k) if common is None else np.intersect1d(common, k)
    monthly = {}
    for m in models:
        mask = np.isin(keys[m], common)
        monthly[m] = pd.Series(errs[m][mask].astype(np.float64)).groupby(keys[m][mask] // 1_000_000).mean()
    monthly = pd.DataFrame(monthly)
    out = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            if a != b:
                d = (monthly[a] - monthly[b]).to_numpy()
                out.loc[a, b] = d.mean() / newey_west_se(d, lags)
    out.attrs["n_months"] = len(monthly)
    return out


def coverage(out_dir, models=None) -> pd.DataFrame:
    """Test years actually present in each model's predictions (30 = complete, 1987-2016),
    independent of how many files a model has (biennial refits store two years per file)."""
    models = models or available_models(out_dir)
    rows = {}
    for m in models:
        d = load_preds(out_dir, m)
        yrs = sorted(d["date"].dt.year.unique())
        missing = [y for y in range(1987, 2017) if y not in yrs]
        rows[m] = {"test years": len(yrs), "missing": ", ".join(map(str, missing)) or "-",
                   "VI files": len(list((Path(out_dir) / m).glob("vi_*.csv")))}
    return pd.DataFrame(rows).T


def variable_importance(out_dir, models=None, normalise=True):
    """Average the R^2 drops over the VI refit years; negative drops -> 0; chars normalised
    to sum to one within each model (paper Figure 4). Returns (chars, macro) DataFrames."""
    models = models or available_models(out_dir)
    ch, mc = {}, {}
    for m in models:
        files = sorted((Path(out_dir) / m).glob("vi_*.csv"))
        if not files:
            continue
        v = pd.concat([pd.read_csv(f, index_col=0).iloc[:, 0] for f in files], axis=1).mean(1).clip(lower=0)
        is_macro = v.index.str.startswith("macro:")
        c, k = v[~is_macro], v[is_macro]
        k.index = k.index.str.replace("macro:", "")
        ch[m] = c / c.sum() if normalise and c.sum() > 0 else c
        mc[m] = k / k.sum() if normalise and k.sum() > 0 else k
    return pd.DataFrame(ch), pd.DataFrame(mc)


def hparam_history(out_dir, model, key) -> pd.Series:
    out = {}
    for f in sorted((Path(out_dir) / model).glob("hparams_*.json")):
        h = json.load(open(f))
        if key in h:
            out[int(f.stem.split("_")[1])] = h[key]
    return pd.Series(out, name=f"{model}:{key}")


def decile_portfolios(out_dir, model, weight="ew", n=10) -> pd.DataFrame:
    """Optional (paper Section 2.4): monthly decile sorts on the forecast, H-L spread."""
    d = load_preds(out_dir, model)
    d["dec"] = d.groupby("date")["pred"].transform(lambda s: pd.qcut(s.rank(method="first"), n, labels=False)) + 1
    if weight == "ew":
        p = d.groupby(["date", "dec"])["y"].mean()
    else:
        d["w"] = market_cap(d["size"])
        p = d.groupby(["date", "dec"]).apply(lambda g: np.average(g.y, weights=g["w"]))
    p = p.unstack()
    p["H-L"] = p[n] - p[1]
    return p


def market_cap(size: pd.Series) -> pd.Series:
    """GKX describe mvel1 as LOG market equity; undo the log if the values look like logs."""
    return np.exp(size) if size.max() < 50 else size.clip(lower=0) + 1e-12


def sharpe(r: pd.Series, k=12):
    return r.mean() / r.std() * np.sqrt(k)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_r2(t: pd.DataFrame, path=None):
    import matplotlib.pyplot as plt
    cols = ["All", "Top 1000", "Bottom 1000"]
    ax = t[cols].plot(kind="bar", figsize=(8, 3.2), width=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(r"monthly $R^2_{oos}$ (%)")
    lo = max(t[cols].min().min(), -1.5)
    ax.set_ylim(lo - 0.1, t[cols].max().max() + 0.2)
    ax.grid(alpha=0.3, axis="y"); ax.legend(frameon=False, ncol=3)
    plt.xticks(rotation=45, ha="right")
    ax.set_title("OLS / OLS+H bars truncated (actual ≈ −22% / −23%)", fontsize=8, loc="right")
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
    return ax


def plot_vi_heatmap(ch: pd.DataFrame, top=20, path=None):
    """Rank of each characteristic within each model (1 = most important), like paper Figure 5."""
    import matplotlib.pyplot as plt
    ranks = ch.rank(ascending=False)
    order = ranks.sum(1).sort_values().index[:top]
    R = ranks.loc[order]
    fig, ax = plt.subplots(figsize=(0.55 * R.shape[1] + 2, 0.28 * top + 1))
    im = ax.imshow(R.to_numpy(), cmap="Blues_r", aspect="auto", vmin=1, vmax=len(ch))
    ax.set_yticks(range(len(R)), R.index, fontsize=7); ax.set_xticks(range(R.shape[1]), R.columns, rotation=45, fontsize=7)
    fig.colorbar(im, ax=ax, label="importance rank")
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_vi_bars(ch: pd.DataFrame, models, top=20, path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 4), sharey=False)
    for ax, m in zip(np.atleast_1d(axes), models):
        s = ch[m].sort_values().tail(top)
        ax.barh(s.index, s.values); ax.set_title(m, fontsize=9); ax.tick_params(labelsize=6)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_complexity(series_dict: dict, path=None):
    """Model complexity over time (paper Figure 3): ENet non-zeros, PCR/PLS components, GBRT trees ..."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(series_dict), figsize=(2.8 * len(series_dict), 2.6))
    for ax, (k, s) in zip(np.atleast_1d(axes), series_dict.items()):
        ax.plot(s.index, pd.to_numeric(s, errors="coerce"), marker=".")
        ax.set_title(k, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def to_latex(df, path, fmt="%.2f", **kw):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(float_format=lambda x: fmt % x, na_rep="--", **kw)
    Path(path).write_text(tex)
    return tex
