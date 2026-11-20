"""Recursive performance evaluation (paper Section 2.1).

For each refit year Y (1987, ..., 2016):
    training   = 1957 .. Y-13   (expanding: 18 years at the first refit, 47 at the last)
    validation = Y-12 .. Y-1    (rolling 12 years, used only to pick hyper-parameters)
    test       = Y               (never seen before)
Each refit writes outputs/<model>/pred_<Y>.parquet and hparams_<Y>.json, so runs can be
split across team members / Colab sessions and resumed after a disconnect.
Variable importance (paper Section 2.3) is computed at refits in cfg.vi_years.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .panel import Panel


def make_model(name: str, cfg: Config, **kw):
    """Model registry. Names follow the paper's tables."""
    from .linear import LinearModel
    linear = {
        "OLS":     dict(kind="ols"),
        "OLS+H":   dict(kind="ols", huber=True),
        "OLS-3":   dict(kind="ols", features="ols3"),
        "OLS-3+H": dict(kind="ols", huber=True, features="ols3"),
        "ENet+H":  dict(kind="enet", huber=True),
        "PCR":     dict(kind="pcr"),
        "PLS":     dict(kind="pls"),
    }
    if name in linear:
        return LinearModel(chunk=cfg.chunk, **{**linear[name], **kw})
    if name in ("RF", "GBRT+H"):
        from .trees import TreeModel
        return TreeModel("rf" if name == "RF" else "gbrt", seed=cfg.seed, **kw)
    if name.startswith("NN"):
        from .nn import NNModel
        return NNModel(name, seed=cfg.seed, **kw)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# variable importance: reduction in R^2 when one predictor is set to zero
# ---------------------------------------------------------------------------
def r2_zero_benchmark(y, yhat):
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)


def variable_importance(model, panel: Panel, rows: np.ndarray, fill: str = "zero") -> pd.Series:
    """For each characteristic j (all its macro interactions) and each macro predictor k
    (all its interactions), the drop in in-sample R^2 when that input is neutralised
    (paper Section 2.3, computed on a training subsample).

    fill="zero"  : set the characteristic to 0 (GKX; 0 = cross-sectional median of a rank-normalised
                   continuous characteristic). Used for all team runs.
    fill="median": set it to its cross-sectional median in each month. Identical to "zero" for
                   continuous characteristics, but correct for binary / heavily tied ones (divi, divo,
                   sin, convind ...), whose "0" group sits slightly below zero while missing values are
                   exactly 0 -- zeroing them marks every stock as "missing", which trees pick up.
    Macro predictors and industry dummies are always set to zero."""
    C, Mx, S = panel.raw(rows)
    y = panel.y[rows]
    base = r2_zero_benchmark(y, model.predict_raw(C, Mx, S))
    groups = pd.factorize(panel.date[rows])[0] if fill == "median" else None
    out = {}
    for j, name in enumerate(panel.chars):
        Cz = C.copy()
        if fill == "median":
            Cz[:, j] = pd.Series(C[:, j]).groupby(groups).transform("median").to_numpy(np.float32)
        else:
            Cz[:, j] = 0.0
        out[name] = base - r2_zero_benchmark(y, model.predict_raw(Cz, Mx, S))
    S_none = np.full_like(S, -1)                        # all industry dummies set to zero
    out["sic2"] = base - r2_zero_benchmark(y, model.predict_raw(C, Mx, S_none))
    for k, name in enumerate(panel.macro_names):
        Mz = Mx.copy(); Mz[:, k] = 0.0
        out["macro:" + name] = base - r2_zero_benchmark(y, model.predict_raw(C, Mz, S))
    return pd.Series(out, name="r2_drop")


def rerun_vi(name: str, panel: Panel, cfg: Config, fill: str = "median", **kw):
    """Re-estimate `name` only at the refits in cfg.vi_years and (re)write vi_<Y>.csv in
    cfg.out_dir/<name>/ with the chosen fill. Forecast files are not touched. The refits at the
    VI years are tuning refits in the team runs (1987, 1992, ... = every 5th refit), and the random
    subsample is drawn exactly as in run_model, so only the fill rule changes."""
    out = Path(cfg.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    model = make_model(name, cfg, **kw)
    rng = np.random.default_rng(cfg.seed)
    for Y in cfg.refit_years():
        if Y not in cfg.vi_years:
            continue
        tr_y, va_y, _ = cfg.split(Y)
        tr, va = panel.rows(tr_y), panel.rows(va_y)
        t0 = time.time()
        model.fit(panel, tr, va, tune=True)
        rows = np.sort(rng.choice(np.arange(tr.start, tr.stop), min(cfg.vi_sample, tr.stop - tr.start),
                                  replace=False))
        variable_importance(model, panel, rows, fill).to_csv(out / f"vi_{Y}.csv")
        print(f"[{name}] {Y}: variable importance ({fill}) saved ({(time.time() - t0) / 60:.1f} min)")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_model(name: str, panel: Panel, cfg: Config, years=None, tune_every: int = 1, vi: bool = True, **kw):
    """Recursive estimation for one model. tune_every=k re-selects hyper-parameters every k refits
    (1 = every year, as in the paper) and otherwise reuses the last choice."""
    out = Path(cfg.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    years = years or cfg.refit_years()
    model = make_model(name, cfg, **kw)
    rng = np.random.default_rng(cfg.seed)
    for i, Y in enumerate(cfg.refit_years()):
        if Y not in years:
            continue
        f = out / f"pred_{Y}.parquet"
        if f.exists():
            print(f"[{name}] {Y}: done, skipping")
            continue
        tr_y, va_y, te_y = cfg.split(Y)
        tr, va, te = panel.rows(tr_y), panel.rows(va_y), panel.rows(te_y)
        scheduled = i % tune_every == 0
        if not scheduled and getattr(model, "best", None) is None and (out / "last_best.json").exists():
            b = json.load(open(out / "last_best.json"))          # resume: reuse the last tuned choice
            model.best = tuple(b) if isinstance(b, list) else b
        tune = scheduled or getattr(model, "best", None) is None
        t0 = time.time()
        print(f"[{name}] refit {Y}: train {tr_y} ({tr.stop - tr.start:,}) | val {va_y} | test {te_y}"
              + (" | tuning" if tune else ""))
        model.fit(panel, tr, va, tune=tune)
        pred = model.predict_rows(panel, te)
        pd.DataFrame({"permno": panel.permno[te], "date": panel.date[te], "y": panel.y[te],
                      "pred": pred.astype(np.float32), "size": panel.size[te]}).to_parquet(f)
        hp = {k: v for k, v in getattr(model, "hparams", {}).items()}
        hp["seconds"] = time.time() - t0
        json.dump(hp, open(out / f"hparams_{Y}.json", "w"), indent=1, default=float)
        if getattr(model, "best", None) is not None:
            json.dump(model.best, open(out / "last_best.json", "w"), default=float)
        r2 = r2_zero_benchmark(panel.y[te], pred)
        print(f"[{name}] {Y}: test R2_oos = {100 * r2:.3f}%  ({hp['seconds'] / 60:.1f} min)")
        if vi and Y in cfg.vi_years:
            rows = np.sort(rng.choice(np.arange(tr.start, tr.stop), min(cfg.vi_sample, tr.stop - tr.start),
                                      replace=False))
            variable_importance(model, panel, rows).to_csv(out / f"vi_{Y}.csv")
            print(f"[{name}] {Y}: variable importance saved")
    return model
