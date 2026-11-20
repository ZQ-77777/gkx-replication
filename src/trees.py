"""Random forest and gradient-boosted regression trees (+Huber), via XGBoost (GPU if available).

Inputs: the 94 rank-normalised characteristics and the sic2 code (95 inputs). Trees model
interactions themselves, so the 846 explicit characteristic x macro interaction terms are not built
(memory ~1.5 GB instead of ~14 GB; documented deviation from GKX, which feeds 920 inputs to every model).

The macro predictors are deliberately NOT given to the trees as standalone levels. In GKX they enter
only through interactions with (cross-sectionally centred) characteristics, so they can shape
cross-sectional differences but not the overall level of the forecasts. A first specification with the
8 macro levels as extra inputs (103 inputs) let the trees forecast the common, market-wide component of
returns and produced negative R2_oos (RF -0.66%, GBRT+H -0.11%); see README.

Hyper-parameters (paper Table A.5):
  RF   : depth 1-6, 300 trees, features per split in {3, 5, 10, 20, 30, 50}
  GBRT : depth 1-2, up to 1000 trees, learning rate {0.01, 0.1}, Huber loss
"""
import itertools

import numpy as np

from .panel import Panel


def raw_matrix(C, Mx, S) -> np.ndarray:
    sic = np.where(S >= 0, S, np.nan).astype(np.float32)[:, None]
    return np.hstack([C, sic])     # no standalone macro levels: in GKX macro enters only via c x macro interactions


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:   # ImportError, or OSError from a broken torch install
        return "cpu"


class TreeModel:
    def __init__(self, kind="gbrt", grid=None, seed=42, n_jobs=-1):
        self.kind, self.seed, self.n_jobs = kind, seed, n_jobs
        if grid is None:
            grid = ({"max_depth": [2, 4, 6], "mtry": [10, 30]} if kind == "rf"
                    else {"max_depth": [1, 2], "learning_rate": [0.01, 0.1]})
        self.grid = grid
        self.booster, self.hparams, self.best = None, {}, None

    def _params(self, hp, n_feat, xi=None):
        base = {"tree_method": "hist", "device": _device(), "seed": self.seed, "max_bin": 64}
        if self.kind == "rf":
            return {**base, "objective": "reg:squarederror", "learning_rate": 1.0,
                    "num_parallel_tree": 300, "subsample": 0.5, "max_depth": hp["max_depth"],
                    "colsample_bynode": min(1.0, hp["mtry"] / n_feat), "reg_lambda": 0.0}, 1
        return {**base, "objective": "reg:pseudohubererror", "huber_slope": xi,
                "max_depth": hp["max_depth"], "learning_rate": hp["learning_rate"],
                "eval_metric": "rmse"}, 1000

    def fit(self, panel: Panel, train: slice, val: slice, tune: bool = True):
        import xgboost as xgb
        Xtr, ytr = raw_matrix(*panel.raw(train)), panel.y[train]
        Xva, yva = raw_matrix(*panel.raw(val)), panel.y[val]
        dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=64)
        dva = xgb.DMatrix(Xva, yva)
        xi = float(np.quantile(np.abs(ytr - ytr.mean()), 0.999))   # Huber threshold
        self.hparams["huber_xi"] = xi
        combos = [dict(zip(self.grid, v)) for v in itertools.product(*self.grid.values())]
        if not tune and self.best is not None:
            combos = [self.best]
        best_mse, best_b, best_hp = np.inf, None, None
        for hp in combos:
            params, rounds = self._params(hp, Xtr.shape[1], xi)
            if self.kind == "rf":
                b = xgb.train(params, dtr, num_boost_round=1)
            else:
                b = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "val")],
                              early_stopping_rounds=50, verbose_eval=False)
            pv = b.predict(dva, iteration_range=(0, b.best_iteration + 1)) if self.kind != "rf" else b.predict(dva)
            mse = float(np.mean((yva - pv) ** 2))
            print(f"    {self.kind} {hp}: val MSE {mse:.6f}"
                  + (f", trees {b.best_iteration + 1}" if self.kind != "rf" else ""))
            if mse < best_mse:
                best_mse, best_b, best_hp = mse, b, hp
        self.booster, self.best = best_b, best_hp
        self.hparams.update({**best_hp, "val_mse": best_mse})
        self.hparams["n_features_used"] = len(best_b.get_score(importance_type="weight"))
        if self.kind != "rf":
            self.hparams["n_trees"] = int(best_b.best_iteration + 1)
        return self

    def predict_raw(self, C, Mx, S):
        import xgboost as xgb
        d = xgb.DMatrix(raw_matrix(C, Mx, S))
        if self.kind == "rf":
            return self.booster.predict(d)
        return self.booster.predict(d, iteration_range=(0, self.booster.best_iteration + 1))

    def predict_rows(self, panel: Panel, sl: slice):
        return self.predict_raw(*panel.raw(sl))
