"""Build the modelling panel (run once, then share data/panel.parquet with the team).

Steps (paper Section 2.1 and Internet Appendix A):
1. read the 94 characteristics + sic2 from datashare.csv;
2. attach monthly stock returns (from datashare if present, otherwise CRSP) and check the timing;
3. excess return = return - risk-free rate (Welch-Goyal Rfree);
4. rank each characteristic cross-sectionally every month into [-1, 1]; missing -> 0 (= median);
5. build the 8 Welch-Goyal macro predictors, lagged so they are known when the forecast is made.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CHARS, MACRO, Config

RET_CANDIDATES = ["RET", "ret", "Ret", "ret_exc", "RET_EXC", "excess_ret", "ret_excess"]


# ---------------------------------------------------------------------------
# 1-2. characteristics and returns
# ---------------------------------------------------------------------------
def read_datashare(path, usecols=None) -> pd.DataFrame:
    """Read datashare.csv with float32 characteristics (~3.7M rows; ~2 GB in RAM)."""
    head = pd.read_csv(path, nrows=5)
    cols = list(head.columns)
    missing = [c for c in CHARS if c not in cols]
    if missing:
        print("WARNING: characteristics not found in file:", missing)
    dtypes = {c: "float32" for c in cols if c not in ("permno", "DATE", "sic2")}
    df = pd.read_csv(path, dtype=dtypes, usecols=usecols)
    df["month"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d").dt.to_period("M")
    df["permno"] = df["permno"].astype("int64")
    print(f"read {len(df):,} rows, {df.permno.nunique():,} stocks, "
          f"{df.month.min()} to {df.month.max()}; columns: {len(cols)}")
    return df


def _month_int(month: pd.Series) -> pd.Series:
    """Period('M') -> consecutive integers (year*12 + month)."""
    return month.dt.year * 12 + month.dt.month


def find_return_column(df: pd.DataFrame):
    for c in RET_CANDIDATES:
        if c in df.columns:
            return c
    return None


def attach_crsp_returns(df: pd.DataFrame, crsp_csv: str) -> pd.DataFrame:
    """Merge CRSP monthly returns (WRDS crsp.msf + msedelist), delisting-adjusted.
    Expected columns: permno, date, ret, optional dlret. See README for the WRDS query."""
    c = pd.read_csv(crsp_csv)
    c.columns = [x.lower() for x in c.columns]
    c["ret"] = pd.to_numeric(c["ret"], errors="coerce")
    if "dlret" in c.columns:
        dl = pd.to_numeric(c["dlret"], errors="coerce")
        c["ret"] = np.where(dl.notna(), (1 + c["ret"].fillna(0)) * (1 + dl) - 1, c["ret"])
    c["month"] = pd.to_datetime(c["date"]).dt.to_period("M")
    c = c.groupby(["permno", "month"], as_index=False)["ret"].last().rename(columns={"ret": "RET"})
    return df.merge(c, on=["permno", "month"], how="left")


def check_alignment(df: pd.DataFrame, ret_col: str, sample_permnos: int = 3000, seed: int = 0) -> dict:
    """Decide which return a row's characteristics should predict.

    mom1m (one-month momentum) is the most recent past monthly return known when the
    characteristics are observed. Compare
      a = corr(mom1m_t,   ret_t)   high  -> ret_t is already known -> target is ret_{t+1}
      b = corr(mom1m_{t+1}, ret_t) high  -> ret_t is the next return -> target is ret_t
    (rank correlations on a random subset of stocks)."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(df["permno"].unique(), min(sample_permnos, df["permno"].nunique()), replace=False)
    d = df.loc[df["permno"].isin(ids), ["permno", "month", "mom1m", ret_col]].sort_values(["permno", "month"])
    nxt = d.groupby("permno")["mom1m"].shift(-1)
    mi = _month_int(d["month"])
    consecutive = (mi.groupby(d["permno"]).shift(-1) - mi) == 1
    a = d[["mom1m", ret_col]].corr("spearman").iloc[0, 1]
    b = pd.concat([nxt[consecutive], d.loc[consecutive, ret_col]], axis=1).corr("spearman").iloc[0, 1]
    shift = int(a > b)        # 1 -> use next month's return as the target
    print(f"corr(mom1m_t, ret_t) = {a:.3f} | corr(mom1m_t+1, ret_t) = {b:.3f} -> "
          + ("target = NEXT month's return (shift -1)" if shift else "target = SAME row's return (no shift)"))
    return {"corr_same": a, "corr_next": b, "shift": shift}


def make_target(df: pd.DataFrame, ret_col: str, shift: int) -> pd.DataFrame:
    """Create ret_month (month of the return to forecast), info_month (= ret_month - 1, when
    the forecast is made) and the raw target return `ret`."""
    df = df.sort_values(["permno", "month"]).copy()
    if shift:
        g = df.groupby("permno")
        mi = _month_int(df["month"])
        ok = (mi.groupby(df["permno"]).shift(-1) - mi) == 1
        df["ret"] = g[ret_col].shift(-1).where(ok)
        df["ret_month"] = df["month"] + 1
    else:
        df["ret"] = df[ret_col]
        df["ret_month"] = df["month"]
    df["info_month"] = df["ret_month"] - 1
    return df.dropna(subset=["ret"])


# ---------------------------------------------------------------------------
# 3 & 5. Welch-Goyal macro predictors and risk-free rate
# ---------------------------------------------------------------------------
def welch_goyal(path: str) -> pd.DataFrame:
    """8 macro predictors following Welch & Goyal (2008), as used by GKX:
    dp = log D12 - log P, ep = log E12 - log P, bm = book-to-market (DJIA), ntis = net equity
    expansion, tbl = T-bill rate, tms = lty - tbl, dfy = BAA - AAA, svar = stock variance.
    Returns a monthly table indexed by Period('M') with MACRO columns and rf (monthly Rfree)."""
    if str(path).endswith(".csv"):
        w = pd.read_csv(path)
    else:
        w = pd.read_excel(path, sheet_name="Monthly")
    w.columns = [str(c).strip() for c in w.columns]
    num = lambda c: pd.to_numeric(w[c].astype(str).str.replace(",", ""), errors="coerce")
    out = pd.DataFrame({
        "dp": np.log(num("D12")) - np.log(num("Index")),
        "ep_m": np.log(num("E12")) - np.log(num("Index")),
        "bm_m": num("b/m"),
        "ntis": num("ntis"),
        "tbl": num("tbl"),
        "tms": num("lty") - num("tbl"),
        "dfy": num("BAA") - num("AAA"),
        "svar": num("svar"),
        "rf": num("Rfree"),
    })
    out.index = pd.PeriodIndex(w["yyyymm"].astype(int).astype(str), freq="M")
    out.index.name = "month"
    return out


# ---------------------------------------------------------------------------
# 4. cross-sectional rank normalisation
# ---------------------------------------------------------------------------
def rank_normalize(df: pd.DataFrame, cols, by="info_month") -> pd.DataFrame:
    """Map each characteristic's cross-sectional rank into [-1, 1]; missing -> 0."""
    g = df.groupby(by)
    for i, c in enumerate(cols):
        r = g[c].rank(method="average")
        n = g[c].transform("count")
        x = (2 * (r - 1) / (n - 1).where(n > 1) - 1).fillna(0.0)
        df[c] = x.astype("float32")
        if (i + 1) % 20 == 0:
            print(f"  ranked {i + 1}/{len(cols)}")
    return df


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------
def build_panel(cfg: Config, shift: int = None) -> pd.DataFrame:
    df = read_datashare(cfg.raw_csv)
    ret_col = find_return_column(df)
    if ret_col is None:
        if cfg.crsp_returns is None:
            raise ValueError("No return column in datashare; download CRSP returns from WRDS and set cfg.crsp_returns")
        df = attach_crsp_returns(df, cfg.crsp_returns)
        ret_col = "RET"
    if shift is None:
        al = check_alignment(df, ret_col)
        shift = al["shift"]
        Path(cfg.panel_path).parent.mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.panel_path).parent / "alignment.json", "w") as f:
            json.dump({"return_column": ret_col, **{k: float(v) for k, v in al.items()}}, f, indent=1)
    df = make_target(df, ret_col, shift)

    macro = welch_goyal(cfg.welch_goyal)
    Path(cfg.macro_path).parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(cfg.macro_path)
    df = df.merge(macro[["rf"]], left_on="ret_month", right_index=True, how="left")
    df["ret_exc"] = (df["ret"] - df["rf"]).astype("float32")
    df = df[df["info_month"].isin(macro.dropna(subset=MACRO).index)]

    df["size_raw"] = df["mvel1"].astype("float32")      # keep raw size for top/bottom-1000 and value weights
    chars = [c for c in CHARS if c in df.columns]
    print("rank-normalising characteristics ...")
    df = rank_normalize(df, chars)
    df["sic2"] = df["sic2"].fillna(-1).astype("int16")

    keep = ["permno", "ret_month", "info_month", "ret_exc", "size_raw", "sic2"] + chars
    df = df[keep].sort_values(["ret_month", "permno"]).reset_index(drop=True)
    df["ret_month"] = df["ret_month"].dt.to_timestamp("M")
    df["info_month"] = df["info_month"].dt.to_timestamp("M")
    Path(cfg.panel_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.panel_path, index=False)
    print(f"saved {cfg.panel_path}: {len(df):,} rows, {df.ret_month.min():%Y-%m} to {df.ret_month.max():%Y-%m}")
    return df
