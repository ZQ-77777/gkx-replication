"""In-memory panel and the paper's feature map.

Features (paper eq. 21):  z_{i,t} = x_t (x) c_{i,t}  plus industry dummies, where
x_t = (1, 8 macro predictors) and c_{i,t} = 94 ranked characteristics:
94 x 9 = 846 interaction terms + 74 sic2 dummies = 920 inputs.

The 920 columns are never stored: every model receives the raw blocks
(C = characteristics, Mx = macro values of each row's info month, S = sic index)
and builds the features chunk by chunk. This keeps memory at ~1.5 GB instead of ~14 GB,
and makes variable importance easy (zero a raw input, rebuild, predict).
"""
import numpy as np
import pandas as pd

from .config import CHARS, MACRO, OLS3_CHARS, Config


class _NotSorted(Exception):
    pass


class Panel:
    """Stock-month panel in memory: characteristics C, macro values per info month, sic index S,
    excess returns y. Only rows from cfg.first_year (1957) on are kept: no model uses earlier data.

    Memory-light loading: the parquet file is read one column (per row group) at a time into a
    preallocated float32 array, so the peak is ~1.5 GB instead of ~5 GB for pandas.read_parquet.
    Falls back to pandas if the file is not sorted by (ret_month, permno) as written by build_panel.
    """
    META = ["permno", "ret_month", "info_month", "ret_exc", "size_raw", "sic2"]

    def __init__(self, cfg: Config, chars=None, first_year=None):
        first_year = cfg.first_year if first_year is None else first_year
        try:
            meta, C, self.chars = self._read_lowmem(cfg.panel_path, chars, first_year)
        except _NotSorted:
            meta, C, self.chars = self._read_pandas(cfg.panel_path, chars, first_year)
        self.C = C
        macro = pd.read_csv(cfg.macro_path, index_col=0)
        macro.index = pd.PeriodIndex(macro.index, freq="M").to_timestamp("M")
        self.macro_names = list(MACRO)
        self.macro = macro[MACRO].to_numpy(np.float32)
        pos = pd.Series(np.arange(len(macro)), index=macro.index)
        info = pd.to_datetime(meta["info_month"]).dt.to_period("M").dt.to_timestamp("M")
        self.month_idx = pos.reindex(info).to_numpy()
        if np.isnan(self.month_idx.astype(float)).any():
            raise ValueError("some info months have no macro data")
        self.month_idx = self.month_idx.astype(np.int32)
        self.sic_codes = np.sort(meta.loc[meta.sic2 >= 0, "sic2"].unique())
        lut = {c: i for i, c in enumerate(self.sic_codes)}
        self.S = meta["sic2"].map(lut).fillna(-1).to_numpy(np.int32)
        self.n_sic = len(self.sic_codes)
        self.y = meta["ret_exc"].to_numpy(np.float32)
        self.size = meta["size_raw"].to_numpy(np.float32)
        self.date = meta["ret_month"].to_numpy()
        self.permno = meta["permno"].to_numpy()
        self.year = pd.to_datetime(meta["ret_month"]).dt.year.to_numpy()
        self.n = len(meta)
        print(f"panel: {self.n:,} rows | {len(self.chars)} chars | {len(MACRO)} macro | {self.n_sic} sic2 codes "
              f"| {self.n_features} features | {self.year.min()}-{self.year.max()}")

    @staticmethod
    def _read_lowmem(path, chars, first_year):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        names = list(pf.schema_arrow.names)
        chars = chars or [c for c in CHARS if c in names]
        meta = pf.read(columns=Panel.META).to_pandas()
        rm = pd.to_datetime(meta["ret_month"])
        key = (rm.dt.year.to_numpy().astype(np.int64) * 100 + rm.dt.month.to_numpy()) * 1_000_000 \
            + meta["permno"].to_numpy().astype(np.int64)
        if not (np.diff(key) >= 0).all():
            raise _NotSorted()
        start = int(np.searchsorted(rm.dt.year.to_numpy(), first_year, side="left"))
        meta = meta.iloc[start:].reset_index(drop=True)
        C = np.empty((len(meta), len(chars)), np.float32)
        lo = 0
        for g in range(pf.num_row_groups):
            hi = lo + pf.metadata.row_group(g).num_rows
            a, b = max(lo, start), hi                       # rows of this group that are kept
            if b > a:
                for j, c in enumerate(chars):
                    col = pf.read_row_group(g, columns=[c]).column(0).to_numpy()
                    C[a - start:b - start, j] = col[a - lo:b - lo]
            lo = hi
        if lo != len(meta) + start:
            raise ValueError("row groups do not add up to the number of rows")
        return meta, C, chars

    @staticmethod
    def _read_pandas(path, chars, first_year):
        df = pd.read_parquet(path)
        df = df.sort_values(["ret_month", "permno"]).reset_index(drop=True)
        df = df[pd.to_datetime(df["ret_month"]).dt.year >= first_year].reset_index(drop=True)
        chars = chars or [c for c in CHARS if c in df.columns]
        return df[Panel.META], df[chars].to_numpy(np.float32), chars

    @property
    def n_features(self) -> int:
        return n_features(len(self.chars), self.n_sic)

    def rows(self, years) -> slice:
        """Contiguous row slice for an inclusive (first_year, last_year) range."""
        a = np.searchsorted(self.year, years[0], side="left")
        b = np.searchsorted(self.year, years[1], side="right")
        return slice(int(a), int(b))

    def raw(self, idx):
        """(C, Mx, S) for a slice or index array."""
        return self.C[idx], self.macro[self.month_idx[idx]], self.S[idx]

    def feature_names(self):
        names = [f"{c}" if k == 0 else f"{c}*{self.macro_names[k - 1]}"
                 for c in self.chars for k in range(len(self.macro_names) + 1)]
        return names + [f"sic_{s}" for s in self.sic_codes]

    def ols3_index(self):
        return [self.chars.index(c) for c in OLS3_CHARS]


def n_features(n_char: int, n_sic: int, n_macro: int = len(MACRO)) -> int:
    return n_char * (n_macro + 1) + n_sic


def build_np(C, Mx, S, n_sic: int) -> np.ndarray:
    """numpy version: (B,94),(B,8),(B,) -> (B, 94*9 + n_sic), char-major ordering."""
    B = len(C)
    xa = np.concatenate([np.ones((B, 1), np.float32), Mx.astype(np.float32)], 1)
    inter = (C[:, :, None] * xa[:, None, :]).reshape(B, -1)
    dummies = np.zeros((B, n_sic), np.float32)
    ok = S >= 0
    dummies[np.flatnonzero(ok), S[ok]] = 1.0
    return np.concatenate([inter, dummies], 1)


def build_torch(C, Mx, S, n_sic: int):
    """torch version of build_np (same ordering), for GPU batches."""
    import torch
    B = C.shape[0]
    xa = torch.cat([torch.ones(B, 1, device=C.device, dtype=C.dtype), Mx], 1)
    inter = (C.unsqueeze(2) * xa.unsqueeze(1)).reshape(B, -1)
    dummies = torch.zeros(B, n_sic + 1, device=C.device, dtype=C.dtype)
    dummies[torch.arange(B, device=C.device), torch.where(S >= 0, S, n_sic).long()] = 1.0
    return torch.cat([inter, dummies[:, :n_sic]], 1)


def ols3_np(C, Mx, S, idx3) -> np.ndarray:
    """OLS-3 benchmark features: size, book-to-market, momentum (no interactions)."""
    return C[:, idx3].astype(np.float32)
