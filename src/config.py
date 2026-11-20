"""Paths, sample splits and constants for the replication of
Gu, Kelly & Xiu (2020, RFS), "Empirical Asset Pricing via Machine Learning"."""
from dataclasses import dataclass
from typing import Optional, Tuple

# The 94 firm characteristics of the paper (column names in datashare.csv)
CHARS = [
    "absacc", "acc", "aeavol", "age", "agr", "baspread", "beta", "betasq", "bm", "bm_ia",
    "cash", "cashdebt", "cashpr", "cfp", "cfp_ia", "chatoia", "chcsho", "chempia", "chinv", "chmom",
    "chpmia", "chtx", "cinvest", "convind", "currat", "depr", "divi", "divo", "dolvol", "dy",
    "ear", "egr", "ep", "gma", "grcapx", "grltnoa", "herf", "hire", "idiovol", "ill",
    "indmom", "invest", "lev", "lgr", "maxret", "mom12m", "mom1m", "mom36m", "mom6m", "ms",
    "mvel1", "mve_ia", "nincr", "operprof", "orgcap", "pchcapx_ia", "pchcurrat", "pchdepr", "pchgm_pchsale", "pchquick",
    "pchsale_pchinvt", "pchsale_pchrect", "pchsale_pchxsga", "pchsaleinv", "pctacc", "pricedelay", "ps", "quick", "rd", "rd_mve",
    "rd_sale", "realestate", "retvol", "roaq", "roavol", "roeq", "roic", "rsup", "salecash", "saleinv",
    "salerec", "secured", "securedind", "sgr", "sin", "sp", "std_dolvol", "std_turn", "stdacc", "stdcf",
    "tang", "tb", "turn", "zerotrade",
]
assert len(CHARS) == 94

# The 8 macro predictors of Welch & Goyal (2008) used in the paper
MACRO = ["dp", "ep_m", "bm_m", "ntis", "tbl", "tms", "dfy", "svar"]   # _m avoids clashes with firm ep/bm

# OLS-3 benchmark: size, book-to-market, 12-month momentum
OLS3_CHARS = ["mvel1", "bm", "mom12m"]


@dataclass
class Config:
    # ---- inputs ----
    raw_csv: str = "data/raw/datashare.csv"                 # from datashare_OLD.zip
    welch_goyal: str = "data/raw/PredictorData2023.xlsx"    # Amit Goyal's website, sheet "Monthly"
    crsp_returns: Optional[str] = None                     # csv(permno,date,ret[,dlret]) if datashare has no returns
    # ---- processed ----
    panel_path: str = "data/panel.parquet"
    macro_path: str = "data/macro.csv"
    out_dir: str = "outputs"
    # ---- recursive scheme (paper Section 2.1) ----
    first_year: int = 1957
    first_train_end: int = 1974          # 18 training years for the first refit
    val_years: int = 12                  # rolling 12-year validation window
    test_start: int = 1987
    test_end: int = 2016
    refit_every: int = 1                 # refit annually (paper); 2-3 saves time for neural nets
    # ---- computation ----
    chunk: int = 250_000                 # rows per chunk when streaming features
    seed: int = 42
    # ---- variable importance ----
    vi_years: Tuple[int, ...] = (1987, 1992, 1997, 2002, 2007, 2012)   # refits at which VI is computed
    vi_sample: int = 300_000             # training rows used for VI (random subsample)

    def refit_years(self):
        return list(range(self.test_start, self.test_end + 1, self.refit_every))

    def split(self, Y: int):
        """Year ranges (inclusive) for the refit that forecasts years Y .. Y+refit_every-1.
        Y=1987 -> train 1957-1974, validation 1975-1986, test 1987."""
        val_end = Y - 1
        val_start = val_end - self.val_years + 1
        train = (self.first_year, val_start - 1)
        test = (Y, min(Y + self.refit_every - 1, self.test_end))
        return train, (val_start, val_end), test
