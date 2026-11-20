"""Linear models fitted from streamed sufficient statistics.

All of OLS, Huber regression, elastic net, PCR and PLS only need
    Z'WZ, Z'Wy (Z = [1, features], W = Huber weights or identity),
so we stream the panel in chunks, build the 920 features on the fly and accumulate
a 921 x 921 Gram matrix. On a GPU (torch) one pass over ~3.5M rows takes seconds.

Hyper-parameters (paper Table A.5): elastic net rho = 0.5, lambda in (1e-4, 1e-1);
PCR / PLS number of components chosen on the validation sample; Huber threshold xi =
99.9% quantile of absolute residuals (paper Section 1.1).
"""
import numpy as np

from .panel import Panel, build_np, build_torch, ols3_np


# ---------------------------------------------------------------------------
# streaming backend
# ---------------------------------------------------------------------------
# Precision of the GPU path. The team runs used float32 matrix products (accumulated into a float64 Gram
# matrix per chunk). Set to True for float64 products throughout (slower on consumer GPUs, but exact for
# nearly collinear inputs such as the 920-input OLS). The CPU (numpy) path is always float64.
GPU_FLOAT64 = False


def _torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:   # ImportError, or OSError from a broken torch install on Windows -> use numpy (CPU)
        return False


class FeatureMap:
    """Callable producing the design matrix for raw blocks; 'full' (920) or 'ols3' (3)."""
    def __init__(self, panel: Panel, kind: str = "full"):
        self.kind, self.n_sic = kind, panel.n_sic
        self.idx3 = panel.ols3_index()
        self.p = panel.n_features if kind == "full" else 3

    def np(self, C, Mx, S):
        return build_np(C, Mx, S, self.n_sic) if self.kind == "full" else ols3_np(C, Mx, S, self.idx3)

    def torch(self, C, Mx, S):
        if self.kind == "full":
            return build_torch(C, Mx, S, self.n_sic)
        return C[:, self.idx3]


def _chunks(sl: slice, size: int):
    for a in range(sl.start, sl.stop, size):
        yield slice(a, min(a + size, sl.stop))


def stream_gram(panel: Panel, sl: slice, fmap: FeatureMap, chunk: int, beta=None, xi=None, use_gpu=None):
    """Accumulate Z'WZ, Z'Wy, sum(W) over rows `sl`.
    If beta and xi are given, W = Huber IRLS weights min(1, xi/|y - Z beta|)."""
    use_gpu = _torch_cuda() if use_gpu is None else use_gpu
    p1 = fmap.p + 1
    if use_gpu:
        import torch
        dev = torch.device("cuda")
        G = torch.zeros(p1, p1, dtype=torch.float64, device=dev)
        b = torch.zeros(p1, dtype=torch.float64, device=dev)
        fdt = torch.float64 if GPU_FLOAT64 else torch.float32
        bt = None if beta is None else torch.as_tensor(beta, dtype=fdt, device=dev)
        wsum = 0.0
        for c in _chunks(sl, chunk):
            C, Mx, S = panel.raw(c)
            Ct, Mt, St = (torch.as_tensor(C, device=dev), torch.as_tensor(Mx, device=dev),
                          torch.as_tensor(S, device=dev))
            Z = fmap.torch(Ct, Mt, St)
            Z = torch.cat([torch.ones(len(Z), 1, device=dev), Z], 1)
            y = torch.as_tensor(panel.y[c], device=dev)
            if bt is not None:
                r = (y.to(fdt) - Z.to(fdt) @ bt).abs().float()
                w = torch.clamp(xi / torch.clamp(r, min=1e-12), max=1.0)
            else:
                w = torch.ones_like(y)
            Zw = Z * w.unsqueeze(1)
            if GPU_FLOAT64:
                G += Zw.double().T @ Z.double()
                b += Zw.double().T @ y.double()
            else:
                G += (Zw.T @ Z).double()
                b += (Zw.T @ y).double()
            wsum += float(w.sum())
        return G.cpu().numpy(), b.cpu().numpy(), wsum
    G = np.zeros((p1, p1)); b = np.zeros(p1); wsum = 0.0
    for c in _chunks(sl, chunk):
        Z = fmap.np(*panel.raw(c))
        Z = np.hstack([np.ones((len(Z), 1), np.float32), Z])
        y = panel.y[c]
        w = np.ones_like(y) if beta is None else np.minimum(1.0, xi / np.maximum(np.abs(y - Z @ beta), 1e-12))
        Zw = Z * w[:, None]
        G += Zw.T.astype(np.float64) @ Z
        b += Zw.T.astype(np.float64) @ y
        wsum += w.sum()
    return G, b, wsum


def stream_predict(panel_or_raw, sl, fmap: FeatureMap, B: np.ndarray, chunk: int, use_gpu=None) -> np.ndarray:
    """Predictions Z @ B for one or many coefficient vectors (B: p+1 or (p+1) x L).
    panel_or_raw is a Panel (rows `sl`) or a tuple (C, Mx, S) of raw arrays."""
    use_gpu = _torch_cuda() if use_gpu is None else use_gpu
    if isinstance(panel_or_raw, tuple):
        C_all, M_all, S_all = panel_or_raw
        get = lambda c: (C_all[c], M_all[c], S_all[c])
        sl = slice(0, len(C_all))
    else:
        get = panel_or_raw.raw
    B2 = B if B.ndim == 2 else B[:, None]
    out = np.zeros((sl.stop - sl.start, B2.shape[1]), np.float64)
    if use_gpu:
        import torch
        dev = torch.device("cuda")
        fdt = torch.float64 if GPU_FLOAT64 else torch.float32
        Bt = torch.as_tensor(B2, dtype=fdt, device=dev)
        for c in _chunks(sl, chunk):
            C, Mx, S = get(c)
            Z = fmap.torch(torch.as_tensor(C, device=dev), torch.as_tensor(Mx, device=dev), torch.as_tensor(S, device=dev))
            out[c.start - sl.start:c.stop - sl.start] = (Z.to(fdt) @ Bt[1:] + Bt[0]).double().cpu().numpy()
    else:
        for c in _chunks(sl, chunk):
            Z = fmap.np(*get(c))
            out[c.start - sl.start:c.stop - sl.start] = Z @ B2[1:] + B2[0]
    return out if B.ndim == 2 else out[:, 0]


def stream_residuals(panel, sl, fmap, beta, chunk):
    return panel.y[sl] - stream_predict(panel, sl, fmap, beta, chunk)


# ---------------------------------------------------------------------------
# solvers on the Gram matrix
# ---------------------------------------------------------------------------
def solve_ols(G, b, ridge=1e-8):
    p1 = len(b)
    reg = ridge * np.trace(G) / p1 * np.eye(p1)
    reg[0, 0] = 0.0
    return np.linalg.solve(G + reg, b)


def standardize(G, b, n):
    """Means, standard deviations, correlation-scale covariance and cross-covariance
    of the features (dropping the intercept), from the (weighted) Gram matrix."""
    mu = G[0, 1:] / n
    ybar = b[0] / n
    cov = G[1:, 1:] / n - np.outer(mu, mu)
    cxy = b[1:] / n - mu * ybar
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    keep = sd > 1e-10
    sdk = np.where(keep, sd, 1.0)
    A = cov / np.outer(sdk, sdk)
    c = cxy / sdk
    A[~keep, :] = 0; A[:, ~keep] = 0; c[~keep] = 0
    return mu, ybar, sdk, keep, A, c


def unstandardize(beta_std, mu, ybar, sd):
    beta = beta_std / sd
    return np.concatenate([[ybar - mu @ beta], beta])


def enet_cd(A, c, lam, rho=0.5, beta0=None, keep=None, tol=1e-7, max_iter=200):
    """Coordinate descent for  0.5 b'Ab - c'b + lam*(rho*|b|_1 + (1-rho)/2*|b|_2^2)
    on standardised features (diag(A) = 1). Full sweeps alternate with sweeps over
    the active set (non-zero coefficients) until a full sweep changes nothing."""
    p = len(c)
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    q = A @ beta
    full = np.flatnonzero(keep) if keep is not None else np.arange(p)
    denom = np.diag(A) + lam * (1 - rho)
    thr = lam * rho

    def sweep(idx):
        max_d = 0.0
        for j in idx:
            rj = c[j] - q[j] + A[j, j] * beta[j]
            new = np.sign(rj) * max(abs(rj) - thr, 0.0) / denom[j]
            d = new - beta[j]
            if d != 0.0:
                q[:] += A[:, j] * d
                beta[j] = new
                max_d = max(max_d, abs(d))
        return max_d

    for _ in range(max_iter):
        if sweep(full) < tol:
            break
        for _ in range(max_iter):
            if sweep(np.flatnonzero(beta)) < tol:
                break
    return beta


def pcr_coefs(A, c, ks):
    """PCR on standardised features: regress y on the first K principal components."""
    vals, vecs = np.linalg.eigh(A)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    out = []
    for k in ks:
        V, L = vecs[:, :k], np.clip(vals[:k], 1e-12, None)
        out.append(V @ ((V.T @ c) / L))
    return np.array(out).T


def pls_coefs(A, c, kmax):
    """PLS1 via an A-orthonormal Krylov basis of {c, Ac, A^2 c, ...}:
    with R'AR = I the K-component PLS coefficient is R_K R_K' c."""
    R, out = [], []
    r = c.copy()
    for k in range(kmax):
        for _ in range(2):                         # re-orthogonalise for stability
            for v in R:
                r = r - (v @ (A @ r)) * v
        nrm = np.sqrt(max(r @ A @ r, 0.0))
        if nrm < 1e-12:
            out.append(out[-1] if out else np.zeros_like(c))
            continue
        r = r / nrm
        R.append(r)
        Rm = np.array(R).T
        out.append(Rm @ (Rm.T @ c))
        r = A @ r
    return np.array(out).T


# ---------------------------------------------------------------------------
# model classes (common API: fit(panel, train, val) -> self; predict_raw(C, Mx, S))
# ---------------------------------------------------------------------------
class LinearModel:
    """kind: 'ols', 'enet', 'pcr', 'pls'; huber: robust Huber loss via IRLS; features: 'full' or 'ols3'."""

    def __init__(self, kind="ols", huber=False, features="full", chunk=250_000,
                 lambdas=None, rho=0.5, pcr_ks=None, pls_kmax=30, irls_iter=5, xi_q=0.999):
        self.kind, self.huber, self.features, self.chunk = kind, huber, features, chunk
        self.lambdas = np.logspace(-1, -4, 13) if lambdas is None else np.asarray(lambdas)
        self.rho, self.pls_kmax, self.irls_iter, self.xi_q = rho, pls_kmax, irls_iter, xi_q
        self.pcr_ks = pcr_ks or [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100]
        self.beta, self.hparams = None, {}

    # -- helpers --
    def _candidates(self, G, b, n):
        """Coefficient vectors (original scale, with intercept) for each hyper-parameter value."""
        if self.kind == "ols":
            return solve_ols(G, b)[:, None], [None]
        mu, ybar, sd, keep, A, c = standardize(G, b, n)
        if self.kind == "enet":
            cols, beta = [], None
            for lam in self.lambdas:
                beta = enet_cd(A, c, lam, self.rho, beta, keep)
                cols.append(unstandardize(beta, mu, ybar, sd))
            return np.array(cols).T, list(self.lambdas)
        if self.kind == "pcr":
            ks = [k for k in self.pcr_ks if k <= keep.sum()]
            Bs = pcr_coefs(A, c, ks)
            return np.array([unstandardize(Bs[:, i], mu, ybar, sd) for i in range(len(ks))]).T, ks
        if self.kind == "pls":
            Bs = pls_coefs(A, c, self.pls_kmax)
            return np.array([unstandardize(Bs[:, i], mu, ybar, sd) for i in range(Bs.shape[1])]).T, \
                list(range(1, Bs.shape[1] + 1))
        raise ValueError(self.kind)

    def fit(self, panel: Panel, train: slice, val: slice, tune: bool = True):
        self.fmap = FeatureMap(panel, self.features)
        G, b, n = stream_gram(panel, train, self.fmap, self.chunk)
        if self.huber:
            # Huber loss via iteratively reweighted least squares (IRLS):
            # w_i = min(1, xi/|r_i|), xi = 99.9% quantile of |OLS residuals|.
            # Weights come from the unpenalised Huber fit; penalised / reduced-rank models are
            # then fitted on the Huber-weighted Gram (one-step approximation of ENet+H etc.).
            beta = solve_ols(G, b)
            r = np.abs(stream_residuals(panel, train, self.fmap, beta, self.chunk))
            xi = float(np.quantile(r, self.xi_q))
            self.hparams["huber_xi"] = xi
            for _ in range(self.irls_iter):
                G, b, n = stream_gram(panel, train, self.fmap, self.chunk, beta=beta, xi=xi)
                beta = solve_ols(G, b)
        B, grid = self._candidates(G, b, n)
        if B.shape[1] > 1:
            pv = stream_predict(panel, val, self.fmap, B, self.chunk)
            mse = ((panel.y[val][:, None] - pv) ** 2).mean(0)
            best = int(np.argmin(mse))
            self.hparams.update({"param": grid[best], "val_mse": float(mse[best]),
                                 "grid": [float(g) for g in grid], "val_mse_path": mse.tolist()})
        else:
            best = 0
        self.beta = B[:, best]
        if self.kind == "enet":
            self.hparams["n_nonzero"] = int((self.beta[1:] != 0).sum())
        return self

    def predict_raw(self, C, Mx, S):
        return stream_predict((C, Mx, S), None, self.fmap, self.beta, self.chunk)

    def predict_rows(self, panel: Panel, sl: slice):
        return stream_predict(panel, sl, self.fmap, self.beta, self.chunk)
