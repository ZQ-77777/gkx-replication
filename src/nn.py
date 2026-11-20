"""Feed-forward networks NN1-NN5 (paper Section 1.7 and Table A.5).

NN1: 32 | NN2: 32-16 | NN3: 32-16-8 | NN4: 32-16-8-4 | NN5: 32-16-8-4-2 hidden units,
ReLU, batch normalisation, L1 penalty on weights, Adam, batch size 10,000, up to 100 epochs,
early stopping (patience 5) on validation MSE, ensemble of independently seeded networks.
Paper grid: L1 penalty in (1e-5, 1e-3), learning rate in (0.001, 0.01); ensemble of 10.

The whole panel (characteristics, sic index, returns) is moved to the GPU once (~1.5 GB) and the
920 inputs are built per mini-batch on the GPU.
"""
import itertools
import time

import numpy as np
import torch
import torch.nn as nn

from .panel import Panel, build_torch

ARCH = {"NN1": [32], "NN2": [32, 16], "NN3": [32, 16, 8], "NN4": [32, 16, 8, 4], "NN5": [32, 16, 8, 4, 2]}


class MLP(nn.Module):
    def __init__(self, p, hidden):
        super().__init__()
        layers, d = [], p
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

    def l1(self):
        return sum(m.weight.abs().sum() for m in self.modules() if isinstance(m, nn.Linear))


class GPUPanel:
    """Torch copies of the panel arrays (shared by all networks)."""
    _cache = {}

    def __init__(self, panel: Panel, device):
        key = (id(panel), str(device))
        if key not in GPUPanel._cache:
            GPUPanel._cache[key] = dict(
                C=torch.as_tensor(panel.C, device=device),
                macro=torch.as_tensor(panel.macro, device=device),
                midx=torch.as_tensor(panel.month_idx, device=device, dtype=torch.long),
                S=torch.as_tensor(panel.S, device=device, dtype=torch.long),
                y=torch.as_tensor(panel.y, device=device))
        self.__dict__.update(GPUPanel._cache[key])
        self.n_sic = panel.n_sic

    def features(self, idx):
        return build_torch(self.C[idx], self.macro[self.midx[idx]], self.S[idx], self.n_sic)


def _mse(model, gp: GPUPanel, sl: slice, bs=100_000):
    model.eval()
    tot, n = 0.0, sl.stop - sl.start
    with torch.no_grad():
        for a in range(sl.start, sl.stop, bs):
            idx = torch.arange(a, min(a + bs, sl.stop), device=gp.C.device)
            tot += float(((model(gp.features(idx)) - gp.y[idx]) ** 2).sum())
    return tot / n


def train_net(panel, gp, train, val, hidden, lr, l1, seed, batch=10_000, epochs=100, patience=5, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = gp.C.device
    model = MLP(panel.n_features, hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train.stop - train.start
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev) + train.start
        for a in range(0, n, batch):
            idx = perm[a:a + batch]
            pred = model(gp.features(idx))
            loss = ((pred - gp.y[idx]) ** 2).mean() + l1 * model.l1()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = _mse(model, gp, val)
        if verbose:
            print(f"      epoch {ep + 1}: val MSE {v:.6f}")
        if v < best - 1e-9:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), best, ep + 1


class NNModel:
    def __init__(self, name="NN3", lrs=(1e-3, 1e-2), l1s=(1e-5, 1e-4, 1e-3), n_ensemble=10, seed=42,
                 batch=10_000, epochs=100, patience=5):
        self.name, self.hidden = name, ARCH[name]
        self.grid = list(itertools.product(lrs, l1s))
        self.n_ensemble, self.seed = n_ensemble, seed
        self.batch, self.epochs, self.patience = batch, epochs, patience
        self.best, self.models, self.hparams = None, [], {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, panel: Panel, train: slice, val: slice, tune: bool = True):
        gp = GPUPanel(panel, self.device)
        self._panel_nsic = panel.n_sic
        t0 = time.time()
        if tune or self.best is None:
            scores = {}
            for lr, l1 in self.grid:                       # one network per grid point
                _, v, ep = train_net(panel, gp, train, val, self.hidden, lr, l1, self.seed,
                                     self.batch, self.epochs, self.patience)
                scores[(lr, l1)] = v
                print(f"    {self.name} lr={lr:g} l1={l1:g}: val MSE {v:.6f} ({ep} epochs)")
            self.best = min(scores, key=scores.get)
        lr, l1 = self.best
        self.models, vals, eps = [], [], []
        for k in range(self.n_ensemble):
            m, v, ep = train_net(panel, gp, train, val, self.hidden, lr, l1, self.seed + 1000 + k,
                                 self.batch, self.epochs, self.patience)
            self.models.append(m); vals.append(v); eps.append(ep)
        self.hparams = {"lr": lr, "l1": l1, "val_mse_members": vals, "epochs": eps,
                        "minutes": (time.time() - t0) / 60}
        return self

    @torch.no_grad()
    def predict_raw(self, C, Mx, S, bs=200_000):
        out = np.zeros(len(C))
        for a in range(0, len(C), bs):
            c = torch.as_tensor(C[a:a + bs], device=self.device)
            mx = torch.as_tensor(Mx[a:a + bs], device=self.device)
            s = torch.as_tensor(S[a:a + bs], device=self.device, dtype=torch.long)
            x = build_torch(c, mx, s, self._panel_nsic)
            out[a:a + bs] = np.mean([m(x).cpu().numpy() for m in self.models], axis=0)
        return out

    def predict_rows(self, panel: Panel, sl: slice):
        return self.predict_raw(*panel.raw(sl))
