"""Autoencoder latent embeddings as dimred (torch MLP)."""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from gulfstream.common import frames, utils
from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common

logger = logging.getLogger(__name__)


def _resolve_rank(df: pl.DataFrame, params: dict, *, default: int = 4) -> int:
    ranks = params.get("algo", {}).get("rank") or []
    n_feat = frames.n_features(df)
    if ranks:
        return max(1, min(int(max(ranks)), n_feat, max(df.height // 5, 1)))
    return max(1, min(default, n_feat, max(df.height // 5, 1)))


def _autoencoder_dimred(
    df: pl.DataFrame,
    *,
    rank: int,
    ae_hidden: int = 32,
    ae_epochs: int = 40,
    ae_lr: float = 1e-2,
    random_state: int | None = 42,
    **kwargs,
) -> DimredResults:
    import torch
    from torch import nn

    X = frames.to_numpy(df).astype(np.float32)
    n, d = X.shape
    k = max(1, min(int(rank), d, max(n // 5, 1)))
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd

    if random_state is not None:
        torch.manual_seed(int(random_state))
        np.random.seed(int(random_state))

    hidden = max(int(ae_hidden), k)
    encoder = nn.Sequential(
        nn.Linear(d, hidden),
        nn.ReLU(),
        nn.Linear(hidden, k),
    )
    decoder = nn.Sequential(
        nn.Linear(k, hidden),
        nn.ReLU(),
        nn.Linear(hidden, d),
    )
    model = nn.Sequential(encoder, decoder)
    opt = torch.optim.Adam(model.parameters(), lr=float(ae_lr))
    loss_fn = nn.MSELoss()
    xt = torch.from_numpy(Z)

    model.train()
    for _ in range(int(ae_epochs)):
        opt.zero_grad()
        recon = model(xt)
        loss = loss_fn(recon, xt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        latents = encoder(xt).cpu().numpy()

    return DimredResults(
        df=frames.with_same_dates(latents, df),
        dimred=DimredMethod.AUTOENCODER,
        rank=k,
        rank_selection_method="user_specified",
        model={"encoder": encoder, "decoder": decoder, "mu": mu, "sd": sd},
    )


def _autoencoder_generator(df: pl.DataFrame, params: dict):
    if not common._need_dimred(DimredMethod.AUTOENCODER, params):
        return
    ranks = params["algo"].get("rank") or [_resolve_rank(df, params)]
    hiddens = params["algo"].get("ae_hidden", [32])
    epochs = params["algo"].get("ae_epochs", [40])
    lrs = params["algo"].get("ae_lr", [1e-2])
    for rank in ranks:
        for hidden in hiddens:
            for ep in epochs:
                for lr in lrs:
                    for rs in utils.algo_grid(params, "random_state", [42]):
                        try:
                            yield _autoencoder_dimred(
                                df,
                                rank=int(rank),
                                ae_hidden=int(hidden),
                                ae_epochs=int(ep),
                                ae_lr=float(lr),
                                random_state=rs,
                            )
                        except Exception as exc:
                            logger.warning("Autoencoder dimred skipped: %s", exc)
                            return
