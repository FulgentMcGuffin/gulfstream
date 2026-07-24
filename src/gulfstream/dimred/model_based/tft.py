"""TFT model-based dimred (delegates to detectors.tft)."""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.model_based._common import _as_frame
from gulfstream.detectors import tft as tft_mod


def _attention_to_matrix(attn: dict, n_times: int) -> np.ndarray:
    """Average attention vectors across series categories onto a time×dim grid."""
    buckets: dict[int, list[np.ndarray]] = {}
    for _cat, by_t in attn.items():
        for t, vec in by_t.items():
            t_i = int(t)
            if 0 <= t_i < n_times:
                buckets.setdefault(t_i, []).append(np.asarray(vec, dtype=float).ravel())
    if not buckets:
        raise ValueError("TFT produced no attention vectors; check encoder/data length.")
    dim = next(iter(next(iter(buckets.values())))).shape[0]
    out = np.full((n_times, dim), np.nan, dtype=float)
    for t, vecs in buckets.items():
        out[t] = np.mean(np.vstack(vecs), axis=0)
    # Forward/back fill NaNs without pandas as primary frame type.
    for j in range(dim):
        col = out[:, j]
        last = np.nan
        for i in range(n_times):
            if not np.isnan(col[i]):
                last = col[i]
            elif not np.isnan(last):
                col[i] = last
        last = np.nan
        for i in range(n_times - 1, -1, -1):
            if not np.isnan(col[i]):
                last = col[i]
            elif not np.isnan(last):
                col[i] = last
        out[:, j] = col
    return out


def _tft_dimred(
    df: pl.DataFrame,
    rank: int,
    tft_encoder_length: int = 30,
    tft_prediction_length: int = 5,
    tft_max_epochs: int = 5,
    tft_training_cutoff: float = 0.8,
    tft_batch_size: int = 32,
    tft_mode: str = "univariate",
    random_state: int | None = 42,
    **kwargs,
) -> DimredResults:
    """Train a TFT and emit attention-vector embeddings (represent's dimred view)."""
    if rank is None or int(rank) < 1:
        raise ValueError("'rank' (hidden_size) must be a positive int for TFT dimred.")
    enc = int(tft_encoder_length)
    pred = int(tft_prediction_length)
    if df.height < enc + pred + 5:
        raise ValueError(
            f"TFT dimred needs len(df) >= encoder+prediction+5 "
            f"({enc}+{pred}+5); got {df.height}."
        )

    if tft_mode == "multivariate":
        training_set, validation_set = tft_mod.process_data_for_tft_multivariate(
            df,
            training_cutoff=tft_training_cutoff,
            min_encoder_length=enc,
            max_encoder_length=enc,
            min_prediction_length=pred,
            max_prediction_length=pred,
        )
    else:
        training_set, validation_set, _cats = tft_mod.process_data_for_tft_univariate(
            df,
            training_cutoff=tft_training_cutoff,
            min_encoder_length=enc,
            max_encoder_length=enc,
            min_prediction_length=pred,
            max_prediction_length=pred,
        )

    tft = tft_mod.train_tft(
        training_set,
        validation_set,
        rank=int(rank),
        max_epochs=int(tft_max_epochs),
        batch_size=int(tft_batch_size),
        enable_tensorboard=False,
        enable_mlflow=False,
        enable_model_summary=False,
        attention_head_size=kwargs.get("attention_head_size", 1),
        accelerator=kwargs.get("accelerator", "cpu"),
    )
    train_dl = training_set.to_dataloader(train=False, batch_size=int(tft_batch_size))
    val_dl = validation_set.to_dataloader(train=False, batch_size=int(tft_batch_size))
    attn = tft_mod.get_attention_vectors(tft, train_dl)
    attn_val = tft_mod.get_attention_vectors(tft, val_dl)
    for cat, by_t in attn_val.items():
        attn.setdefault(cat, {}).update(by_t)

    mat = _attention_to_matrix(attn, df.height)
    if mat.shape[1] != int(rank):
        n_comp = min(int(rank), mat.shape[0], mat.shape[1])
        pca = PCA(n_components=n_comp, random_state=random_state)
        projected = pca.fit_transform(mat)
        if projected.shape[1] < int(rank):
            pad = np.zeros((projected.shape[0], int(rank) - projected.shape[1]))
            projected = np.hstack([projected, pad])
        mat = projected
        model: dict = {"tft": tft, "pca": pca}
    else:
        model = {"tft": tft}

    return DimredResults(
        df=_as_frame(mat, df, "tft_"),
        dimred="tft",
        rank=mat.shape[1],
        rank_selection_method="user_specified",
        model=model,
    )


def _tft_generator(df: pl.DataFrame, params: dict):
    if DimredMethod.TFT not in params.get("algo", {}).get("dimred", []):
        return
    ranks = params["algo"].get("rank")
    if not ranks:
        raise ValueError("'rank' must be provided when using TFT dimred.")

    def _first(key, default):
        val = params["algo"].get(key, default)
        if isinstance(val, list):
            return val[0] if val else default
        return val

    for rank in ranks:
        for enc in params["algo"].get("tft_encoder_length", [30]):
            for pred in params["algo"].get("tft_prediction_length", [5]):
                for epochs in params["algo"].get("tft_max_epochs", [5]):
                    for mode in params["algo"].get("tft_mode", ["univariate"]):
                        yield _tft_dimred(
                            df,
                            rank=rank,
                            tft_encoder_length=enc,
                            tft_prediction_length=pred,
                            tft_max_epochs=epochs,
                            tft_training_cutoff=_first("tft_training_cutoff", 0.8),
                            tft_batch_size=_first("tft_batch_size", 32),
                            tft_mode=mode,
                        )
