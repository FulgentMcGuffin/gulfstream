"""Temporal Fusion Transformer helpers (optional heavy deps).

pytorch_forecasting / lightning / mlflow are imported lazily so the rest of
the detectors package can load without them. Call any function here only when
those packages are installed.

Feature prep is done in polars; conversion to pandas happens only at the
``TimeSeriesDataSet`` boundary (pytorch_forecasting requires ``pd.DataFrame``).
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

import polars as pl

from gulfstream.common import frames

logger = logging.getLogger(__name__)

_TFT_RESERVED = frozenset({"time_idx", frames.DATE_COL})


def _sanitize_tft_column_name(name: str, *, used: set[str]) -> str:
    """Make a gulfstream feature name safe for pytorch-forecasting.

    ``TimeSeriesDataSet`` rejects column / group ids containing ``'.'``.
    """
    safe = str(name).replace(".", "_")
    base = safe
    n = 2
    while safe in used or safe in _TFT_RESERVED:
        safe = f"{base}_{n}"
        n += 1
    used.add(safe)
    return safe


def _sanitize_feature_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename feature columns so TFT accepts them."""
    used: set[str] = set(_TFT_RESERVED)
    renames: dict[str, str] = {}
    for col in frames.feature_columns(df):
        safe = _sanitize_tft_column_name(col, used=used)
        if safe != col:
            renames[col] = safe
    return df.rename(renames) if renames else df


def sanitize_feature_columns_for_tft(df: pl.DataFrame) -> pl.DataFrame:
    """Public alias: rename feature columns for pytorch-forecasting compatibility."""
    return _sanitize_feature_columns(df)


def _require_tft_stack():
    """Import TFT dependencies or raise a clear error.

    ``mlflow`` is optional (often conflicts with newer ``pyarrow``); dimred
    training works without it.
    """
    try:
        import torch
        import lightning.pytorch as lightning_pl
        from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
        from lightning.pytorch.loggers import TensorBoardLogger
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TFT helpers require torch, lightning, and pytorch-forecasting. "
            "Install with `uv add lightning pytorch-forecasting`. "
            f"Original error: {exc}"
        ) from exc

    out = {
        "torch": torch,
        "lightning": lightning_pl,
        "EarlyStopping": EarlyStopping,
        "LearningRateMonitor": LearningRateMonitor,
        "TensorBoardLogger": TensorBoardLogger,
        "TimeSeriesDataSet": TimeSeriesDataSet,
        "TemporalFusionTransformer": TemporalFusionTransformer,
        "GroupNormalizer": GroupNormalizer,
        "QuantileLoss": QuantileLoss,
        "MLFlowLogger": None,
        "mlflow_pytorch": None,
    }
    try:
        from lightning.pytorch.loggers import MLFlowLogger
        import mlflow.pytorch

        out["MLFlowLogger"] = MLFlowLogger
        out["mlflow_pytorch"] = mlflow.pytorch
    except ImportError:
        pass
    return out


def _feature_wide(df: pl.DataFrame) -> pl.DataFrame:
    """Dated gulfstream frame → wide numeric features + ``time_idx``."""
    df = frames.ensure_date_column(df)
    df = _sanitize_feature_columns(df)
    feat_cols = frames.feature_columns(df)
    if not feat_cols:
        raise ValueError("TFT requires at least one feature column")
    return df.select(feat_cols).with_columns(
        pl.int_range(0, pl.len()).alias("time_idx")
    )


def _to_tft_pandas(df: pl.DataFrame):
    """Last-mile conversion for pytorch_forecasting (requires pandas)."""
    return df.to_pandas()


def process_data_for_tft_univariate(
    df: pl.DataFrame,
    training_cutoff: float = 0.8,
    min_encoder_length: int = 90,
    max_encoder_length: int = 90,
    min_prediction_length: int = 5,
    max_prediction_length: int = 5,
) -> Tuple[Any, Any, dict]:
    deps = _require_tft_stack()
    TimeSeriesDataSet = deps["TimeSeriesDataSet"]
    GroupNormalizer = deps["GroupNormalizer"]

    wide = _feature_wide(df)
    feat_cols = [c for c in wide.columns if c != "time_idx"]
    long = wide.unpivot(
        index=["time_idx"],
        on=feat_cols,
        variable_name="Series",
        value_name="Value",
    ).with_columns(pl.col("Value").cast(pl.Float64))
    long = long.with_columns(
        (pl.col("Value") - pl.col("Value").min().over("Series") + 1e-5).alias("Value")
    )

    series_order = long["Series"].unique(maintain_order=True).to_list()
    cat_mapping = {category: code for code, category in enumerate(series_order)}
    training_cutoff_idx = int(long["time_idx"].max() * training_cutoff)

    train_pdf = _to_tft_pandas(long.filter(pl.col("time_idx") <= training_cutoff_idx))
    val_pdf = _to_tft_pandas(long.filter(pl.col("time_idx") > training_cutoff_idx))
    train_pdf["Series"] = train_pdf["Series"].astype("category")
    val_pdf["Series"] = val_pdf["Series"].astype("category")

    training_set = TimeSeriesDataSet(
        train_pdf,
        time_idx="time_idx",
        target="Value",
        group_ids=["Series"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Series"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["Value"],
        target_normalizer=GroupNormalizer(groups=["Series"], transformation="softplus"),
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation_set = TimeSeriesDataSet.from_dataset(
        training_set,
        val_pdf,
        predict=False,
        stop_randomization=True,
    )
    return training_set, validation_set, cat_mapping


def process_data_for_tft_multivariate(
    df: pl.DataFrame,
    training_cutoff: float = 0.8,
    min_encoder_length: int = 90,
    max_encoder_length: int = 90,
    min_prediction_length: int = 5,
    max_prediction_length: int = 5,
    target_col: str | None = None,
) -> Tuple[Any, Any]:
    deps = _require_tft_stack()
    TimeSeriesDataSet = deps["TimeSeriesDataSet"]
    GroupNormalizer = deps["GroupNormalizer"]

    wide = _feature_wide(df)
    feat_cols = [c for c in wide.columns if c != "time_idx"]
    shift_exprs = []
    for c in feat_cols:
        min_val = wide[c].min()
        if min_val is not None and min_val < 0:
            shift_exprs.append((pl.col(c) - pl.col(c).min() + 1e-5).alias(c))
    if shift_exprs:
        wide = wide.with_columns(shift_exprs)
    wide = wide.with_columns(pl.lit("0").alias("Series"))

    if target_col is None:
        target_col = feat_cols[0]
    elif target_col not in feat_cols:
        raise KeyError(f"target_col {target_col!r} not in features {feat_cols}")

    training_cutoff_idx = int(wide["time_idx"].max() * training_cutoff)
    train_pdf = _to_tft_pandas(wide.filter(pl.col("time_idx") <= training_cutoff_idx))
    val_pdf = _to_tft_pandas(wide.filter(pl.col("time_idx") > training_cutoff_idx))

    unknown_reals = [
        c for c in train_pdf.columns if c not in ("time_idx", target_col, "Series")
    ]
    training_set = TimeSeriesDataSet(
        train_pdf,
        time_idx="time_idx",
        target=target_col,
        group_ids=["Series"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Series"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=["Series"], transformation="softplus"),
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation_set = TimeSeriesDataSet.from_dataset(
        training_set,
        val_pdf,
        predict=False,
        stop_randomization=True,
    )
    return training_set, validation_set


def train_tft(training_set, validation_set, rank: int, **kwargs):
    deps = _require_tft_stack()
    TemporalFusionTransformer = deps["TemporalFusionTransformer"]
    QuantileLoss = deps["QuantileLoss"]
    EarlyStopping = deps["EarlyStopping"]
    LearningRateMonitor = deps["LearningRateMonitor"]
    TensorBoardLogger = deps["TensorBoardLogger"]
    MLFlowLogger = deps["MLFlowLogger"]
    lightning_pl = deps["lightning"]
    mlflow_pytorch = deps["mlflow_pytorch"]

    tft = TemporalFusionTransformer.from_dataset(
        training_set,
        learning_rate=kwargs.get("learning_rate", 0.001),
        hidden_size=rank,
        attention_head_size=kwargs.get("attention_head_size", 4),
        dropout=kwargs.get("dropout", 0.1),
        hidden_continuous_size=kwargs.get("hidden_continuous_size", 8),
        loss=kwargs.get("loss", QuantileLoss()),
        log_interval=kwargs.get("log_interval", 10),
        optimizer=kwargs.get("optimizer", "adam"),
        reduce_on_plateau_patience=kwargs.get("reduce_on_plateau_patience", 4),
    )
    logger.info("Number of parameters in network: %.1fk.", tft.size() / 1e3)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    callbacks = [early_stop_callback]
    loggers = []
    if kwargs.get("enable_tensorboard", False):
        loggers.append(TensorBoardLogger("lightning_logs"))
    if MLFlowLogger is not None and kwargs.get("enable_mlflow", False):
        loggers.append(MLFlowLogger(experiment_name="gulfstream_tft"))
    if loggers:
        callbacks.insert(0, LearningRateMonitor())

    trainer = lightning_pl.Trainer(
        max_epochs=kwargs.get("max_epochs", 100),
        accelerator=kwargs.get("accelerator", "cpu"),
        enable_model_summary=kwargs.get("enable_model_summary", True),
        gradient_clip_val=kwargs.get("gradient_clip_val", 0.1),
        limit_train_batches=kwargs.get("limit_train_batches", 1.0),
        fast_dev_run=kwargs.get("fast_dev_run", False),
        callbacks=callbacks,
        logger=loggers or False,
        enable_checkpointing=kwargs.get("enable_checkpointing", False),
    )
    train_dataloader = training_set.to_dataloader(
        train=True, batch_size=kwargs.get("batch_size", 32)
    )
    val_dataloader = validation_set.to_dataloader(
        train=False, batch_size=kwargs.get("batch_size", 32)
    )
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    if mlflow_pytorch is not None and kwargs.get("enable_mlflow", False) and loggers:
        try:
            run_id = getattr(loggers[-1], "run_id", None)
            if run_id:
                mlflow_pytorch.log_model(tft, artifact_path="model", run_id=run_id)
        except Exception:
            logger.exception("Failed to log TFT model to mlflow; continuing.")
    return tft


def get_attention_vectors(tft, train_dataloader) -> dict:
    deps = _require_tft_stack()
    torch = deps["torch"]
    attn: dict = {}
    with torch.no_grad():
        for batch in train_dataloader:
            x, _y = batch
            ts = (x["decoder_time_idx"][:, 0] - 1).tolist()
            cats = (x["decoder_cat"][:, 0].reshape(-1)).tolist()
            output = tft(x)
            attn_weights = torch.cat(
                [output["encoder_attention"], output["decoder_attention"]], dim=-1
            )
            for i in range(attn_weights.shape[0]):
                if cats[i] not in attn:
                    attn[cats[i]] = {}
                mean_attn_weights = attn_weights[i, :, :].mean(dim=1)
                attn[cats[i]][ts[i]] = mean_attn_weights.view(-1).cpu().numpy()
    return attn
