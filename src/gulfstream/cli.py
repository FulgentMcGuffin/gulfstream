"""CLI entry for Graph 1 / Graph 2 regime pipelines."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gulfstream.common import frames, utils
from gulfstream.pipelines import graph1, graph2
from gulfstream.pipelines.hamilton.driver import load_features

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gulfstream Graph 1 / Graph 2 regime pipeline")
    p.add_argument(
        "--config",
        type=Path,
        default=_project_root() / "config" / "graph1" / "default_core.yaml",
        help="Algo/test/metrics YAML config",
    )
    p.add_argument(
        "--source-config",
        type=Path,
        default=_project_root() / "config" / "sources" / "synthetic.yaml",
        help="Data source YAML (type + parameters)",
    )
    p.add_argument(
        "--mode",
        choices=["graph1", "graph2"],
        default=None,
        help="Pipeline mode. Default: graph2 if config has 'retrain', else graph1.",
    )
    p.add_argument(
        "--img-dir",
        type=Path,
        default=_project_root() / "outputs" / "metrics",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=_project_root() / "outputs" / "logs",
    )
    return p


def _resolve_mode(explicit: str | None, params: dict) -> str:
    if explicit is not None:
        return explicit
    return "graph2" if params.get("retrain") is not None else "graph1"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.img_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    params = utils.read_config_yaml(
        str(args.config),
        img_dir=str(args.img_dir.resolve()),
        log_dir=str(args.log_dir.resolve()),
    )
    if not params["metrics"].get("features_to_plot"):
        params["metrics"]["features_to_plot"] = ["__auto__"]
    if not params["metrics"].get("explainability_features"):
        params["metrics"]["explainability_features"] = ["__auto__"]

    mode = _resolve_mode(args.mode, params)
    if mode == "graph2" and not params.get("retrain"):
        logger.error("Graph 2 mode requires a 'retrain' section in --config.")
        return 1

    df, source_type = load_features(
        args.source_config,
        project_root=_project_root(),
    )
    feat_cols = frames.feature_columns(df)
    n_feat = len(feat_cols)
    if params["metrics"]["features_to_plot"] == ["__auto__"]:
        params["metrics"]["features_to_plot"] = feat_cols[: min(3, n_feat)]
    if params["metrics"]["explainability_features"] == ["__auto__"]:
        params["metrics"]["explainability_features"] = feat_cols[: min(5, n_feat)]

    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Running %s on DF shape=%s source=%s (%s)",
        mode,
        frames.shape_features(df),
        source_type,
        args.source_config,
    )
    if mode == "graph2":
        graph2.run_graph2(df, params)
    else:
        graph1.run_graph1(df, params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
