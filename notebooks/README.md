# Gulfstream tutorial notebooks

Walkthroughs using the public `gulfstream` API on real DuckDB tables.

| Notebook | Data |
|----------|------|
| [`01_ycs_zero_rates_workflow.ipynb`](01_ycs_zero_rates_workflow.ipynb) | `zero_rates` (+ FX) in `D:/data/duckdb/ycs_data.duckdb` |
| [`02_equity_eod_workflow.ipynb`](02_equity_eod_workflow.ipynb) | `equity_eod` in `D:/data/duckdb/equity_eod_data.duckdb` |

Each notebook:

- **Parts A–C** — PCA / kPCA / DMD with Graph 1 (`run_single_segmentation`) + Graph 2 (`refine_regimes`)
- **Parts D–J** (equity; YCS focuses on A–C + K) — Search, tests, ESS window, classical detectors, TFT, curve/ICA dimred
- **Part K** — Product: uncertainty bands + CI ribbon overlays, Excel export (`export.excel`), NDJSON event stream (`events`), streaming Graph 1, panel joint breakpoints
- **Comparison** — covering, **adjusted Rand index** (equity), and breakpoint F1 vs the PCA baseline

```bash
uv sync
uv run jupyter lab notebooks
```

Or open the `.ipynb` files in Cursor / VS Code with the project `.venv`.

Artifacts land under `outputs/notebooks/{ycs,equity}/…` (Part K under `…/product/`). Close DBeaver before the equity notebook if the DuckDB file is locked (falls back to `equity_eod_data_copy.duckdb` when present).

YCS source settings live in `config/sources/notebook_ycs.yaml`. Product YAML examples: `config/graph1/graph1_export_events.yaml`, `graph1_streaming_expanding.yaml`, `graph1_panel_joint.yaml`, `graph1_uncertainty_bands.yaml`.
