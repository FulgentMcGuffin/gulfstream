# Gulfstream tutorial notebooks

Walkthroughs using the public `gulfstream` API on real DuckDB tables.

| Notebook | Data |
|----------|------|
| [`01_ycs_zero_rates_workflow.ipynb`](01_ycs_zero_rates_workflow.ipynb) | `zero_rates` (+ FX) in `D:/data/duckdb/ycs_data.duckdb` |
| [`02_equity_eod_workflow.ipynb`](02_equity_eod_workflow.ipynb) | `equity_eod` in `D:/data/duckdb/equity_eod_data.duckdb` |

Each notebook:

- **Parts A–C** — PCA / kPCA / DMD with Graph 1 (`run_single_segmentation`) + Graph 2 (`refine_regimes`)
- **Part D** — Binseg / BottomUp search
- **Part E** — `energy_distance` / `mmd_unbiased` tests
- **Part F** — ESS window hyperparameter
- **Comparison** — covering + breakpoint F1 vs the PCA baseline

```bash
uv sync
uv run jupyter lab notebooks
```

Or open the `.ipynb` files in Cursor / VS Code with the project `.venv`.

Artifacts land under `outputs/notebooks/{ycs,equity}/…`. Close DBeaver before the equity notebook if the DuckDB file is locked (falls back to `equity_eod_data_copy.duckdb` when present).

YCS source settings live in `config/sources/notebook_ycs.yaml`.
