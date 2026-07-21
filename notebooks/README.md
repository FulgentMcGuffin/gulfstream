# Gulfstream tutorial notebooks

End-to-end walkthroughs of the Graph 1 core path on real DuckDB tables.

| Notebook | Data |
|----------|------|
| [`01_ycs_zero_rates_workflow.ipynb`](01_ycs_zero_rates_workflow.ipynb) | `zero_rates` (+ FX) in `D:/data/duckdb/ycs_data.duckdb` |
| [`02_equity_eod_workflow.ipynb`](02_equity_eod_workflow.ipynb) | `equity_eod` in `D:/data/duckdb/equity_eod_data.duckdb` |

## How to run

From the repo root (recommended):

```bash
uv sync
uv run jupyter lab notebooks
```

Or open the `.ipynb` files in Cursor / VS Code and select the project `.venv`.

Artifacts from the notebooks land under `outputs/notebooks/`.

## Notes

- Close DBeaver (or any exclusive DuckDB client) before the equity notebook if you hit a file-lock error.
- Source YAML for the YCS notebook: `config/sources/notebook_ycs.yaml`.
