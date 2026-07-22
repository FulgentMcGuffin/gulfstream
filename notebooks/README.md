# Gulfstream tutorial notebooks

These notebooks walk through Graph 1 and Graph 2 on real DuckDB data, using PCA, kernel PCA, and DMD embeddings. They are meant to be run top to bottom in order — each section builds on variables and results from the previous one.

| Notebook | Data |
|----------|------|
| [`01_ycs_zero_rates_workflow.ipynb`](01_ycs_zero_rates_workflow.ipynb) | `zero_rates` (+ FX) in `D:/data/duckdb/ycs_data.duckdb` |
| [`02_equity_eod_workflow.ipynb`](02_equity_eod_workflow.ipynb) | `equity_eod` in `D:/data/duckdb/equity_eod_data.duckdb` |

Each notebook loads and engineers features from its table, then runs three passes over the same data. Part A uses PCA: Graph 1 finds an initial segmentation, and Graph 2 auto-retrain refines it using that result as a seed. Parts B and C repeat the same Graph 1 → Graph 2 sequence with kernel PCA and DMD. A final comparison table shows how many breakpoints each embedding kept.

Graph 2 seeding is the important detail. Rather than starting from an empty partition, the notebook converts the Graph 1 `SegmentResults` into a `regimes_df` and passes it to `retrain.regimes_df`. The retrain loop then builds L2 heatmaps, picks the worst regime and features, reruns detection on that slice, and merges new breakpoints until the loss threshold or iteration limit is reached.

## How to run

From the repo root:

```bash
uv sync
uv run jupyter lab notebooks
```

You can also open the `.ipynb` files directly in Cursor or VS Code and select the project `.venv` as the kernel.

Plots and other artifacts are written under `outputs/notebooks/{ycs,equity}/{pca,kpca,dmd}/`. Graph 2 runs additionally produce retrain heatmaps and an HTML gallery under each `graph2/` subdirectory.

## Notes

The equity database is often locked if DBeaver or another DuckDB client has it open. Close those connections before running `02_equity_eod_workflow.ipynb`; the notebook falls back to `equity_eod_data_copy.duckdb` when that file exists.

The YCS notebook reads its source settings from `config/sources/notebook_ycs.yaml`. Adjust date ranges, sources, or tenors there rather than editing the notebook cells.
