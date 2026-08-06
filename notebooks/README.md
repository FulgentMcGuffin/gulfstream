# Gulfstream tutorial notebooks

Walkthroughs using the public `gulfstream` API.

| Notebook | Data |
|----------|------|
| [`01_ycs_zero_rates_workflow.ipynb`](01_ycs_zero_rates_workflow.ipynb) | User supplied yield curve and FX data |
| [`02_equity_eod_workflow.ipynb`](02_equity_eod_workflow.ipynb) | User supplied equity data |
| [`03_faker_hmm_workflow.ipynb`](03_faker_hmm_workflow.ipynb) | Faker HMM panel (~10y daily, 10 features, 4 regimes with 2 repeating; each regime = 2-state MVN HMM) |
| [`04_parquet_hmm_workflow.ipynb`](04_parquet_hmm_workflow.ipynb) | Same HMM panel via parquet (`data/synthetic/hmm_panel.parquet`) |
| [`05_ycs_panelyzer_workflow.ipynb`](05_ycs_panelyzer_workflow.ipynb) | Same YCS DuckDB window; features via panelyzer (Parts A–B only) |

Each notebook:

- **Parts A–E** — PCA / kPCA / DMD / t-SNE / UMAP with Graph 1 (`run_single_segmentation`) + Graph 2 (`refine_regimes`)
- **Parts F–L** — Search, tests, ESS window, classical detectors, TFT, curve/ICA dimred
- **Part K** — Product: uncertainty bands + CI ribbon overlays, Excel export (`export.excel`), NDJSON event stream (`events`), streaming Graph 1, panel joint breakpoints
- **Part L** — Graph 2 retrain scores (`retrain.score_method`): compare pick tables, then run Graph 2 with `mse_on_diff` / `factor_residual` / `energy_split` / `mmd_split`
- **Comparison** — covering, **adjusted Rand index**, and breakpoint F1 (notebooks 03/04 score against **ground-truth** outer breakpoints)

```bash
uv sync
uv run jupyter lab notebooks
```

Or open the `.ipynb` files in Cursor / VS Code with the project `.venv`.

### Plotnine return helpers

Each workflow notebook defines `run_g1` / `run_g2` helpers that optionally return plotnine `ggplot` objects for notebook inspection:

- **Graph 1** — `proc, fig = run_g1(..., return_fig=True)` returns the regime shading plot as `fig`.
- **Graph 2** — `out_dir, refined, figs = run_g2(..., return_figs=True)` returns a dict `figs` with keys `retrain_iteration_0`, … (score heatmaps) and `regime` (final shading plot).
- **Standalone `plot_regimes(...)`** — Part K product cells assign the returned ggplot to a `fig_*` variable (`emit=False`) and display it explicitly.

Default calls (no flags) keep prior auto-display behavior.

**TFT (Part I):** All notebooks call TFT through `run_g1` / `run_single_segmentation`, which uses `gulfstream.detectors.tft`. Feature columns containing `.` (common on equity tickers) are renamed automatically before training — no notebook-side sanitization is required.

Artifacts land under `outputs/notebooks/{ycs,equity,faker_hmm,parquet_hmm,ycs_panelyzer}/…` (Part K under `…/product/`, Part L under `…/graph2_scores/`). Close any client that has a DuckDB file open if a notebook cannot acquire a read lock.

Source YAMLs:

- YCS: `config/sources/notebook_ycs.yaml`
- YCS + panelyzer: `config/sources/notebook_ycs_panelyzer.yaml` → `config/features/ycs_panelyzer_subset.yaml`
- Faker HMM: `config/sources/notebook_faker_hmm.yaml`
- Parquet HMM: `config/sources/notebook_parquet_hmm.yaml` (`create_if_missing: true`)

- Product YAML: `config/graph1/graph1_export_events.yaml`, `graph1_streaming_expanding.yaml`, `graph1_panel_joint.yaml`, `graph1_uncertainty_bands.yaml`
- Graph 2 score YAML: `config/graph2/full_graph2.yaml`, `graph2_score_diff.yaml`, `graph2_score_factor.yaml`, `graph2_score_energy.yaml`, `graph2_score_mmd.yaml`
