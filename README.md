# Gulfstream

Gulfstream implements pipelines for detecting structural breaks between time series regimes. We use the example of FX and yield-curve data, but any time series dataset will work. The core data structure must be (one or more) **polars** `DataFrame` with a required `date` column (`gulfstream.common.frames`). 

Three pipeline modes are available. 

* **Graph 1** runs full regime detection end to end: features → PCA/kPCA → RFF → ruptures → MMD → postprocess, then optional plots, insights, explainability, robustness, and stability. 
* **Graph 2** wraps that core in a targeted retrain loop: build an L2 heatmap, pick the worst regime and features, detect on the slice, merge breakpoints, and repeat. 
* **Legacy** exposes classical detectors (k-means, HDBSCAN, OPTICS, HMM, Bayesian GMM, MSAR, ruptures, Wasserstein) with optional DMD, t-SNE, or UMAP dimred.

Single-pass segmentation — shared by Graph 2 and the robustness/stability stages — and CLI feature loading are implemented as [Apache Hamilton](https://hamilton.apache.org/) dataflows under `gulfstream.pipelines.hamilton`.

## Setup

```bash
uv sync
uv pip install -e .
```

## Run

The CLI separates *what* you analyze from *how* you analyze it. A **source YAML** (`--source-config`) defines the input data; an **algo YAML** (`--config`) defines detection settings and post-run metrics. Use `--mode` to pick `graph1`, `graph2`, or `legacy`. If you omit `--mode`, Graph 2 is selected automatically when the config contains a `retrain` section.

```bash
# Core path (deferred stages off), synthetic jump data
uv run python -m gulfstream.cli --config config/graph1/default_core.yaml --source-config config/sources/synthetic.yaml

# Full Graph 1 with different sources
uv run python -m gulfstream.cli --mode graph1 --config config/graph1/full_graph1.yaml --source-config config/sources/synthetic.yaml
uv run python -m gulfstream.cli --config config/graph1/full_graph1.yaml --source-config config/sources/parquet.yaml
uv run python -m gulfstream.cli --config config/graph1/full_graph1.yaml --source-config config/sources/duckdb_smoke.yaml
uv run python -m gulfstream.cli --config config/graph1/full_graph1.yaml --source-config config/sources/csv.yaml

# Graph 2 auto retrain (empty seed → heatmaps → merge until threshold / max_iter)
uv run python -m gulfstream.cli --config config/graph2/full_graph2.yaml --source-config config/sources/synthetic.yaml
uv run python -m gulfstream.cli --mode graph2 --config config/graph2/full_graph2.yaml --source-config config/sources/duckdb_smoke.yaml

# Graph 2 interactive (stdin prompts for regime / features)
uv run python -m gulfstream.cli --config config/graph2/graph2_interactive.yaml --source-config config/sources/synthetic.yaml

# Legacy detectors (k-means / HDBSCAN / OPTICS / HMM smoke configs)
uv run python -m gulfstream.cli --mode legacy --config config/legacy/legacy_kmeans.yaml --source-config config/sources/synthetic.yaml
uv run python -m gulfstream.cli --mode legacy --config config/legacy/legacy_hdbscan.yaml --source-config config/sources/synthetic.yaml
uv run python -m gulfstream.cli --mode legacy --config config/legacy/legacy_optics.yaml --source-config config/sources/synthetic.yaml
uv run python -m gulfstream.cli --mode legacy --config config/legacy/legacy_hmm.yaml --source-config config/sources/synthetic.yaml
```

Dimred options for Graph 1, Graph 2, and legacy configs:

| Method | Embedding emitted |
|--------|-------------------|
| `pca` / `kpca` / `raw` / `dmd` / `tsne`  | classical / manifold embeddings |
| `bayesian_gmm` | mixture responsibilities |
| `hmm` | posterior state probabilities |
| `kmeans` | distances to cluster centers |
| `hdbscan` | distances to HDBSCAN centroids (discovered ``k``) |
| `optics` | distances to OPTICS centroids (discovered ``k``) |
| `msar` | smoothed MSAR regime probabilities |
| `wasserstein` | Wasserstein distances to cluster centroids (window→series) |
| `ruptures` | distances to PELT/window/Dynp segment means |
| `tft` | TFT attention-vector embeddings (`rank` = hidden size) |

TFT needs `lightning` and `pytorch-forecasting` (both installed); Model-based dimred methods read `algo.regimes` where relevant — HDBSCAN, OPTICS, Wasserstein, and ruptures-pelt can omit it; TFT uses `rank`. Post-information regime clustering is controlled by `metrics.regime_cluster_algorithms` (`kmeans`, `hdbscan`, or `optics`; default `kmeans`).

```bash
# Graph 1 with model-based dimred
uv run python -m gulfstream.cli --mode graph1 --config config/graph1/graph1_hmm_dimred.yaml --source-config config/sources/synthetic.yaml
uv run python -m gulfstream.cli --mode graph1 --config config/graph1/graph1_kmeans_dimred.yaml --source-config config/sources/synthetic.yaml
uv run python -m gulfstream.cli --mode graph1 --config config/graph1/graph1_hdbscan_dimred.yaml --source-config config/sources/synthetic.yaml
uv run python -m gulfstream.cli --mode graph1 --config config/graph1/graph1_optics_dimred.yaml --source-config config/sources/synthetic.yaml
uv run python -m gulfstream.cli --mode graph1 --config config/graph1/graph1_ruptures_dimred.yaml --source-config config/sources/synthetic.yaml
uv run python -m gulfstream.cli --mode graph1 --config config/graph1/graph1_tft_dimred.yaml --source-config config/sources/synthetic.yaml
```

### Source types

Source configs under `config/sources/` map to the following loaders:

| `type` | Key parameters |
|--------|----------------|
| `synthetic` | `case`, `regimes`, `features`, `durations`, `regime_params`, `seed` |
| `faker` | `n_days`, `n_regimes`, `sources`, `tenors`, `fx_pairs`, `seed` |
| `parquet` | `path`, optional `create_if_missing`, `generate_features` |
| `csv` | `path`, `date_column`, optional `columns`, `sep`, `start_date`/`end_date` |
| `duckdb` | `db_path`, `rate_table`, `sources`, `tenors`, `fx_pairs`, dates |
| `sqlite` | same shape as `duckdb` |

### Tutorial notebooks

For a guided walkthrough on real DuckDB data, see [`notebooks/`](notebooks/). `01_ycs_zero_rates_workflow.ipynb` covers `zero_rates` and FX from `D:/data/duckdb/ycs_data.duckdb`; `02_equity_eod_workflow.ipynb` covers `equity_eod` from `D:/data/duckdb/equity_eod_data.duckdb`. Both run Graph 1 and Graph 2 with PCA, kernel PCA, and DMD. Launch instructions are in [`notebooks/README.md`](notebooks/README.md).

Run outputs land under `outputs/metrics/` and `outputs/logs/`.

---

## Workflow

A single CLI entry loads data once, then dispatches to Graph 1, Graph 2, or legacy mode.

### Entry call path

```mermaid
flowchart TD
  cli["cli.main"] --> loadCfg["read_config_yaml --config"]
  cli --> loadSrc["load_features_from_source_yaml --source-config"]
  loadCfg --> mode{"--mode / retrain present?"}
  loadSrc --> mode
  mode -->|graph1| g1["evaluate_regimes_with_user_specified_df"]
  mode -->|graph2| g2["targeted_retrain_with_user_specified_df"]
  mode -->|legacy| leg["legacy_evaluate_regimes_with_user_specified_df"]
  g1 --> out1["outputs/metrics + gallery"]
  g2 --> out2["outputs/metrics + retrain heatmaps + gallery"]
  leg --> out3["outputs/metrics + regime plot"]
```

The split between `--source-config` and `--config` is deliberate: one file chooses the series, the other chooses the detector and which post-run metrics to emit. You can point the same algo YAML at synthetic smoke data or a DuckDB window without touching code.

---

### Graph 1 — full regime detection

Graph 1 searches for breakpoints over the full series in one parameter-grid pass, then optionally scores and explains the result.

```mermaid
flowchart TB
  subgraph stageA ["Stage A — Core detection"]
    df["Feature DataFrame"] --> dimred["PCA / kPCA / raw"]
    dimred --> rff["Kernel feature map RFF"]
    rff --> rupt["Ruptures candidates"]
    rupt --> mmd["MMD statistical tests"]
    mmd --> post["Post-process majority vote / min length"]
    post --> res["CustomAlgoResults"]
  end

  subgraph stageB ["Stage B — Metrics and diagnostics"]
    res --> viz["Regime plots"]
    res --> insights["L2 heatmaps, distances, clustering"]
    res --> trees["Breakpoint hierarchy trees"]
    res --> exp["Explainability trees"]
    res --> trans["Transition matrices"]
    res --> robust["Robustness / persistence"]
    res --> stab["Stability ablations"]
  end
```

| Stage | What it does | Why it exists |
|-------|----------------|---------------|
| **A — Core detection** | Reduce dimensions, map to a kernel feature space, propose breakpoints with ruptures, accept/reject with MMD, then clean segments. | Produce a single segmentation (`bkpts`, labels, hierarchy, stats) that is the baseline for everything else. |
| **B — Metrics** | Plot regimes, feature×regime L2 loss, regime distances/clusters, hierarchy trees, decision-tree explanations, transition probabilities; optionally perturb HPs (robustness) or drop sources/tenors/windows (stability). | Turn raw breakpoints into interpretable, stress-tested outputs under `outputs/metrics/`. |

Stage B is controlled by `metrics.plot` and by `robustness.enabled` / `stability.enabled` in YAML. `config/full_graph1.yaml` turns the deferred stages on; `default_core.yaml` keeps them off for faster smoke runs.

---

### Graph 2 — targeted retrain loop

Graph 2 does not define a separate detector. It wraps Graph 1's single-run path (`run_single_segmentation`) in an outer loop that focuses compute on poorly explained regimes.

```mermaid
flowchart LR
  seed["Seed regimes_df may be empty"] --> heat["L2 feature x regime heatmap"]
  heat --> pick["Select regime + features"]
  pick --> slice["run_single_segmentation on slice"]
  slice --> merge["Merge bkpts into hierarchy"]
  merge --> heat
```

In **auto mode** (`retrain.interactive: false`), the loop continues while the maximum L2 loss exceeds `threshold` and `iters < max_iter`. Each iteration picks the worst regime and its top-`num_worst_features`, reruns detection on that slice, merges the result, and refreshes the heatmap. The loop stops early if a slice yields no new breakpoints.

In **interactive mode** (`retrain.interactive: true`), the same steps run, but you choose the regime and features from stdin. Type `q`, `quit`, `exit`, or `stop` to finish.

| Step | Purpose |
|------|---------|
| Seed | Start from nothing or from a prior Graph 1 export (`regimes_df`). |
| Heatmap | Show which features are farthest from their regime mean (where the current partition fails). |
| Select | Decide *where* and *on which columns* to spend another detection pass. |
| Slice detect | Reuse Graph 1 core on `df.iloc[start:end][features]` only. |
| Merge | Attach the slice hierarchy with local indices, shift bkpts/stats to global time, then loop. |

Final results are built from the **merged** `processed_bkpts`, not the seed list. The driver then writes optional Excel/report output, runs `_produce_all_metrics`, and generates an HTML gallery.

---

### How the two graphs relate

```mermaid
flowchart LR
  g1core["Graph 1 core path"] --> single["single_run.run_single_segmentation"]
  single --> g2loop["Graph 2 outer loop"]
  g1full["Graph 1 full driver"] --> metrics["_produce_all_metrics"]
  g2loop --> metrics
```

Graph 1 is the global search. Graph 2 is local refinement: it repeatedly calls the same single-segmentation primitive on troubled intervals until the L2 heatmap is good enough or the iteration budget is spent.

---

## Architecture

```text
gulfstream/
├── config/
│   ├── graph1/             # default_core, full_graph1, graph1_*_dimred
│   ├── graph2/             # full_graph2, graph2_interactive
│   ├── legacy/             # legacy_*.yaml
│   └── sources/            # synthetic, duckdb, parquet, csv, ...
├── src/gulfstream/
│   ├── cli.py              # Entry: load configs → dispatch mode
│   ├── common/             # frames, results, utils, logging, plotting (plotnine)
│   ├── data/               # source_loader, synth, feature_generation
│   ├── pipelines/          # graph1, graph2, legacy, single_pass, _shared
│   │   └── hamilton/       # Apache Hamilton DAGs (segmentation, load_features)
│   ├── detection/          # algorithm, stat_tests, time_index, trees, hyperparams, postprocess
│   ├── dimred/             # dispatcher, classical/*, model_based, density
│   ├── features/           # kernel_map, names
│   ├── metrics/            # evaluation, plots, insights, explainability, robustness, ...
│   ├── legacy/detectors/   # kmeans, hmm, hdbscan, optics, msar, ruptures, ...
│   └── validation/         # flattened YAML schema checks
└── outputs/
    ├── metrics/            # PNGs, Excel, gallery.html
    └── logs/
```

### Layers

| Layer | Responsibility |
|-------|----------------|
| **CLI** | Parse `--config`, `--source-config`, `--mode`; materialize `${IMG_DIR}` / `${LOG_DIR}`; call Graph 1, Graph 2, or legacy. |
| **Data** | Build a dated feature matrix from DuckDB/SQLite, parquet/CSV, or synthetic/faker generators. No detection logic. |
| **Pipelines** | Mode orchestrators (`graph1` / `graph2` / `legacy`) plus shared helpers and `single_pass`. The linear segmentation core and CLI feature load run as [Apache Hamilton](https://hamilton.apache.org/) DAGs (`pipelines/hamilton/`). |
| **Detection** | Custom ruptures + MMD + hyperparams + postprocess. |
| **Dimred / features** | Classical and model-based embeddings; RFF / Nyström maps. |
| **Metrics** | Evaluation primitives, heatmaps, trees, explainability, transitions, robustness, stability. |
| **Legacy detectors** | Classical labelers (also reused as model-based dimred). |
| **Validation** | Reject bad YAML knobs before a long run starts. |
| **Outputs** | Timestamped `bkpt_tests_*` dirs under `outputs/metrics/`, plus logs under `outputs/logs/`. |

### Key result type

Runs accumulate into `SegmentResults` (alias `CustomAlgoResults`): breakpoints, invalid breakpoints, per-bkpt stats, hierarchy, labels, and optional `persistence`, `low_confidence_bkpts`, and `stability_score` from the deferred Graph 1 stages.

### Config split

Algo configs (`config/graph1|graph2|legacy/*.yaml`) hold `algo`, `test`, `metrics`, and optional `robustness`, `stability`, `retrain`, and `log` sections. Source configs (`config/sources/*.yaml`) describe only how to load or generate the feature DataFrame. That separation lets you reuse the same detector settings across synthetic smoke data and production DuckDB extracts without code changes.
