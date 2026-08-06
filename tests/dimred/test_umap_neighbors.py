"""UMAP n_neighbors coercion for Graph 2 regime slices."""
import numpy as np
import polars as pl

from gulfstream.dimred.classical.umap import _umap_dimred


def test_umap_neighbors_coerced_to_int_on_short_slice():
    n = 12
    df = pl.DataFrame(
        {
            "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 12), eager=True),
            **{f"f{i}": np.random.rand(n) for i in range(4)},
        }
    )
    result = _umap_dimred(
        df,
        rank_selection_method="user_specified",
        rank=2,
        umap_num_neighbors=15.0,
        umap_min_dist=0.1,
        umap_metric="euclidean",
        random_state=42,
    )
    assert result.rank == 2
    assert result.umap_num_neighbors == n - 1
