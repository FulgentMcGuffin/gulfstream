"""Stdin prompts for interactive Graph 2 retrain."""
from __future__ import annotations

import numpy as np
import polars as pl

from gulfstream.common import frames


def _get_user_requested_regime(bkpts: list[int], suggested: int) -> str:
    quits = ["q", "quit", "exit", "stop"]
    regime = input(
        f"Enter a regime number to break. Valid regime numbers are between 0 and "
        f"{len(bkpts)} inclusive. Suggested: regime {suggested}. "
        "Or enter 'q' to quit. "
    )
    not_digit_or_quit = not regime.isdigit() and regime.lower() not in quits
    digit_out_of_range = regime.isdigit() and (int(regime) < 0 or int(regime) > len(bkpts))
    while not_digit_or_quit or digit_out_of_range:
        if not_digit_or_quit:
            regime = input(
                "Invalid input. Enter a regime number to break. "
                f"Valid regime numbers are between 0 and {len(bkpts)} inclusive. "
                f"Suggested: regime {suggested}. Or enter 'q' to quit. "
            )
        else:
            regime = input(
                "Invalid regime number. Regime number must be between 0 and "
                f"{len(bkpts)} inclusive. Enter a regime number to break. "
                f"Suggested: regime {suggested}. Or enter 'q' to quit. "
            )
        not_digit_or_quit = not regime.isdigit() and regime.lower() not in quits
        digit_out_of_range = regime.isdigit() and (int(regime) < 0 or int(regime) > len(bkpts))
    return regime.lower()


def _get_user_requested_features(
    df: pl.DataFrame,
    loss_matrix: np.ndarray,
    regime: int,
    num_worst_features: int,
) -> list[str]:
    k = min(num_worst_features, loss_matrix.shape[0])
    worst_features_idx = np.argpartition(loss_matrix[:, regime], -k)[-k:]
    while True:
        feat_cols = frames.feature_columns(df)
        allowed_features = ",\n".join(feat_cols)
        print(f"Valid features to retrain on:\n{allowed_features}\n")
        highest_error_features = ",\n".join([feat_cols[i] for i in worst_features_idx])
        print(f"Highest error features in regime {regime} are:\n{highest_error_features}")
        features = input(
            "Enter a list of features, separated by a comma and space, to "
            "retrain on. Or press enter to use highest error features "
            "listed above. "
        )
        if not features.strip():
            return [feat_cols[i] for i in worst_features_idx]
        valid_format = all(word.strip() for word in features.split(", "))
        if valid_format:
            requested_columns = list({word.strip() for word in features.split(", ")})
            invalid_columns = [col for col in requested_columns if col not in feat_cols]
            if requested_columns and not invalid_columns:
                return requested_columns
            if not requested_columns:
                print("No valid columns provided.")
            if invalid_columns:
                print(
                    "The following requested features are not valid: "
                    + ", ".join(invalid_columns)
                )
