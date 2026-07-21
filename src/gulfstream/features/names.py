"""Resolve feature name specs to concrete column names."""
from __future__ import annotations


def get_column_names(features_spec) -> list[str]:
    """Expand a feature request into concrete column names.

    Accepts list[str] (pass-through) or dict with optional 'columns' key.
    """
    if features_spec is None:
        return []
    if isinstance(features_spec, list):
        return [str(x) for x in features_spec]
    if isinstance(features_spec, dict):
        if "columns" in features_spec:
            return [str(x) for x in features_spec["columns"]]
        return [str(x) for x in features_spec.keys()]
    return [str(features_spec)]
