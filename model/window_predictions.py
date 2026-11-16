"""Prediction provenance shared by window training, fusion and backtesting."""

import pandas as pd


def annotate_predictions(frame, fold):
    """Mark first-window diagnostics separately from deployable predictions.

    Boundaries use candle open timestamps, matching the training window planner.
    Predictions observe the corresponding closed candle. Availability describes
    the simulated training cutoff, rather than the wall-clock training run.
    """
    frame = frame.copy()
    times = pd.to_datetime(frame["open_time_date_utc"], utc=True)
    frame["fold_id"] = fold["fold_id"]
    frame["model_available_at"] = fold["oos_start"]
    frame["prediction_scope"] = "oos"
    frame.loc[times < pd.Timestamp(fold["oos_start"]), "prediction_scope"] = "in_sample_validation"
    frame.loc[times < pd.Timestamp(fold["validation_start"]), "prediction_scope"] = "in_sample_train"
    return frame


def validate_predictions(frame, manifest, *, allow_in_sample=False):
    """Reject overlaps, unknown models and predictions outside model windows."""
    required = {"symbol", "open_time_date_utc", "fold_id", "prediction_scope", "model_available_at"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Window prediction provenance is missing {sorted(missing)}; regenerate training artifacts")
    if frame[list(required)].isna().any().any():
        raise ValueError("Window prediction provenance contains missing values")
    if frame.duplicated(["symbol", "open_time_date_utc"]).any():
        raise ValueError("Window predictions overlap for the same symbol and timestamp")
    folds = {fold["fold_id"]: fold for fold in manifest["folds"]}
    if len(folds) != len(manifest["folds"]) or not folds:
        raise ValueError("Window manifest requires unique fold identifiers")
    first_id = manifest["folds"][0]["fold_id"]
    for fold_id, rows in frame.groupby("fold_id", sort=False):
        if fold_id not in folds:
            raise ValueError(f"Predictions reference unknown fold {fold_id}")
        fold = folds[fold_id]
        times = pd.to_datetime(rows["open_time_date_utc"], utc=True)
        available = pd.to_datetime(rows["model_available_at"], utc=True)
        train, valid, start, end = (pd.Timestamp(fold[key]) for key in
                                    ("train_start", "validation_start", "oos_start", "oos_end"))
        if not train < valid < start < end:
            raise ValueError(f"Invalid model time boundaries for fold {fold_id}")
        if not available.eq(start).all():
            raise ValueError(f"Model availability disagrees with fold {fold_id}")
        scopes = annotate_predictions(rows, fold)["prediction_scope"]
        if not rows["prediction_scope"].eq(scopes).all():
            raise ValueError(f"Prediction scope disagrees with timestamps for fold {fold_id}")
        lower = train if allow_in_sample and fold_id == first_id else start
        if not ((times >= lower) & (times < end)).all():
            raise ValueError(f"Predictions use a future model or exceed fold {fold_id} boundaries")
