"""Fuse independent window runs by exact prediction time and model provenance."""

import json
import os
import shutil

import numpy as np
import pandas as pd

from model.artifacts import copy_data_config_meta, save_fusion_run
from model.train_config import TrainTask
from model.window_predictions import annotate_predictions, validate_predictions


def fuse_window_runs(*, task_type, role_directories, fusion_dir, logger):
    """Combine recorded probabilities without running any checkpoint on history.

    Different update schedules are aligned on exact (symbol, candle) keys. Each
    resulting fold references the two checkpoints that produced those rows.
    Missing component predictions remain missing; no forward/backward filling.
    """
    roles = list(role_directories)
    manifests, sources = {}, {}
    for role, directory in role_directories.items():
        path = os.path.join(directory, "rolling_manifest.json")
        if not os.path.isfile(path):
            raise ValueError(f"Fusion requires a train_windows output root, not a checkpoint: {directory}")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "complete" or manifest.get("prediction_contract") != 1:
            raise ValueError(f"Fusion requires a complete window prediction export: {directory}")
        if len(manifest["configs"]) != 1 or manifest["configs"][0]["train_task"].lower() != role:
            raise ValueError(f"Training root does not contain the expected standalone {role} task")
        frame = pd.read_pickle(os.path.join(directory, manifest["window_predictions"]))
        validate_predictions(frame, manifest, allow_in_sample=True)
        columns = ["symbol", "open_time_date_utc", "fold_id", "prediction_scope",
                   f"sub_{role}_pred", f"sub_{role}_prob_0", f"sub_{role}_prob_1"]
        sources[role] = frame[columns].rename(columns={"fold_id": f"{role}_fold_id",
                                                       "prediction_scope": f"{role}_scope"})
        manifests[role] = manifest
    left, right = (manifests[role] for role in roles)
    if not left.get("data_id") or left["data_id"] != right["data_id"]:
        raise ValueError("Fusion components have different preparation data_id values")
    if pd.Timestamp(left["forward_start"]) != pd.Timestamp(right["forward_start"]):
        raise ValueError("Fusion components have different forward boundaries")
    if os.path.exists(os.path.join(fusion_dir, "rolling_manifest.json")):
        raise FileExistsError(f"Fusion output already exists: {fusion_dir}")
    for directory in role_directories.values():
        if os.path.abspath(directory) == os.path.abspath(fusion_dir):
            raise ValueError("Fusion output must differ from its source directories")

    keys = ["symbol", "open_time_date_utc"]
    aligned = sources[roles[0]].merge(sources[roles[1]], on=keys, how="inner", validate="one_to_one")
    aligned["open_time_date_utc"] = pd.to_datetime(aligned["open_time_date_utc"], utc=True)
    first_ids = {role: manifests[role]["folds"][0]["fold_id"] for role in roles}
    both_oos = np.logical_and.reduce([aligned[f"{role}_scope"].eq("oos") for role in roles])
    first_pair = np.logical_and.reduce([aligned[f"{role}_fold_id"].eq(first_ids[role]) for role in roles])
    probabilities = [f"sub_{role}_prob_{index}" for role in roles for index in (0, 1)]
    finite = np.isfinite(aligned[probabilities].to_numpy(float)).all(axis=1)
    aligned = aligned.loc[(both_oos | first_pair) & finite].copy()
    if aligned.empty:
        raise ValueError("Fusion components have no common usable prediction timestamps")
    for role in roles:
        probs = aligned[[f"sub_{role}_prob_0", f"sub_{role}_prob_1"]].to_numpy(float)
        if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1.0):
            raise ValueError(f"Invalid probabilities in {role} export")
    if task_type == TrainTask.TRIGGER_DIRECTION:
        action = aligned.sub_trigger_prob_1.to_numpy() > aligned.sub_trigger_prob_0.to_numpy()
        p_short = np.where(action, aligned.sub_direction_prob_0, 0.0)
        p_long = np.where(action, aligned.sub_direction_prob_1, 0.0)
        p_neutral = (~action).astype(float)
        aligned["direction_pred"] = np.where(aligned.sub_direction_pred == 0, -1, 1)
    elif task_type == TrainTask.LONG_SHORT_OVR:
        long = aligned.sub_long_ovr_prob_1.to_numpy()
        short = aligned.sub_short_ovr_prob_1.to_numpy()
        p_short, p_long = short * (1 - long), long * (1 - short)
        p_neutral = (1 - short) * (1 - long) + short * long
        aligned["direction_pred"] = np.nan
    else:
        raise ValueError(f"Unsupported window fusion task: {task_type}")
    probs = np.column_stack((p_short, p_neutral, p_long))
    aligned[["prob_short", "prob_neutral", "prob_long"]] = probs
    aligned["pred"] = np.array([-1, 0, 1])[probs.argmax(axis=1)]
    aligned["pred_prob"] = probs.max(axis=1)
    aligned["net_score"] = p_long - p_short
    aligned["regression_score"] = np.nan

    os.makedirs(fusion_dir, exist_ok=True)
    first_directory = role_directories[roles[0]]
    copy_data_config_meta(first_directory, fusion_dir)
    shutil.copyfile(os.path.join(first_directory, "train_config.json"), os.path.join(fusion_dir, "train_config.json"))
    manifest = {"status": "running", "prediction_contract": 1, "data_id": left["data_id"],
                "window_mode": "fusion", "forward_start": left["forward_start"],
                "configs": [m["configs"][0] for m in manifests.values()], "folds": [],
                "sources": {role: os.path.abspath(path) for role, path in role_directories.items()},
                "predictions": "oos_predictions.pkl", "window_predictions": "window_predictions.pkl",
                "development_predictions": "development_predictions.pkl"}
    parts = []
    group_columns = [f"{role}_fold_id" for role in roles]
    aligned = aligned.sort_values(["open_time_date_utc", "symbol"])
    for pair, rows in aligned.groupby(group_columns, sort=False):
        source_folds = {role: next(f for f in manifests[role]["folds"] if f["fold_id"] == fold_id)
                        for role, fold_id in zip(roles, pair)}
        start = max(pd.Timestamp(f["oos_start"]) for f in source_folds.values())
        end = min(pd.Timestamp(f["oos_end"]) for f in source_folds.values())
        if start >= end:
            logger.warning("Skipping component pair with no common OOS window: %s", pair)
            continue
        phases = {f["phase"] for f in source_folds.values()}
        if len(phases) != 1:
            raise ValueError("Fusion component phases do not agree")
        fold = {"fold_id": f"{len(parts):04d}", "phase": phases.pop(),
                "train_start": max(pd.Timestamp(f["train_start"]) for f in source_folds.values()).isoformat(),
                "validation_start": max(pd.Timestamp(f["validation_start"]) for f in source_folds.values()).isoformat(),
                "oos_start": start.isoformat(), "oos_end": end.isoformat(),
                "source_folds": source_folds}
        times = rows["open_time_date_utc"]
        diagnostic = all(f["fold_id"] == first_ids[role] for role, f in source_folds.items())
        lower = pd.Timestamp(fold["train_start"]) if diagnostic else start
        rows = rows.loc[(times >= lower) & (times < end)].copy()
        if rows.empty:
            continue
        directory = os.path.join(fusion_dir, fold["fold_id"])
        checkpoint = os.path.join(directory, "fusion")
        save_fusion_run(task_type=task_type, fusion_dir=checkpoint,
                        role_directories={role: os.path.join(role_directories[role], f["model_dir"])
                                          for role, f in source_folds.items()})
        fold["model_dir"] = os.path.relpath(checkpoint, fusion_dir)
        rows = annotate_predictions(rows, fold)
        rows = rows.drop(columns=[f"{role}_scope" for role in roles])
        oos = rows.loc[rows.prediction_scope == "oos"]
        fold["window_predictions"] = f"{fold['fold_id']}/window_predictions.pkl"
        fold["predictions"] = f"{fold['fold_id']}/oos_predictions.pkl"
        fold["prediction_count"] = len(oos)
        rows.to_pickle(os.path.join(fusion_dir, fold["window_predictions"]))
        oos.to_pickle(os.path.join(fusion_dir, fold["predictions"]))
        with open(os.path.join(checkpoint, "window_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({**fold, "data_id": manifest["data_id"],
                       "training_root": os.path.relpath(fusion_dir, checkpoint)}, handle, indent=2)
        manifest["folds"].append(fold)
        parts.append(rows)
    if not parts:
        raise ValueError("Fusion has no overlapping OOS windows")
    window = pd.concat(parts, ignore_index=True).sort_values(["symbol", "open_time_date_utc"])
    oos = window.loc[window.prediction_scope == "oos"]
    validate_predictions(window, manifest, allow_in_sample=True)
    validate_predictions(oos, manifest)
    development_ids = {fold["fold_id"] for fold in manifest["folds"] if fold["phase"] == "development"}
    for key, frame in (("window_predictions", window), ("predictions", oos),
                       ("development_predictions", window.loc[window.fold_id.isin(development_ids)])):
        frame.to_pickle(os.path.join(fusion_dir, manifest[key]))
    manifest.update(status="complete", prediction_columns=list(window.columns))
    manifest_path = os.path.join(fusion_dir, "rolling_manifest.json")
    with open(manifest_path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(manifest_path + ".tmp", manifest_path)
    logger.info("Fused window predictions | folds=%d | OOS rows=%d | diagnostic rows=%d | output=%s",
                len(parts), len(oos), len(window) - len(oos), fusion_dir)
    return manifest
