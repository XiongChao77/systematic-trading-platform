from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any

import torch

from data_process import common
from model.evaluator import (
    confusion_frame,
    plot_precision_recall_curve,
    plot_threshold_precision_recall_curve,
    plot_training_curves,
)
from model.training_types import FitResult, PredictionResult


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def copy_data_config_meta(source_dir: str, target_dir: str) -> str:
    """Copy preparation config and its data provenance record into an artifact dir.

    Downstream consumers (e.g. trade.runner.backtest_runner) read the preparation params
    straight from the training output dir, so every artifact dir must carry its
    own copy of the meta written by data_process.preparation.
    """
    source = common.get_data_config_path_in_dir(source_dir)
    if not os.path.exists(source):
        raise FileNotFoundError(f"❌ data_config_meta.json not found in {source_dir}")
    os.makedirs(target_dir, exist_ok=True)
    target = common.get_data_config_path_in_dir(target_dir)
    if os.path.abspath(source) != os.path.abspath(target):
        shutil.copyfile(source, target)
    manifest_source = common.get_data_manifest_path_in_dir(source_dir)
    manifest_target = common.get_data_manifest_path_in_dir(target_dir)
    if os.path.abspath(manifest_source) != os.path.abspath(manifest_target):
        shutil.copyfile(manifest_source, manifest_target)
    return target


def save_single_run(
    *,
    model: torch.nn.Module,
    task_type: str,
    fit_result: FitResult,
    prediction: PredictionResult,
    metrics: dict,
    save_dir: str,
    prep_output_dir: str,
    feature_names: list[str],
    feature_list: list[str],
    label_col: str,
    seq_len: int,
    train_config: dict[str, Any],
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    copy_data_config_meta(prep_output_dir, save_dir)
    model_path = os.path.join(save_dir, "model.pt")
    meta_path = os.path.join(save_dir, "meta.json")
    metrics_path = os.path.join(save_dir, "metrics.json")
    history_path = os.path.join(save_dir, "history.json")
    train_config_path = os.path.join(save_dir, "train_config.json")

    state_source = model._orig_mod if hasattr(model, "_orig_mod") else model
    meta = state_source.export_meta(
        feature_cols=feature_names,
        label_col=label_col,
        classes=metrics["labels"],
        seq_len=seq_len,
        task_type=task_type,
    )
    meta["arch_hash"] = state_source.architecture_hash(meta)

    torch.save(
        {
            "state_dict": state_source.state_dict(),
            "architecture": state_source.identity(),
            "arch_hash": meta["arch_hash"],
            "feature_list": feature_list,
            "classes": metrics["labels"],
            "channel": len(feature_names),
            "seq_len": seq_len,
            "feature_cols": feature_names,
            "label_col": label_col,
            "selection_metric": fit_result.selection_metric,
            "validation_score": fit_result.validation_score,
            "task_type": task_type,
        },
        model_path,
    )

    _write_json(meta_path, meta)
    metrics_report = {
        "task_type": task_type,
        "selection": {
            "metric": fit_result.selection_metric,
            "validation_score": fit_result.validation_score,
            "validation_metrics": fit_result.validation_metrics,
        },
        "test": metrics,
    }
    _write_json(metrics_path, metrics_report)
    _write_json(history_path, fit_result.history)
    _write_json(train_config_path, train_config)
    confusion_frame(metrics).to_csv(os.path.join(save_dir, "confusion_matrix.csv"))
    plot_precision_recall_curve(
        prediction,
        metrics["labels"],
        os.path.join(save_dir, "precision_recall_curve.png"),
    )
    plot_threshold_precision_recall_curve(
        prediction,
        metrics["labels"],
        os.path.join(save_dir, "threshold_precision_recall_curve.png"),
    )
    plot_training_curves(
        fit_result,
        os.path.join(save_dir, "training_curve.png"),
    )

    description = {
        "task_type": task_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {"main": {"model": "model.pt", "meta": "meta.json"}},
        "metrics": "metrics.json",
        "history": "history.json",
        "precision_recall_curve": "precision_recall_curve.png",
        "threshold_precision_recall_curve": "threshold_precision_recall_curve.png",
        "training_curve": "training_curve.png" if fit_result.history else None,
        "train_config": "train_config.json",
        "data_config": "data_config_meta.json",
        "data_manifest": common.DATA_MANIFEST_FILENAME,
    }
    _write_json(os.path.join(save_dir, "task_description.json"), description)
    return metrics_report


def save_fusion_run(
    *,
    task_type: str,
    role_directories: dict[str, str],
    fusion_dir: str,
) -> dict:
    """Point a fusion artifact dir at two independently trained single-model runs.

    Unlike save_single_run, this does not train or evaluate anything: it just
    records, per role, the (relative) path to an existing save_single_run
    output directory so inference-time code (ModelHandler/FusionWrapper) can
    load both checkpoints and fuse them.
    """
    os.makedirs(fusion_dir, exist_ok=True)
    for directory in role_directories.values():
        if os.path.exists(common.get_data_config_path_in_dir(directory)):
            copy_data_config_meta(directory, fusion_dir)
            break
    else:
        raise FileNotFoundError(
            "❌ data_config_meta.json not found in any role directory: "
            f"{sorted(role_directories.values())}"
        )
    description = {
        "task_type": task_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            role: os.path.relpath(os.path.abspath(directory), os.path.abspath(fusion_dir))
            for role, directory in role_directories.items()
        },
        "data_config": "data_config_meta.json",
        "data_manifest": common.DATA_MANIFEST_FILENAME,
    }
    _write_json(os.path.join(fusion_dir, "task_description.json"), description)
    return description
