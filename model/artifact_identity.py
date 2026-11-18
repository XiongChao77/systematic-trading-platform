"""Canonical training identity including every fusion component and window plan."""

import json
from pathlib import Path


def training_identity(directory, single_config):
    path = Path(directory) / "rolling_manifest.json"
    if not path.is_file():
        return single_config
    manifest = json.loads(path.read_text())
    if len(manifest["configs"]) == 1:
        return single_config
    components = {}
    for role, source in sorted(manifest.get("sources", {}).items()):
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        child = json.loads((source_path / "rolling_manifest.json").read_text())
        components[role] = {key: child[key] for key in ("configs", "planned_windows", "data_id")}
    return {
        "schema": "window_training_v1",
        "configs": manifest["configs"],
        "planned_windows": manifest.get("planned_windows", [
            {key: fold[key] for key in ("fold_id", "train_start", "validation_start", "oos_start", "oos_end", "phase")}
            for fold in manifest["folds"]
        ]),
        "data_id": manifest["data_id"],
        "components": components,
        "fusion_rule": "hard_trigger_direction_or_independent_ovr_v1" if len(manifest["configs"]) > 1 else None,
    }
