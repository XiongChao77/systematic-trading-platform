"""Persist the first balance and start time for each strategy in a runner."""

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def initialize_strategies(path, pipelines):
    path = Path(path)
    records = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(records, dict):
        raise ValueError("Initial strategy state must be an object")
    for strategy_id, record in records.items():
        if not isinstance(record, dict) or set(record) != {
            "initial_balance", "start_time", "strategy_hash"
        }:
            raise ValueError(f"Invalid initial state for {strategy_id}")
        balance = record["initial_balance"]
        if balance is not None and (
            isinstance(balance, bool) or not isinstance(balance, (int, float))
            or not math.isfinite(balance) or balance < 0
        ):
            raise ValueError(f"Invalid initial balance for {strategy_id}")
        if datetime.fromisoformat(record["start_time"]).utcoffset() is None:
            raise ValueError(f"Initial start time must include a timezone: {strategy_id}")
        if not isinstance(record["strategy_hash"], str) or not record["strategy_hash"]:
            raise ValueError(f"Invalid strategy hash for {strategy_id}")
    changed = False
    for pipeline in pipelines:
        strategy_id = pipeline.spec.strategy_id
        record = records.get(strategy_id)
        if record is not None:
            if record["strategy_hash"] != pipeline.spec.hash_id:
                raise ValueError(f"Initial state strategy hash mismatch: {strategy_id}")
        else:
            balance = float(pipeline.venue.get_dashboard_balance().balance)
            if not math.isfinite(balance) or balance < 0:
                raise ValueError(f"Invalid initial balance for {strategy_id}")
            records[strategy_id] = {
                "initial_balance": balance,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "strategy_hash": pipeline.spec.hash_id,
            }
            changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
                temporary = handle.name
                json.dump(records, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
    for pipeline in pipelines:
        record = records[pipeline.spec.strategy_id]
        pipeline.initial_balance = record["initial_balance"]
        pipeline.start_time = record["start_time"]
