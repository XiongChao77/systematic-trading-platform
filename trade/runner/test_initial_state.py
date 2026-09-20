"""Initial account baselines survive restarts and failed collection."""

import json
from datetime import datetime
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from trade.runner.initial_state import initialize_strategies


def pipeline(name, balance=100):
    return NS(
        spec=NS(strategy_id=name, hash_id=f"hash-{name}"),
        venue=NS(get_dashboard_balance=Mock(return_value=NS(balance=balance))),
    )


def test_restart_preserves_baseline_and_adds_only_new_strategies(tmp_path):
    path = tmp_path / "initial.json"
    first = pipeline("first")
    initialize_strategies(path, [first])
    original = json.loads(path.read_text())["first"]
    assert first.initial_balance == 100
    assert datetime.fromisoformat(first.start_time).utcoffset() is not None
    first.venue.get_dashboard_balance.side_effect = AssertionError("Existing baseline must not be fetched")
    second = pipeline("second", 200)
    initialize_strategies(path, [first, second])
    records = json.loads(path.read_text())
    assert records["first"] == original
    assert records["second"]["initial_balance"] == 200
    assert first.start_time == original["start_time"]
    before = path.read_bytes()
    initialize_strategies(path, [first])
    assert path.read_bytes() == before


def test_failed_collection_does_not_partially_save_baselines(tmp_path):
    path = tmp_path / "initial.json"
    initialize_strategies(path, [pipeline("existing")])
    before = path.read_bytes()
    failed = pipeline("failed")
    failed.venue.get_dashboard_balance.side_effect = TimeoutError("Account unavailable")
    with pytest.raises(TimeoutError):
        initialize_strategies(path, [pipeline("new"), failed])
    assert path.read_bytes() == before


def test_hash_mismatch_does_not_overwrite_history(tmp_path):
    path = tmp_path / "initial.json"
    item = pipeline("test")
    initialize_strategies(path, [item])
    before = path.read_bytes()
    item.spec.hash_id = "different"
    with pytest.raises(ValueError, match="hash mismatch"):
        initialize_strategies(path, [item])
    assert path.read_bytes() == before


def test_invalid_json_is_not_replaced(tmp_path):
    path = tmp_path / "initial.json"
    path.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        initialize_strategies(path, [pipeline("test")])
    assert path.read_text() == "{"
