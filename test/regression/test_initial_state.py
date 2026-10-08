"""Offline startup and trading main-flow checks using simulated exchange I/O."""

import json


from datetime import datetime


from types import SimpleNamespace as NS


from unittest.mock import Mock




from trade.runner.initial_state import initialize_strategies


def pipeline(name, balance=100):
    return NS(
        spec=NS(instance_id=name, hash_id=f"hash-{name}"),
        venue=NS(get_dashboard_balance=Mock(return_value=NS(balance=balance))),
    )



def test_startup_records_account_baseline(tmp_path):
    path = tmp_path / "initial.json"
    item = pipeline("main", 100)
    initialize_strategies(path, [item])
    assert item.initial_balance == 100
    assert datetime.fromisoformat(item.start_time).utcoffset() is not None
    assert json.loads(path.read_text())["main"]["initial_balance"] == 100
