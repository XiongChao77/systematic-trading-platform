"""Build per-strategy backtest and live curves from configured report artifacts."""

from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from math import isfinite
import sqlite3
from tempfile import NamedTemporaryFile

from experiment.report_service import (
    PERIODS,
    combine_daily_account,
    load_report_dataset,
    load_report_details,
    materialize_period_report,
)
from trade.recording.archive_bitget import connect
from trade.recording.report_bitget import bill_checkpoints, complete_bill_dates, day_stamp, report


CURVES_REPORT_NAME = "strategy_curves.json"
LOGGER = logging.getLogger(__name__)


def account_live_points(db, account, snapshots, earliest, settled_day):
    first_fill = db.execute(
        """SELECT min(created_ms) FROM archive
           WHERE stream='fills' AND account_id=? AND created_ms>=?""",
        (account, day_stamp(earliest, timezone.utc)),
    ).fetchone()[0]
    if first_fill is None:
        return None, []
    launch = datetime.fromtimestamp(first_fill / 1000, timezone.utc).date()
    real = {
        date.fromisoformat(row["date"]): row["account_equity"][account]
        for row in snapshots if account in row.get("account_equity", {})
        and row["date"] >= str(launch)
    }

    first, last = complete_bill_dates(bill_checkpoints(db, [account]), timezone.utc)
    start = max(launch, first)
    first_equity = min(real) if real else settled_day + timedelta(days=1)
    end = min(settled_day, last, max(start, first_equity - timedelta(days=1)))
    if start > end:
        return str(launch), []

    rows, metrics = report(db, [account], timezone.utc, start, end)
    opening = metrics["opening_balance"]
    if opening <= 0:
        return str(launch), []
    balances = [(day, float(balance)) for day, balance, *_ in rows if day < first_equity]
    values = balances + sorted(real.items())
    return str(launch), [{"time": str(day), "value": value / opening} for day, value in values]


def strategy_curve(record, root, db, account, snapshots, earliest, settled_day):
    details = load_report_details(record, root)
    periods = [
        materialize_period_report(record.raw, period, details)
        for period in PERIODS if period in details.get("results", {})
    ]
    daily = combine_daily_account(*periods)
    backtest = {str(row["date"])[:10]: row["end_equity"] for row in daily if isfinite(row["end_equity"])}
    try:
        launch, live = account_live_points(db, account, snapshots, earliest, settled_day)
    except (ValueError, sqlite3.Error) as exc:
        LOGGER.warning("Live strategy valuation unavailable for %s: %s", account, exc)
        launch, live = None, []
    backtest_points = [
        {"time": day, "value": backtest[day]} for day in sorted(backtest)
        if launch is None or day < launch
    ]
    if not backtest_points or backtest_points[-1]["value"] <= 0:
        raise ValueError("Backtest has no positive ending equity before live trading")
    scale = backtest_points[-1]["value"]
    return {
        "strategy_hash": record.raw["params"]["hash"],
        "backtest": backtest_points,
        "live_start": launch,
        "live": [{"time": point["time"], "value": point["value"] * scale} for point in live],
    }


def publish_strategy_curves(directory, config_path, accounts, snapshots, earliest, settled_day):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entries = [
        entry for entry in config["strategy"]
        if entry["venue"].strip().casefold() == "bitget"
        and str(entry["account_id"]).strip() in accounts
        and entry.get("enable", True) and entry.get("run_live")
    ]
    account_counts = Counter(str(entry["account_id"]).strip() for entry in entries)
    by_report = defaultdict(list)
    for entry in entries:
        account = str(entry["account_id"]).strip()
        if account_counts[account] == 1:
            path = (config_path.resolve().parent / entry["config_path"]).resolve()
            by_report[path].append((account, entry["hash"]))

    curves = {}
    with closing(connect(directory / "history.sqlite3")) as db:
        db.execute("BEGIN")
        for path, strategies in by_report.items():
            try:
                dataset = load_report_dataset(
                    [path.name], allowed_root=path.parent, deduplicate=False, allow_named_jsonl=True,
                )
            except (ValueError, OSError) as exc:
                LOGGER.warning("Cannot load backtest report %s: %s", path, exc)
                continue
            records = {}
            for record in dataset.records:
                records.setdefault(record.raw["params"].get("hash"), record)
            for account, strategy_hash in strategies:
                try:
                    curves[account] = strategy_curve(
                        records[strategy_hash], path.parent, db, account, snapshots, earliest, settled_day,
                    )
                except (KeyError, ValueError, OSError) as exc:
                    LOGGER.warning("Cannot build equity curve for strategy %s: %s", strategy_hash, exc)

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(curves, handle, allow_nan=False)
            handle.close()
            temporary.replace(directory / CURVES_REPORT_NAME)
        finally:
            temporary.unlink(missing_ok=True)
