"""Merge and deduplicate execution traces from multiple live-run directories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from trade.core.execution_status import execution_status, order_status, TERMINAL_STATUSES
from trade.recording.execution_trace import LiveExecutionTraceRecorder, SCHEMA_VERSION

_INTERNAL_SOURCE = "__source_path"

def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(value: float | None) -> str:
    return "" if value is None else format(float(value), ".17g")


def _parse_time(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Execution timestamp must be timezone-aware: {text}")
    return parsed.astimezone(UTC)


def _latest_time(*values: object) -> str:
    timestamps = [_parse_time(value) for value in values]
    present = [value for value in timestamps if value is not None]
    return max(present).isoformat() if present else ""


def _earliest_time(*values: object) -> str:
    timestamps = [_parse_time(value) for value in values]
    present = [value for value in timestamps if value is not None]
    return min(present).isoformat() if present else ""


def _scope(row: dict[str, str]) -> str:
    if row.get("trader_login"):
        return f"trader_login:{row['trader_login']}"
    account_id = row.get("account_id", "")
    if account_id:
        return f"account:{account_id}"
    return ":".join(
        (
            "strategy",
            row.get("runner_id", ""),
            row.get("instance_id", ""),
        )
    )


def _external_key(
    row: dict[str, str],
    identifier: str,
    kind: str,
) -> tuple[str, ...] | None:
    value = str(row.get(identifier, "") or "")
    if not value:
        return None
    return (
        kind,
        row.get("venue", ""),
        _scope(row),
        row.get("venue_symbol", ""),
        value,
    )


class _UnionFind:
    def __init__(self) -> None:
        self.parents: dict[tuple[str, ...], tuple[str, ...]] = {}

    def find(self, value: tuple[str, ...]) -> tuple[str, ...]:
        parent = self.parents.setdefault(value, value)
        if parent != value:
            self.parents[value] = self.find(parent)
        return self.parents[value]

    def union(self, left: tuple[str, ...], right: tuple[str, ...]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        canonical = min(left_root, right_root)
        other = right_root if canonical == left_root else left_root
        self.parents[other] = canonical


def _run_id_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name == "execution_traces":
            return parent.parent.name
    return path.parent.name


def _runner_id_from_filename(path: Path) -> str:
    marker = path.name.find("_20")
    return path.name[:marker] if marker > 0 else "unknown"


def _legacy_identity(path: Path, row: dict[str, str]) -> dict[str, str]:
    symbol = row.get("symbol", "")
    return {
        "schema_version": "1",
        "runner_id": _runner_id_from_filename(path),
        "run_id": _run_id_from_path(path),
        "instance_id": row.get("instance_id", ""),
        "strategy_hash": row.get("strategy_hash", ""),
        "venue": row.get("venue", ""),
        "account_id": "",
        "strategy_symbol": symbol,
        "venue_symbol": symbol,
    }


def _legacy_execution(path: Path, row: dict[str, str]) -> dict[str, str]:
    identity = _legacy_identity(path, row)
    converted = {field: "" for field in LiveExecutionTraceRecorder.EXECUTION_FIELDS}
    converted.update(identity)
    for field in (
        "execution_id",
        "side",
        "status",
        "reason",
        "decision_at_utc",
        "completed_at_utc",
        "requested_quantity",
        "filled_quantity",
        "decision_price",
        "arrival_price",
        "best_fill_price",
        "worst_fill_price",
        "fill_vwap",
        "execution_slippage_vwap_bps",
        "execution_slippage_worst_bps",
        "bid",
        "ask",
        "spread_pct",
        "fill_count",
    ):
        converted[field] = row.get(field, "")
    converted["order_role"] = "entry"
    converted["submitted_quantity"] = (
        row.get("filled_quantity", "")
        if row.get("status") == "filled" and row.get("filled_quantity")
        else row.get("requested_quantity", "")
    )
    converted["submitted_at_utc"] = ""
    converted["accepted_at_utc"] = ""
    converted["first_fill_at_utc"] = ""
    converted["decision_slippage_bps"] = row.get(
        "latency_slippage_bps",
        "",
    )
    return converted


def _legacy_fill(path: Path, row: dict[str, str]) -> dict[str, str]:
    identity = _legacy_identity(path, row)
    converted = {field: "" for field in LiveExecutionTraceRecorder.FILL_FIELDS}
    converted.update(identity)
    for field in (
        "execution_id",
        "side",
        "fill_index",
        "order_id",
        "deal_id",
        "executed_at_utc",
        "price",
        "quantity",
        "is_aggregate",
    ):
        converted[field] = row.get(field, "")
    converted["order_role"] = "entry"
    return converted


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row[_INTERNAL_SOURCE] = str(path)
    return rows


def _discover(
    input_dir: Path,
    output_dir: Path,
) -> dict[str, list[Path]]:
    discovered = {name: [] for name in ("executions", "orders", "fills", "events")}
    resolved_output = output_dir.resolve()
    for path in sorted(input_dir.rglob("*.csv")):
        try:
            path.resolve().relative_to(resolved_output)
            continue
        except ValueError:
            pass
        for kind in discovered:
            if path.name == f"{kind}.csv" or path.name.endswith(f"_{kind}.csv"):
                discovered[kind].append(path)
                break
    return discovered


def _load_rows(
    paths: dict[str, list[Path]],
) -> tuple[dict[str, list[dict[str, str]]], int]:
    rows = {name: [] for name in paths}
    legacy_files = 0
    for path in paths["executions"]:
        rows["executions"].extend(_read_csv(path))
    for path in paths["orders"]:
        loaded = _read_csv(path)
        if loaded and "schema_version" not in loaded[0]:
            legacy_files += 1
            rows["executions"].extend(_legacy_execution(path, row) for row in loaded)
        else:
            rows["orders"].extend(loaded)
    for path in paths["fills"]:
        loaded = _read_csv(path)
        if loaded and "schema_version" not in loaded[0]:
            legacy_files += 1
            rows["fills"].extend(_legacy_fill(path, row) for row in loaded)
        else:
            rows["fills"].extend(loaded)
    for path in paths["events"]:
        rows["events"].extend(_read_csv(path))
    return rows, legacy_files


def _normalize_order_metadata(rows):
    """Use real order/deal identity to repair sparse response observations."""
    sides = {}
    client_orders = defaultdict(set)
    for kind in ("orders", "fills"):
        for row in rows[kind]:
            order = _external_key(row, "order_id", "order")
            if order and row.get("side"):
                sides[order] = row["side"]
    for values in rows.values():
        for row in values:
            order = _external_key(row, "order_id", "order")
            if order in sides:
                row["side"] = sides[order]
            client = _external_key(row, "client_order_id", "client_order")
            if client and order:
                client_orders[client + (row.get("side", ""),)].add(row["order_id"])
    for values in rows.values():
        for row in values:
            client = _external_key(row, "client_order_id", "client_order")
            candidates = client_orders.get(client + (row.get("side", ""),), set()) if client else set()
            if not row.get("order_id") and len(candidates) == 1:
                row["order_id"] = next(iter(candidates))


def _separate_protection_orders(rows):
    """Separate opposite-side protection fills attached to an entry report."""
    reports = {_execution_key(row): row for row in rows["executions"]}
    for kind in ("orders", "fills", "events"):
        for row in rows[kind]:
            report = reports.get(_execution_key(row))
            if (report and row.get("side") and report.get("side")
                    and row["side"] != report["side"]):
                identifier = row.get("order_id") or row.get("client_order_id") or row.get("deal_id")
                if identifier:
                    row["execution_id"] += f":protection:{identifier}"
                    row["order_role"] = "exit"


def _link_execution_ids(rows: dict[str, list[dict[str, str]]]) -> _UnionFind:
    union_find = _UnionFind()
    identifiers = {}
    for kind in ("executions", "orders", "fills", "events"):
        for row in rows[kind]:
            if not row.get("execution_id"):
                continue
            node = _execution_key(row) + (row.get("side", ""),)
            union_find.find(node)
            keys = [_external_key(row, "order_id", "order"),
                    _external_key(row, "client_order_id", "client_order"),
                    _external_key(row, "deal_id", "deal")]
            for key in (key for key in keys if key is not None):
                # Opening and closing orders can echo the same client identifier.
                key += (row.get("side", ""),)
                previous = identifiers.setdefault(key, node)
                union_find.union(previous, node)
    return union_find


def _canonicalize(rows, union_find):
    for values in rows.values():
        for row in values:
            if row.get("execution_id"):
                node = _execution_key(row) + (row.get("side", ""),)
                row["execution_id"] = union_find.find(node)[3]


def _deduplicate(
    rows: Iterable[dict[str, str]],
    key_function,
    preference_function,
) -> tuple[list[dict[str, str]], int]:
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    duplicate_count = 0
    for row in rows:
        key = key_function(row)
        previous = selected.get(key)
        if previous is None:
            selected[key] = row
            continue
        duplicate_count += 1
        if preference_function(row) > preference_function(previous):
            selected[key] = row
    return [selected[key] for key in sorted(selected)], duplicate_count


def _fill_key(row: dict[str, str]) -> tuple[str, ...]:
    deal_key = _external_key(row, "deal_id", "deal")
    if deal_key is not None:
        return deal_key
    return (
        "fill_fingerprint",
        row.get("venue", ""),
        _scope(row),
        row.get("venue_symbol", ""),
        row.get("order_id", ""),
        row.get("executed_at_utc", ""),
        row.get("price", ""),
        row.get("quantity", ""),
    )


def _event_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("venue", ""), _scope(row), row.get("venue_symbol", ""),
        row.get("order_id", ""), row.get("client_order_id", ""),
        row.get("event_id", ""), row.get("status", ""),
        row.get("deal_id", ""), row.get("event_at_utc", ""),
    )


def _order_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        _external_key(row, "order_id", "order")
        or _external_key(row, "client_order_id", "client_order")
        or (
            "local_order",
            row.get("venue", ""), _scope(row),
            row.get("execution_id", ""),
            row.get("order_index", ""),
        )
    )


def _execution_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        "execution",
        row.get("venue", ""),
        _scope(row),
        row.get("execution_id", ""),
    )


def _time_preference(row: dict[str, str], field: str) -> tuple[str, str]:
    return row.get(field, ""), row.get(_INTERNAL_SOURCE, "")


def _overlay_execution_rows(
    executions: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in executions:
        grouped[_execution_key(row)].append(row)
    merged = []
    duplicate_count = 0
    for key in sorted(grouped):
        duplicate_count += len(grouped[key]) - 1
        candidates = sorted(
            grouped[key],
            key=lambda row: _time_preference(row, "completed_at_utc"),
        )
        output = {field: "" for field in LiveExecutionTraceRecorder.EXECUTION_FIELDS}
        for candidate in candidates:
            for field in output:
                if candidate.get(field, "") != "":
                    output[field] = candidate[field]
        output["run_id"] = "merged"
        merged.append(output)
    return merged, duplicate_count


def _synthesize_orders(
    orders: list[dict[str, str]],
    events: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    rows = list(orders)
    for event in events:
        if not event.get("order_id") and not event.get("client_order_id"):
            continue
        row = {field: "" for field in LiveExecutionTraceRecorder.CHILD_ORDER_FIELDS}
        for field in LiveExecutionTraceRecorder.IDENTITY_FIELDS:
            row[field] = event.get(field, "")
        row.update(
            execution_id=event.get("execution_id", ""),
            order_role=event.get("order_role", ""),
            side=event.get("side", ""),
            order_index="",
            order_id=event.get("order_id", ""),
            client_order_id=event.get("client_order_id", ""),
            submitted_quantity=event.get("submitted_quantity", ""),
            status=event.get("status", ""),
        )
        row[_INTERNAL_SOURCE] = event.get(_INTERNAL_SOURCE, "")
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[_order_key(row)].append(row)
    deduplicated = []
    duplicate_count = 0
    for candidates in grouped.values():
        duplicate_count += len(candidates) - 1
        state = order_status([row.get("status", "") for row in candidates])
        selected = next((row for row in reversed(candidates) if row.get("status") == state), candidates[-1])
        result = dict(selected)
        # Sparse events should not erase order size or client identifiers.
        for candidate in candidates:
            for field, value in candidate.items():
                if not result.get(field) and value:
                    result[field] = value
        result["status"] = state
        deduplicated.append(result)
    for index, row in enumerate(deduplicated):
        if not row.get("order_index"):
            row["order_index"] = str(index)
        row["run_id"] = "merged"
    return deduplicated, duplicate_count


def _drop_superseded_aggregate_fills(
    fills: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    detailed_orders = {
        (
            row.get("venue", ""),
            _scope(row),
            row.get("venue_symbol", ""),
            row.get("order_id", ""),
        )
        for row in fills
        if row.get("is_aggregate", "").casefold() != "true" and row.get("order_id")
    }
    output = []
    dropped = 0
    for row in fills:
        key = (
            row.get("venue", ""),
            _scope(row),
            row.get("venue_symbol", ""),
            row.get("order_id", ""),
        )
        if (
            row.get("is_aggregate", "").casefold() == "true"
            and row.get("order_id")
            and key in detailed_orders
        ):
            dropped += 1
            continue
        output.append(row)
    return output, dropped


def _status_from_lifecycle(submitted_quantity, filled_quantity, orders, fallback):
    return execution_status(
        [order.get("status", "unknown") for order in orders],
        submitted_quantity, filled_quantity, fallback,
    )


def _calculate_bps(
    side: str,
    final_price: float | None,
    reference_price: float | None,
) -> float | None:
    if final_price is None or reference_price is None or reference_price <= 0:
        return None
    direction = 1.0 if side == "buy" else -1.0
    return direction * (final_price - reference_price) / reference_price * 10_000.0


def _finalize_executions(
    executions: list[dict[str, str]],
    orders: list[dict[str, str]],
    fills: list[dict[str, str]],
    events: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_execution = {_execution_key(row): row for row in executions}
    identity_sources = events + fills + orders
    for source in identity_sources:
        execution_id = source.get("execution_id", "")
        key = _execution_key(source)
        if not execution_id or key in by_execution:
            continue
        row = {field: "" for field in LiveExecutionTraceRecorder.EXECUTION_FIELDS}
        for field in LiveExecutionTraceRecorder.IDENTITY_FIELDS:
            row[field] = source.get(field, "")
        row["run_id"] = "merged"
        row["execution_id"] = execution_id
        row["order_role"] = source.get("order_role", "")
        row["side"] = source.get("side", "")
        row["status"] = source.get("status", "submitted")
        row["reason"] = source.get("reason", "")
        by_execution[key] = row

    orders_by_execution: dict[str, list[dict[str, str]]] = defaultdict(list)
    fills_by_execution: dict[str, list[dict[str, str]]] = defaultdict(list)
    events_by_execution: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in orders:
        orders_by_execution[_execution_key(row)].append(row)
    for row in fills:
        fills_by_execution[_execution_key(row)].append(row)
    for row in events:
        events_by_execution[_execution_key(row)].append(row)

    finalized = []
    for execution_id in sorted(by_execution):
        row = by_execution[execution_id]
        execution_orders = orders_by_execution[execution_id]
        execution_fills = fills_by_execution[execution_id]
        execution_events = events_by_execution[execution_id]
        submitted_quantity = _float(row.get("submitted_quantity"))
        if submitted_quantity is None and execution_orders:
            quantities = [
                _float(order.get("submitted_quantity")) for order in execution_orders
            ]
            submitted_quantity = sum(
                quantity for quantity in quantities if quantity is not None
            )
        for order in execution_orders:
            child_fills = [fill for fill in execution_fills if _order_key(fill) == _order_key(order)]
            quantity = sum(_float(fill.get("quantity")) or 0 for fill in child_fills)
            if quantity:
                target = _float(order.get("submitted_quantity"))
                if target and quantity + 1e-12 >= target:
                    order["status"] = "filled"
                elif order["status"] not in TERMINAL_STATUSES:
                    order["status"] = "partially_filled"
        priced_fills = [
            (_float(fill.get("price")), _float(fill.get("quantity")), fill)
            for fill in execution_fills
        ]
        priced_fills = [
            (price, quantity, fill)
            for price, quantity, fill in priced_fills
            if price is not None and quantity is not None and quantity > 0
        ]
        filled_quantity = sum(quantity for _, quantity, _ in priced_fills)
        fill_vwap = (
            sum(price * quantity for price, quantity, _ in priced_fills)
            / filled_quantity
            if filled_quantity > 0
            else None
        )
        detailed_prices = [
            price
            for price, _, fill in priced_fills
            if fill.get("is_aggregate", "").casefold() != "true"
        ]
        side = row.get("side", "")
        best_fill = (
            min(detailed_prices)
            if side == "buy" and detailed_prices
            else max(detailed_prices) if side == "sell" and detailed_prices else None
        )
        worst_fill = (
            max(detailed_prices)
            if side == "buy" and detailed_prices
            else min(detailed_prices) if side == "sell" and detailed_prices else None
        )
        bid = _float(row.get("bid"))
        ask = _float(row.get("ask"))
        arrival = ask if side == "buy" else bid if side == "sell" else None
        decision_price = _float(row.get("decision_price"))
        fill_times = [
            _parse_time(fill.get("executed_at_utc")) for fill in execution_fills
        ]
        fill_times = [value for value in fill_times if value is not None]
        first_fill = min(fill_times) if fill_times else None
        decision_at = _parse_time(row.get("decision_at_utc"))
        submitted_at = _parse_time(row.get("submitted_at_utc"))
        row["run_id"] = "merged"
        row["submitted_quantity"] = _number(submitted_quantity)
        if not row.get("requested_quantity"):
            row["requested_quantity"] = _number(submitted_quantity)
        row["filled_quantity"] = _number(filled_quantity)
        row["fill_vwap"] = _number(fill_vwap)
        row["best_fill_price"] = _number(best_fill)
        row["worst_fill_price"] = _number(worst_fill)
        row["arrival_price"] = _number(arrival)
        row["first_fill_at_utc"] = (
            first_fill.isoformat() if first_fill is not None else ""
        )
        row["status"] = _status_from_lifecycle(
            submitted_quantity,
            filled_quantity,
            execution_orders,
            row.get("status", "submitted"),
        )
        latest_reason = next(
            (
                event.get("reason", "")
                for event in sorted(
                    execution_events,
                    key=lambda item: item.get("event_at_utc", ""),
                    reverse=True,
                )
                if event.get("reason")
            ),
            "",
        )
        if latest_reason:
            row["reason"] = latest_reason
        terminal = execution_orders and all(order.get("status") in TERMINAL_STATUSES for order in execution_orders)
        row["completed_at_utc"] = _latest_time(
            *[event.get("event_at_utc", "") for event in execution_events if event.get("status") in TERMINAL_STATUSES],
            *[fill.get("executed_at_utc", "") for fill in execution_fills],
        ) if terminal else ""
        row["accepted_at_utc"] = _earliest_time(
            row.get("accepted_at_utc"),
            *[
                event.get("event_at_utc", "")
                for event in execution_events
                if event.get("status") == "accepted"
            ],
        )
        accepted_at = _parse_time(row.get("accepted_at_utc"))
        row["decision_slippage_bps"] = _number(
            _calculate_bps(side, fill_vwap, decision_price)
        )
        row["implementation_shortfall_bps"] = _number(
            _calculate_bps(side, fill_vwap, arrival)
        )
        row["execution_slippage_vwap_bps"] = _number(
            _calculate_bps(side, fill_vwap, best_fill)
        )
        row["execution_slippage_worst_bps"] = _number(
            _calculate_bps(side, worst_fill, best_fill)
        )
        row["decision_to_first_fill_ms"] = _number(
            (first_fill - decision_at).total_seconds() * 1_000.0
            if first_fill is not None and decision_at is not None
            else None
        )
        row["submit_to_first_fill_ms"] = _number(
            (first_fill - submitted_at).total_seconds() * 1_000.0
            if first_fill is not None and submitted_at is not None
            else None
        )
        row["acknowledgement_latency_ms"] = _number(
            (accepted_at - submitted_at).total_seconds() * 1_000.0
            if accepted_at is not None and submitted_at is not None
            else None
        )
        row["child_order_count"] = str(len(execution_orders))
        row["fill_count"] = str(len(execution_fills))
        finalized.append(row)
    return finalized


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[dict[str, str]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(temporary, path)


def merge_execution_traces(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Merge all execution trace sessions below ``input_dir``."""

    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Execution trace input directory not found: {root}")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else root / "merged_execution_traces"
    )
    destination.mkdir(parents=True, exist_ok=True)
    paths = _discover(root, destination)
    rows, legacy_files = _load_rows(paths)
    if not any(rows.values()):
        raise ValueError(f"No execution trace records found below {root}")

    input_counts = {kind: len(values) for kind, values in rows.items()}
    initial_count = sum(":initial:" in row.get("event_id", "") for row in rows["events"])
    rows["events"] = [row for row in rows["events"] if ":initial:" not in row.get("event_id", "")]
    _normalize_order_metadata(rows)
    _separate_protection_orders(rows)
    union_find = _link_execution_ids(rows)
    _canonicalize(rows, union_find)

    fills, duplicate_fills = _deduplicate(
        rows["fills"],
        _fill_key,
        lambda row: (
            row.get("is_aggregate", "").casefold() != "true",
            row.get("executed_at_utc", ""),
            row.get(_INTERNAL_SOURCE, ""),
        ),
    )
    fills, superseded_aggregate_fills = _drop_superseded_aggregate_fills(fills)
    events, duplicate_events = _deduplicate(
        rows["events"],
        _event_key,
        lambda row: _time_preference(row, "event_at_utc"),
    )
    orders, duplicate_orders = _synthesize_orders(rows["orders"], events)
    executions, duplicate_executions = _overlay_execution_rows(rows["executions"])
    executions = _finalize_executions(
        executions,
        orders,
        fills,
        events,
    )

    for collection in (executions, orders, fills, events):
        for row in collection:
            row["schema_version"] = SCHEMA_VERSION
            row["run_id"] = "merged"
    executions.sort(
        key=lambda row: (
            row.get("decision_at_utc", ""),
            row.get("execution_id", ""),
        )
    )
    orders.sort(key=_order_key)
    fills.sort(key=_fill_key)
    events.sort(
        key=lambda row: (
            row.get("event_at_utc", ""),
            _event_key(row),
        )
    )

    _write_csv(
        destination / "executions.csv",
        LiveExecutionTraceRecorder.EXECUTION_FIELDS,
        executions,
    )
    _write_csv(
        destination / "orders.csv",
        LiveExecutionTraceRecorder.CHILD_ORDER_FIELDS,
        orders,
    )
    _write_csv(
        destination / "fills.csv",
        LiveExecutionTraceRecorder.FILL_FIELDS,
        fills,
    )
    _write_csv(
        destination / "events.csv",
        LiveExecutionTraceRecorder.EVENT_FIELDS,
        events,
    )
    report = {
        "schema_version": int(SCHEMA_VERSION),
        "input_dir": str(root),
        "output_dir": str(destination),
        "source_files": {kind: len(values) for kind, values in paths.items()},
        "input_records": input_counts,
        "output_records": {
            "executions": len(executions),
            "orders": len(orders),
            "fills": len(fills),
            "events": len(events),
        },
        "ignored_response_snapshot_events": initial_count,
        "deduplication": {
            "duplicate_executions": duplicate_executions,
            "duplicate_orders": duplicate_orders,
            "duplicate_fills": duplicate_fills,
            "duplicate_events": duplicate_events,
            "superseded_aggregate_fills": superseded_aggregate_fills,
        },
        "legacy_files_converted": legacy_files,
        "limitations": (
            [
                "Legacy traces lack stable account and venue-symbol identifiers; "
                "their cross-process deduplication is best-effort."
            ]
            if legacy_files
            else []
        ),
    }
    report_path = destination / "merge_report.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    with temporary_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_report, report_path)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge live execution traces below a runner directory.",
    )
    parser.add_argument("input_dir", help="Runner or execution-trace root")
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: INPUT/merged_execution_traces)",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = merge_execution_traces(
        arguments.input_dir,
        arguments.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
