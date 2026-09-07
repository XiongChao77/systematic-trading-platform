"""Single-process in-memory aggregation for live runner snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from UI.backend.modules.live.models import RunnerSnapshot, StrategySnapshot


RUNNER_STALE_SECONDS = 5.0


class LiveSnapshotConflict(ValueError):
    """Raised when two active runners claim the same identity or strategy."""


@dataclass
class RunnerState:
    instance_id: str
    started_at: datetime
    last_sequence: int
    last_received_at: datetime
    strategy_ids: set[str] = field(default_factory=set)


@dataclass
class StrategyState:
    runner_id: str
    runner_instance_id: str
    received_at: datetime
    snapshot: StrategySnapshot


class LiveSnapshotStore:
    """Keep only the newest complete snapshot from every runner."""

    def __init__(self, stale_seconds: float = RUNNER_STALE_SECONDS) -> None:
        self.stale_seconds = float(stale_seconds)
        self._lock = RLock()
        self._runners: dict[str, RunnerState] = {}
        self._strategies: dict[str, StrategyState] = {}
        self._conflicts: dict[str, set[str]] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def update(self, payload: RunnerSnapshot) -> dict[str, Any]:
        received_at = self._now()
        with self._lock:
            runner = self._runners.get(payload.runner_id)
            if runner is not None and runner.instance_id != payload.runner_instance_id:
                if payload.runner_started_at <= runner.started_at:
                    raise LiveSnapshotConflict(
                        f"Runner {payload.runner_id!r} is already owned by a newer instance"
                    )
                self._remove_runner_strategies(payload.runner_id, runner.instance_id)
                runner = None

            if runner is not None and payload.sequence <= runner.last_sequence:
                return {"accepted": False, "reason": "stale_sequence"}

            incoming = {item.strategy_id: item for item in payload.strategies}
            if len(incoming) != len(payload.strategies):
                raise LiveSnapshotConflict("Runner snapshot contains duplicate strategy IDs")

            for strategy_id, snapshot in incoming.items():
                current = self._strategies.get(strategy_id)
                if current is None or current.runner_id == payload.runner_id:
                    continue
                owner = self._runners.get(current.runner_id)
                owner_available = owner is not None and self._runner_available(
                    owner,
                    received_at,
                )
                if owner_available and current.snapshot.status != "stopped":
                    self._conflicts[strategy_id] = {
                        current.runner_id,
                        payload.runner_id,
                    }
                    raise LiveSnapshotConflict(
                        f"Strategy {strategy_id!r} is already reported by "
                        f"runner {current.runner_id!r}"
                    )

            previous_ids = runner.strategy_ids if runner is not None else set()
            removed_ids = previous_ids - set(incoming)
            for strategy_id in removed_ids:
                current = self._strategies.get(strategy_id)
                if current is not None and current.runner_id == payload.runner_id:
                    self._strategies.pop(strategy_id, None)
                    self._conflicts.pop(strategy_id, None)

            for strategy_id, snapshot in incoming.items():
                previous = self._strategies.get(strategy_id)
                if previous is not None and previous.runner_id != payload.runner_id:
                    previous_runner = self._runners.get(previous.runner_id)
                    if previous_runner is not None:
                        previous_runner.strategy_ids.discard(strategy_id)
                self._strategies[strategy_id] = StrategyState(
                    runner_id=payload.runner_id,
                    runner_instance_id=payload.runner_instance_id,
                    received_at=received_at,
                    snapshot=snapshot,
                )
                self._conflicts.pop(strategy_id, None)

            self._runners[payload.runner_id] = RunnerState(
                instance_id=payload.runner_instance_id,
                started_at=payload.runner_started_at,
                last_sequence=payload.sequence,
                last_received_at=received_at,
                strategy_ids=set(incoming),
            )
            return {"accepted": True, "strategy_count": len(incoming)}

    def _remove_runner_strategies(self, runner_id: str, instance_id: str) -> None:
        for strategy_id, state in list(self._strategies.items()):
            if state.runner_id == runner_id and state.runner_instance_id == instance_id:
                self._strategies.pop(strategy_id, None)
                self._conflicts.pop(strategy_id, None)

    def _runner_available(self, runner: RunnerState, now: datetime) -> bool:
        return (now - runner.last_received_at).total_seconds() <= self.stale_seconds

    def _public_strategy(self, state: StrategyState, now: datetime) -> dict[str, Any]:
        payload = state.snapshot.model_dump(mode="json")
        runner = self._runners.get(state.runner_id)
        runner_available = runner is not None and self._runner_available(runner, now)
        runner_age_seconds = (
            None
            if runner is None
            else max(0.0, (now - runner.last_received_at).total_seconds())
        )
        conflict_runners = self._conflicts.get(state.snapshot.strategy_id)
        available = (
            state.snapshot.status == "stopped"
            or (runner_available and not conflict_runners)
        )
        payload.update(
            {
                "runner_id": state.runner_id,
                "runner_instance_id": state.runner_instance_id,
                "received_at": state.received_at.isoformat(),
                "runner_sequence": None if runner is None else runner.last_sequence,
                "runner_age_seconds": runner_age_seconds,
                "available": available,
            }
        )
        if not available:
            payload["status"] = None
            payload["availability"] = {
                "account": False,
                "position": False,
                "latest_signal": False,
            }
            payload["errors"] = [
                *payload.get("errors", []),
                {
                    "component": "runner",
                    "message": (
                        "Strategy is reported by multiple runners"
                        if conflict_runners
                        else "Runner snapshot is unavailable"
                    ),
                },
            ]
        return payload

    def strategies(self) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            items = [
                self._public_strategy(state, now)
                for _, state in sorted(self._strategies.items())
            ]
            available_runners = sum(
                self._runner_available(runner, now)
                for runner in self._runners.values()
            )
            return {
                "items": [
                    {
                        "strategy_id": item["strategy_id"],
                        "model_type": item["model_type"],
                        "venue": item["venue"],
                        "symbol": item["symbol"],
                        "interval": item["interval"],
                        "balance": self._balance(item),
                        "margin_utilization": self._margin_utilization(item),
                        "unrealized_pnl": self._unrealized_pnl(item),
                        "status": item["status"],
                        "available": item["available"],
                        "runner_id": item["runner_id"],
                        "received_at": item["received_at"],
                        "runner_sequence": item["runner_sequence"],
                        "runner_age_seconds": item["runner_age_seconds"],
                    }
                    for item in items
                ],
                "runners": {
                    "total": len(self._runners),
                    "available": available_runners,
                    "unavailable": len(self._runners) - available_runners,
                    "stale_after_seconds": self.stale_seconds,
                },
            }

    @staticmethod
    def _balance(item: dict[str, Any]) -> float | None:
        if not item["available"] or not item["availability"]["account"]:
            return None
        account = item["account"]
        return None if account is None else account["balance"]

    @staticmethod
    def _margin_utilization(item: dict[str, Any]) -> float | None:
        if not item["available"] or not item["availability"]["account"]:
            return None
        account = item["account"]
        return None if account is None else account["margin_utilization"]

    @staticmethod
    def _unrealized_pnl(item: dict[str, Any]) -> float | None:
        if not item["available"] or not item["availability"]["position"]:
            return None
        position = item["position"]
        return 0.0 if position is None else position["unrealized_pnl"]

    def strategy(self, strategy_id: str) -> dict[str, Any] | None:
        now = self._now()
        with self._lock:
            state = self._strategies.get(strategy_id)
            return None if state is None else self._public_strategy(state, now)


live_snapshot_store = LiveSnapshotStore()
