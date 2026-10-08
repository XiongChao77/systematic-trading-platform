"""Persistent CSV journals with replay deduplication and schema validation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


class AppendCsv:
    """Append facts, ignoring exact replays across process sessions.

    Status changes remain separate facts. Execution summaries are observations;
    consumers should use merge_execution_traces for reconciled lifecycle totals.
    """

    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen = set()
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            with self.path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != self.fieldnames:
                    raise ValueError(f"CSV schema mismatch: {self.path}")
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"Incomplete CSV record: {self.path}")
                    self.seen.add(self._key(row))
        self.handle = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        if not exists:
            self.writer.writeheader()
            self.handle.flush()

    def _key(self, row):
        values = {key: str(row.get(key, "")) for key in self.fieldnames
                  if key not in {"run_id", "schema_version"}}
        # A venue deal is the same fill even if recovered under a new execution ID.
        if "is_aggregate" in values and values.get("deal_id") and values.get("is_aggregate") == "false":
            values = {key: values.get(key, "") for key in
                      ("venue", "account_id", "trader_login", "venue_symbol", "order_id", "deal_id")}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).digest()

    def writerow(self, row):
        key = self._key(row)
        if key not in self.seen:
            self.writer.writerow(row)
            self.seen.add(key)

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
