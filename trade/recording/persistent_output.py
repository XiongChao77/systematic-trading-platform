"""Import historical sessions into stable runner journals before startup."""
from __future__ import annotations

import csv
import fcntl
import json
import mmap
import os
import re
from pathlib import Path

from trade.recording.execution_trace import LiveExecutionTraceRecorder, SCHEMA_VERSION
from trade.recording.persistent_csv import AppendCsv
from trade.recording.prediction_trace import FeedPredictionTraceWriter

SESSION = re.compile(r"\d{8}T\d{12}Z")


class OutputLease:
    """Prevent concurrent processes from mutating the same runner output."""

    def __init__(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.handle = (root / ".writer.lock").open("a")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError(f"Runner output is already in use: {root}") from None

    def close(self):
        if not self.handle.closed:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sources(root):
    sources = {}
    for session in sorted(root.iterdir()):
        if session.is_dir() and SESSION.fullmatch(session.name):
            for pattern in ("execution_traces/*.csv", "prediction_traces/*.csv", "logs/**/*.log", "logs/**/*.log.[0-9]*"):
                for path in sorted(session.glob(pattern)):
                    if path.is_file() and (path.suffix == ".csv" or re.search(r"\.log(?:\.\d+)?$", path.name)):
                        sources[str(path.relative_to(root))] = (path, {})
    # The previous identity migration preserved open file descriptors here.
    # Import their final tails too, even when the old process has not restarted.
    backups = root.parent / "migration_backups"
    for backup in sorted(backups.glob("*")):
        mapping_path = backup / "identity_map.json"
        if not mapping_path.exists():
            continue
        mapping = json.loads(mapping_path.read_text())
        active = backup / "active_writer_sources"
        for path in sorted(active.glob("*/*/*")):
            relative = str(path.relative_to(active))
            parts = Path(relative).parts
            if (path.is_file() and len(parts) == 3 and SESSION.fullmatch(parts[0])
                    and parts[1] in {"logs", "execution_traces", "prediction_traces"}
                    and (path.suffix == ".csv" or path.name.endswith(".log"))):
                sources[relative] = (path, mapping)
    return sources


def _complete_log_end(handle, offset, end):
    """Find the last newline without discarding an arbitrarily long partial line."""
    cursor = end
    while cursor > offset:
        start = max(offset, cursor - 1024 * 1024)
        handle.seek(start)
        block = handle.read(cursor - start)
        newline = block.rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        cursor = start
    return offset


def _same_log_prefix(first, second, length):
    if not first.exists() or min(first.stat().st_size, second.stat().st_size) < length:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        remaining = length
        while remaining:
            size = min(remaining, 1024 * 1024)
            if left.read(size) != right.read(size):
                return False
            remaining -= size
    return True


def _included_log_snapshot(source, destination):
    """Verify a replacement snapshot byte-for-byte before resetting its offset.

    Identity-migration backups may have been removed after importing their tails.
    Their raw byte offsets cannot be used for the remaining transformed snapshot.
    """
    size = source.stat().st_size
    if not size:
        return True
    if not destination.exists() or destination.stat().st_size < size:
        return False
    with source.open("rb") as original, destination.open("rb") as target:
        prefix = original.read(min(4096, size))
        with mmap.mmap(target.fileno(), 0, access=mmap.ACCESS_READ) as merged:
            position = merged.find(prefix)
            while position >= 0:
                original.seek(0)
                checked = 0
                while checked < size:
                    block = original.read(min(1024 * 1024, size - checked))
                    if merged[position + checked:position + checked + len(block)] != block:
                        break
                    checked += len(block)
                if checked == size:
                    return True
                position = merged.find(prefix, position + 1)
    return False


def _rows(path, mapping):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = []
        for key in reader.fieldnames or []:
            if key == "strategy_id":
                key = "instance_id"
            elif "__" in key:
                owner, suffix = key.rsplit("__", 1)
                key = mapping.get(owner, owner) + "__" + suffix
            fields.append(key)
        for original in reader:
            # An old running process may currently be writing its last row.
            if None in original or any(value is None for value in original.values()):
                continue
            row = dict(zip(fields, original.values()))
            if "instance_id" in row:
                row["instance_id"] = mapping.get(row["instance_id"], row["instance_id"])
            if "schema_version" in row:
                row["schema_version"] = SCHEMA_VERSION
                row.setdefault("trader_login", "")
            yield row


def prepare_output(root):
    """Idempotently import changed historical sources; caller holds OutputLease.

    CSV imports replay through persistent deduplication. Log offsets are committed
    with a rollback journal so interrupted imports cannot duplicate log blocks.
    Historical source files are never changed or removed.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / ".history_import.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    pending_path = root / ".log_import_pending.json"
    log_path = root / "logs/session.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if pending_path.exists():
        pending = json.loads(pending_path.read_text())
        if state.get(pending["source"], {}).get("offset") != pending["end"]:
            with log_path.open("r+b") as handle:
                handle.truncate(pending["before"])
        pending_path.unlink()
    writers = {}
    prediction_writers = {}
    imported = 0
    recovered_snapshots = 0
    imported_log_bytes = 0
    pending_log_bytes = 0
    try:
        for relative, (source, mapping) in _sources(root).items():
            stat = source.stat()
            fingerprint = [str(source), stat.st_size, stat.st_mtime_ns]
            previous = state.get(relative, {})
            if previous.get("fingerprint") == fingerprint:
                if "/logs/" in relative:
                    pending_log_bytes += stat.st_size - previous.get("offset", 0)
                continue
            if "/execution_traces/" in relative:
                kind = source.stem.rsplit("_", 1)[-1]
                fields = {"executions": LiveExecutionTraceRecorder.EXECUTION_FIELDS,
                          "orders": LiveExecutionTraceRecorder.CHILD_ORDER_FIELDS,
                          "fills": LiveExecutionTraceRecorder.FILL_FIELDS,
                          "events": LiveExecutionTraceRecorder.EVENT_FIELDS}.get(kind)
                if fields is None:
                    continue
                if kind not in writers:
                    writers[kind] = AppendCsv(root / "execution_traces" / f"{kind}.csv", fields)
                for row in _rows(source, mapping):
                    writers[kind].writerow(row)
                writers[kind].flush()
                os.fsync(writers[kind].handle.fileno())
            elif "/prediction_traces/" in relative:
                name = re.sub(r"_\d{8}T\d{12}Z$", "", source.stem) + ".csv"
                records = list(_rows(source, mapping))
                if not records:
                    continue
                ids = list(dict.fromkeys(key.rsplit("__", 1)[0] for key in records[0] if "__" in key))
                if name in prediction_writers:
                    prediction_writers.pop(name).close()
                writer = FeedPredictionTraceWriter(str(root / "prediction_traces" / name), ids)
                prediction_writers[name] = writer
                writer._write_records(records)
                os.fsync(writer._handle.fileno())
            else:
                offset = previous.get("offset", 0)
                previous_path = previous.get("fingerprint", [str(source)])[0]
                if (previous_path != str(source) and previous
                        and not _same_log_prefix(Path(previous_path), source, offset)):
                    if _included_log_snapshot(source, log_path):
                        state[relative] = {"fingerprint": fingerprint, "offset": stat.st_size,
                                           "verified_snapshot_of": previous_path}
                        _save(state_path, state)
                        recovered_snapshots += 1
                        continue
                    # Do not reuse offsets from another representation of a log.
                    raise ValueError(f"Replacement historical log is not a verified merged snapshot: {source}")
                if stat.st_size < offset:
                    raise ValueError(f"Historical log was truncated: {source}")
                with log_path.open("ab") as target, source.open("rb") as handle:
                    handle.seek(offset)
                    end = stat.st_size
                    end = _complete_log_end(handle, offset, end)
                    pending_log_bytes += stat.st_size - end
                    if end == offset:
                        state[relative] = {"fingerprint": fingerprint, "offset": offset}
                        _save(state_path, state)
                        continue
                    _save(pending_path, {"source": relative, "end": end, "before": target.tell()})
                    handle.seek(offset)
                    replacements = [(f"{prefix}{old}".encode(), f"{new_prefix}{new}".encode())
                                    for old, new in sorted(mapping.items(), key=lambda item: -len(item[0]))
                                    for prefix, new_prefix in (("strategy_id=", "instance_id="), ("strategy=", "strategy="), ("id=", "id="))]
                    if replacements:
                        while handle.tell() < end:
                            line = handle.readline(end - handle.tell())
                            for old, new in replacements:
                                line = line.replace(old, new)
                            target.write(line)
                    else:
                        while handle.tell() < end:
                            target.write(handle.read(min(1024 * 1024, end - handle.tell())))
                    target.flush()
                    os.fsync(target.fileno())
                state[relative] = {"fingerprint": fingerprint, "offset": end}
                _save(state_path, state)
                pending_path.unlink()
                imported += 1
                imported_log_bytes += end - offset
                continue
            state[relative] = {"fingerprint": fingerprint}
            _save(state_path, state)
            imported += 1
    finally:
        for writer in (*writers.values(), *prediction_writers.values()):
            writer.close()
    return {"imported_sources": imported, "tracked_sources": len(state),
            "recovered_log_snapshots": recovered_snapshots,
            "imported_log_bytes": imported_log_bytes,
            "pending_log_bytes": pending_log_bytes}


def main():
    """Import historical tails without constructing a runner or venue client."""
    import argparse

    parser = argparse.ArgumentParser(description="Import historical runner files into persistent journals.")
    parser.add_argument("output_dir", help="Persistent runner output directory")
    arguments = parser.parse_args()
    with OutputLease(arguments.output_dir):
        print(json.dumps(prepare_output(arguments.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
