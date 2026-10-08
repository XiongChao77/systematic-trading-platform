"""Convert persisted cTrader instance identities to trader logins while stopped."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from trade.recording.persistent_output import OutputLease
from trade.recording.execution_trace import SCHEMA_VERSION


def convert_identity(value, accounts):
    for account, login in accounts.items():
        value = value.replace(f"ctrader|{account}|", f"ctrader|{login}|")
        value = value.replace(f"ctrader|login%3A{login}|", f"ctrader|{login}|")
    return value


def convert_record(row, accounts):
    row = dict(row)
    if "instance_id" in row:
        row["instance_id"] = convert_identity(row["instance_id"], accounts)
    if "account_id" in row:
        row.setdefault("trader_login", "")
        if row.get("instance_id", "").startswith("ctrader|"):
            login = row["instance_id"].split("|")[1]
            if login not in set(accounts.values()):
                raise ValueError(f"No login mapping for cTrader identity: {row['instance_id']}")
            if row["trader_login"] and row["trader_login"] != login:
                raise ValueError("Conflicting cTrader trader_login")
            row["trader_login"] = login
            row["account_id"] = ""
        row["schema_version"] = SCHEMA_VERSION
    return row


def migrate(root, accounts, *, dry_run=False):
    root = Path(root)
    report = {"json_records": 0, "csv_files": 0, "csv_rows": 0, "log_lines_changed": 0,
              "dry_run": dry_run}
    for filename in ("initial.json", "order_owners.json"):
        path = root / filename
        if not path.exists():
            continue
        original = json.loads(path.read_text())
        converted = {}
        for key, value in original.items():
            new_key = convert_identity(key, accounts)
            if new_key in converted and converted[new_key] != value:
                raise ValueError(f"Conflicting identity records in {path}: {new_key}")
            converted[new_key] = value
            report["json_records"] += int(new_key != key)
        if not dry_run and converted != original:
            temporary = path.with_suffix('.json.login.tmp')
            temporary.write_text(json.dumps(converted, indent=2) + '\n')
            os.replace(temporary, path)
    for path in sorted(root.rglob('*.csv')):
        if not any('execution_traces' in part or part == 'prediction_traces' for part in path.parts):
            continue
        temporary = path.with_suffix('.csv.login.tmp')
        try:
            with path.open(newline='', encoding='utf-8') as reader_handle:
                reader = csv.DictReader(reader_handle)
                old_fields = reader.fieldnames or []
                fields = [convert_identity(field, accounts) for field in old_fields]
                if len(set(fields)) != len(fields):
                    raise ValueError(f"Conflicting converted CSV columns: {path}")
                if 'account_id' in fields and 'trader_login' not in fields:
                    fields.insert(fields.index('account_id') + 1, 'trader_login')
                # Validate every row even during dry runs.
                with temporary.open('w', newline='', encoding='utf-8') as writer_handle:
                    writer = csv.DictWriter(writer_handle, fieldnames=fields)
                    writer.writeheader()
                    for row in reader:
                        if None in row or any(value is None for value in row.values()):
                            raise ValueError(f"Incomplete CSV record: {path}")
                        converted = {convert_identity(key, accounts): value for key, value in row.items()}
                        converted = convert_record(converted, accounts)
                        writer.writerow(converted)
                        report['csv_rows'] += 1
                    writer_handle.flush()
                    os.fsync(writer_handle.fileno())
            report['csv_files'] += 1
            if not dry_run:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    for path in sorted(root.rglob('*.log')):
        temporary = path.with_suffix('.log.login.tmp')
        replacements = [(f'ctrader|{account}|'.encode(), f'ctrader|{login}|'.encode())
                        for account, login in accounts.items()]
        try:
            with path.open('rb') as reader, temporary.open('wb') as writer:
                for line in reader:
                    converted = line
                    for old, new in replacements:
                        converted = converted.replace(old, new)
                    report['log_lines_changed'] += int(converted != line)
                    writer.write(converted)
                writer.flush()
                os.fsync(writer.fileno())
            if not dry_run:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--account', action='append', required=True, metavar='API_ID=TRADER_LOGIN')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    accounts = dict(value.split('=', 1) for value in args.account)
    if not all(key.isdigit() and value.isdigit() and int(key) > 0 and int(value) > 0 for key, value in accounts.items()):
        raise ValueError('Account mappings must contain positive integer IDs and logins')
    if args.dry_run:
        print(json.dumps(migrate(args.root, accounts, dry_run=True), indent=2))
    else:
        with OutputLease(args.root):
            print(json.dumps(migrate(args.root, accounts), indent=2))


if __name__ == '__main__':
    main()
