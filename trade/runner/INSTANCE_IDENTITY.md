# Live instance identity

The `strategy` section in `LiveTrading/live_config.json` is a list of deployment
objects. There are no manually assigned strategy IDs. Each object supplies its
model `hash` and venue configuration. cTrader specifies `trader_login` only;
its `account_id` must not be saved in the live configuration. Other venues retain
configured `account_id`. Symbol and interval come from the hash's report entry.
Runtime and offline identities use:

```
ctrader|trader_login|SYMBOL|interval|hash
other_venue|account_id|SYMBOL|interval|hash
```

Components are URL-escaped separately. Venue is lowercase and symbol uppercase.
Quote instance IDs when passing them to shell commands because they contain `|`.
Environment is not part of the identity. cTrader login identifies a deployment
before any API connection, so offline and live selectors are identical. Different
logins can deploy the same model. Duplicate identities are rejected before model
loading and checked again during venue initialization.

The cTrader API account ID is resolved in memory for authentication and routing.
It is checked against the constructed venue but never replaces the login in a
strategy identity. Monitoring and execution records use `trader_login`; their
`account_id` field is empty for cTrader. Other venues continue using `account_id`.
Existing opaque execution and event IDs are preserved for reconciliation.

Execution CSVs use schema version 4, adding `trader_login`. Stop the runner before
migrating existing output; the migration acquires its writer lock:

```bash
python -m trade.runner.migrate_ctrader_logins ../quant_output/live_runner/runner-main \
  --account 48411225=5051362 --account 48424623=7600560
```

Use `--dry-run` to validate first. Migration updates initial-state keys, ownership
mapping keys, prediction columns, trace identities and log identity text. Baseline
values and original exchange ownership strings are preserved. It does not create
backups. Restart the updated runner and monitoring backend together with the
updated frontend. Historical source CSVs must also be migrated before importing
them into the schema-4 journals.

The frontend displays `SYMBOL-interval-venue-hash` and a separate account column.
API detail routes, enable/disable operations, prediction columns, monitoring
caches and `initial.json` keys use the generated `instance_id`. Execution CSVs
use schema version 4 and the `instance_id` column. `run_id` still identifies a
process session. Initial balances and start times are not reset by migration.

## Existing exchange orders

`runner-main/order_owners.json` maps migrated instance IDs to their original
exchange ownership strings. The runner uses these strings for venue labels and
client-order prefixes so it can still recognize live positions and protective
orders. These values are opaque exchange metadata, not configuration IDs or
frontend names. New instances use their generated identity as the ownership
string. Preserve this file with the runner's persistent data.

The identity migration leaves exchange order IDs, client order IDs, deal IDs and
execution IDs unchanged. Historical prediction values, fills, balances and
timestamps are unchanged. Original files and the old-to-new mapping are backed
up outside `runner-main` under `live_runner/migration_backups/`.

If an old runner is still writing during migration, its open file descriptors
continue writing to the original inodes retained under `active_writer_sources`
in the backup directory. Stop that old process before finalizing the switch,
then start the updated runner. Startup automatically imports those sources into
the persistent journals, including their final appended data. Do not delete those
sources before the old process has stopped and the final import is complete.

Deploy the updated backend, frontend and runner together. Loading configuration
or running the offline checks does not place orders or restart a live process.
