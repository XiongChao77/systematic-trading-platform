# Order book recording and offline slippage analysis

Run from the repository root. Both tools use only the Python standard library.
The recorder reads public Binance USD-M data; no API keys or orders are involved.
The analyzer operates entirely offline and never modifies source recordings.

## 1. Record raw data

```bash
python3 trade/recording/record_orderbook.py --interval 5 --output orderbook_output
```

Default symbols: DOGEUSDT, XLMUSDT, AAVEUSDT. Default depth: 1000 levels per side.
Use `--symbols DOGEUSDT XLMUSDT` to select symbols. `--samples 1` samples one round;
`--samples 0` (default) runs until Ctrl+C. Requests are sequential, so the three
symbols are not sampled simultaneously. These are periodic REST snapshots, not
an exhaustive feed of every order book update.

Each invocation creates a UTC run directory with daily files per symbol:

- `SYMBOL_YYYY-MM-DD.jsonl.gz`: raw depth response, local receive timestamp,
  exchange transaction timestamp, update ID and request latency.
- `config.json`: recording parameters.
- `errors.jsonl`: failed requests or invalid snapshots, if any.

The recorder does not accept order amounts or calculate slippage. No CSV is
produced. Rate limits trigger cooldowns; permanent HTTP access errors and disk
write failures stop the program. Failed snapshots are never replaced by stale
ones. Long recordings consume disk space; choose depth and interval accordingly.

## 2. Analyze recorded data

```bash
python3 trade/recording/orderbook_slippage.py orderbook_output \
  --amounts 100 500 1000 5000 10000 50000 100000 \
  --output analysis/slippage.csv
```

Input can be a specific run directory, multiple `.jsonl.gz`/`.jsonl` files, or a
parent directory (recursively discovers compressed recordings). Input file paths
are deduplicated. Snapshot observations are retained even if the exchange update
ID repeats, because each observation has a different receive time.

For a second study, reuse the same input with different amounts:

```bash
python3 trade/recording/orderbook_slippage.py orderbook_output \
  --amounts 2000 20000 200000 --output analysis/larger_orders.csv
```

Analysis streams one snapshot at a time, keeping memory bounded. It writes a
single CSV with both buy and sell estimates for every amount. Reusing the same
output path replaces the previous analysis after successful completion; it does
not append duplicate results. Malformed or truncated recordings fail analysis
and leave any prior output intact. Analyze completed files or stable copies of
files; do not read a gzip file while the recorder is appending to it.

## Interpretation

Amounts are hypothetical position **notionals in USDT**, not margin. The same
base quantity is simulated for both directions:

```text
mid_price = (best_bid + best_ask) / 2
target_quantity = target_notional_usdt / mid_price
```

Buys consume asks from low to high; sells consume bids from high to low. Positive
slippage means an adverse cost.

- `slippage_pct_vs_best`: depth cost versus best ask (buy) or best bid (sell).
- `slippage_pct_vs_mid`: depth cost plus half-spread versus midpoint.
- `slippage_usdt_*`: corresponding absolute cost for the simulated quantity.
- `slippage_bps_vs_best`: basis points; 1 bps = 0.01%.
- `vwap`: quantity-weighted fill price; `worst_price`: last consumed price.
- `coverage_pct`: percentage of target quantity covered by visible depth.

When `status=insufficient_depth`, full-order slippage fields are blank. VWAP and
filled quote describe only the visible partial fill. Filter to `status=complete`
when comparing full-order costs. Quantities are theoretical and are not rounded
to exchange filters. Fees, funding, future cancellations, latency, own concurrent
orders and cross-venue basis are excluded. Exchange timestamps require a
synchronized local clock to estimate data age. Visible liquidity does not
promise future fills.

Official API reference:
https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Order-Book

## Offline checks

```bash
python3 -m unittest trade.recording.check_orderbook_slippage -v
```
