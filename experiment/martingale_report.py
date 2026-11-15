"""Serve the local scan report and rebuild its chart data after each scan."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def externalize_report_scripts(report_dir: Path) -> None:
    import re

    page = report_dir / "martingale-breach-report.html"
    content = page.read_text(encoding="utf-8")
    for name, relative in (("library", "_shared/js/echarts.min.js"), ("data", "assets/data.js"), ("charts", "assets/charts.js"), ("backtest", "assets/backtest.js")):
        replacement = f'<script src="./{relative}"></script>'
        pattern = rf'<script data-report-part="{name}">[\s\S]*?</script>|<script src="\./{re.escape(relative)}"></script>'
        content = re.sub(pattern, lambda match: replacement, content)
    temporary = page.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(page)


def validate_scan_sources(bars, sources):
    from data_process import common

    metadata = {}
    for side, directory in sources.items():
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        item = manifest.get("label_metadata", {}).get(side)
        if item is None or item.get("schema") != "martingale_close_cap9_v1":
            raise ValueError("Scan source must use common close-entry capped-nine labels; rerun the scan")
        params = common.MartingaleLabelParams(**item["price_parameters"])
        current = common.martingale_metadata(bars, params)
        if item["main_sha256"] != current["main_sha256"]:
            raise ValueError("Scan market data fingerprint does not match report data")
        metadata[side] = item
    rules = [{k: v for k, v in item.items() if k != "price_parameters"} for item in metadata.values()]
    prices = [{k: v for k, v in item["price_parameters"].items() if k != "side"} for item in metadata.values()]
    if any(rule != rules[0] for rule in rules) or any(price != prices[0] for price in prices):
        raise ValueError("Scan sources use different data or price rules")
    return metadata


def write_report_data(bars, summaries, source: Path, destination: Path) -> None:
    from data_process import common
    from trade.sim import martingale_scan as ms

    times = pd.DatetimeIndex(pd.to_datetime(bars.time, utc=True))
    parameters = common.DOGE_5m
    frame = pd.DataFrame({name: getattr(bars, name) for name in ("open", "high", "low", "close")})
    signals = {}
    for span in (10, 20, 40, 80, 160, 320):
        expected_vol = common.calculate_thresholds(
            frame.copy(), replace(parameters, vol_ewma_span=span)
        )["expected_vol"].to_numpy()
        signals[str(span)] = np.concatenate(([np.nan], expected_vol[:-1])) * np.sqrt(bars.periods_per_year) * 100
    signal = signals[str(parameters.vol_ewma_span)]
    market = pd.DataFrame({"price": bars.close, "volatility": signal}, index=times)
    sampled = market.resample("1h").last().dropna(subset=["price"])
    milliseconds = lambda values: [int(t.timestamp() * 1000) for t in values]
    market_rows = [[t, round(float(p), 8), None if pd.isna(v) else round(float(v), 4)]
                   for t, p, v in zip(milliseconds(sampled.index), sampled.price, sampled.volatility)]
    runs = []
    edges = [0, 1, *range(10, 151, 10), 200, 300, 500, np.inf]
    labels = [f"{low}–{high}%" if np.isfinite(high) else f"≥{low}%"
              for low, high in zip(edges[:-1], edges[1:])]
    market_counts_by_span = {
        span: np.histogram(values[np.isfinite(values)], bins=edges)[0]
        for span, values in signals.items()
    }
    sources = source if isinstance(source, dict) else {run["side"]: source for run in summaries}
    label_metadata = validate_scan_sources(bars, sources)
    days = pd.date_range(times[0].floor("D"), times[-1].floor("D"), freq="D")
    for summary in summaries:
        run = dict(summary)
        cycles = pd.read_csv(Path(sources[run['side']]) / f"cycles_{run['side']}_{run['layers']}additions.csv")
        stops = cycles.loc[cycles.terminal.eq(ms.GRID_STOP)]
        events, daily = {}, {}
        for mode, time_column, price_column in (("entry", "start_time", "entry_price"), ("breach", "breach_time", "breach_price")):
            event_times = pd.DatetimeIndex(pd.to_datetime(stops[time_column], utc=True))
            indices = times.get_indexer(event_times)
            if (indices < 0).any():
                raise ValueError("Event timestamps are missing from market data")
            frame = pd.DataFrame({"time": event_times, "price": stops[price_column].to_numpy(), "volatility": signal[indices]})
            grouped = frame.groupby("time").agg(price=("price", "mean"), volatility=("volatility", "mean"), count=("price", "size"))
            events[mode] = [[t, round(float(p), 8), None if pd.isna(v) else round(float(v), 4), int(n)]
                            for t, p, v, n in zip(milliseconds(grouped.index), grouped.price, grouped.volatility, grouped['count'])]
            counts = pd.Series(1, index=event_times, dtype=int).resample("1D").sum().reindex(days, fill_value=0)
            daily[mode] = list(map(list, zip(milliseconds(days), map(int, counts))))
        buckets_by_span = {}
        start_indices = cycles.start_index.to_numpy(dtype=int)
        stopped = cycles.terminal.eq(ms.GRID_STOP).to_numpy()
        for span, values in signals.items():
            entry_signal = values[start_indices]
            buckets = []
            for low, high, label, market_count in zip(edges[:-1], edges[1:], labels, market_counts_by_span[span]):
                mask = (entry_signal >= low) & (entry_signal < high)
                starts = int(mask.sum())
                breaches = int((mask & stopped).sum())
                buckets.append(dict(label=label, starts=starts, breaches=breaches,
                                    rate=breaches / starts * 100 if starts else None,
                                    market_bars=int(market_count), market_share=float(market_count / len(bars) * 100)))
            buckets_by_span[span] = buckets
        run.update(events=events, daily=daily, buckets_by_span=buckets_by_span,
                   warmup_starts=int(np.isnan(signal[start_indices]).sum()))
        runs.append(run)
    data = {"meta": {"bars": len(bars), "start": str(times[0]), "end": str(times[-1]),
                     "volatilityMethod": "Rogers-Satchell + EWMA", "volatilityEwmaSpan": parameters.vol_ewma_span,
                     "periodsPerYear": bars.periods_per_year,
                     "marketPoints": len(market_rows), "sources": {side: str(path) for side, path in sources.items()},
                     "label_metadata": label_metadata,
                     "marketWarmupBars": int(np.isnan(signal).sum()),
                     "updatedAt": datetime.now(timezone.utc).isoformat()}, "market": market_rows, "runs": runs}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text("window.REPORT_DATA=" + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    temporary.replace(destination)
    externalize_report_scripts(destination.parent.parent)


class ScanJob:
    def __init__(self, report_dir, workers, limit):
        self.report_dir, self.workers, self.limit = report_dir, workers, limit
        self.lock = threading.Lock()
        self.state = {"status": "idle", "message": "Ready"}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def start(self):
        with self.lock:
            if self.state["status"] == "running":
                return False
            self.state = {"status": "running", "message": "Loading market data", "id": uuid.uuid4().hex}
        threading.Thread(target=self.run, daemon=True).start()
        return True

    def run(self):
        command = [sys.executable, str(ROOT / "experiment/martingale_layer_scan.py"), "--side", "both", "--workers", str(self.workers), "--report-data", str(self.report_dir / "assets/data.js")]
        if self.limit:
            command += ["--limit", str(self.limit)]
        try:
            with tempfile.TemporaryFile(mode="w+t") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    log.write(line)
                    with self.lock:
                        self.state["message"] = line.strip()[-500:]
                code = process.wait()
                if code:
                    raise RuntimeError(f"Scan exited with code {code}: {self.snapshot()['message']}")
            with self.lock:
                self.state.update(status="completed", message="Scan completed; charts updated")
        except Exception as error:
            logging.exception("Report scan failed")
            with self.lock:
                self.state.update(status="failed", message=str(error))


class ReportHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def respond(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/backtest":
            self.respond(200, self.server.job.snapshot())
        elif self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/martingale-breach-report.html")
            self.end_headers()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/backtest":
            self.respond(404, {"message": "Unknown endpoint"})
            return
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            self.respond(403, {"message": "Same-origin requests required"})
            return
        started = self.server.job.start()
        self.respond(202 if started else 409, self.server.job.snapshot())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(ReportHandler, directory=str(args.report_dir.resolve())))
    server.job = ScanJob(args.report_dir.resolve(), args.workers, args.limit)
    print(f"Report ready: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
