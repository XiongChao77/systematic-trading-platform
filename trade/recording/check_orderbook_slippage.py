"""Offline unittest suite for execution-cost estimates and recording."""

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from trade.recording.orderbook_common import parse_book
from trade.recording.orderbook_slippage import analyze_files, estimate
from trade.recording.record_orderbook import save_snapshot


class OrderBookSlippageTests(unittest.TestCase):
    def test_sell_walks_multiple_levels(self):
        result = estimate([(100000, .2), (99990, .3), (99980, .8)], "sell", 100000, 100000)
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(result["vwap"], 99987)
        self.assertAlmostEqual(result["slippage_usdt_vs_best"], 13)
        self.assertAlmostEqual(result["slippage_pct_vs_best"], .013)
        self.assertAlmostEqual(result["slippage_bps_vs_best"], 1.3)
        self.assertEqual(result["worst_price"], 99980)

    def test_buy_separates_spread_from_depth_cost(self):
        result = estimate([(101, 1), (103, 2)], "buy", 200, 100)
        self.assertEqual(result["target_quantity"], 2)
        self.assertEqual(result["vwap"], 102)
        self.assertEqual(result["slippage_usdt_vs_best"], 2)
        self.assertEqual(result["slippage_usdt_vs_mid"], 4)
        self.assertEqual(result["slippage_pct_vs_mid"], 2)

    def test_insufficient_depth_does_not_report_full_order_slippage(self):
        result = estimate([(101, 1)], "buy", 200, 100)
        self.assertEqual(result["status"], "insufficient_depth")
        self.assertEqual(result["coverage_pct"], 50)
        self.assertEqual(result["vwap"], 101)
        self.assertIsNone(result["slippage_pct_vs_best"])
        self.assertIsNone(result["slippage_usdt_vs_mid"])

    def test_exact_depth_boundary(self):
        result = estimate([(101, 1)], "buy", 100, 100)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["slippage_usdt_vs_best"], 0)

    def test_invalid_book_rejected(self):
        for payload in (
            {"bids": [[102, 1]], "asks": [[101, 1]]},
            {"bids": [], "asks": [[101, 1]]},
            {"bids": [[99, 1]], "asks": [[101, "nan"]]},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_book(payload)

    def test_record_then_analyze_again_with_different_amounts(self):
        snapshot = {
            "received_at_utc": "2026-09-08T00:00:00.000+00:00",
            "symbol": "DOGEUSDT", "last_update_id": 123,
            "exchange_time_ms": 123456, "request_latency_ms": 10,
            "depth": {"bids": [["99", "10"]], "asks": [["101", "10"]]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for _ in range(2):
                save_snapshot(directory, snapshot)
            self.assertFalse(list(directory.glob("*.csv")))
            with gzip.open(next(directory.glob("*.gz")), "rt") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(records, [snapshot, snapshot])
            source = next(directory.glob("*.gz"))
            original = source.read_bytes()
            output = directory / "analysis.csv"
            self.assertEqual(analyze_files([source], output, [100, 2000]), 8)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[1]["status"], "insufficient_depth")
            self.assertEqual(rows[1]["slippage_pct_vs_best"], "")
            self.assertEqual(analyze_files([source], output, [500]), 4)
            self.assertEqual(source.read_bytes(), original)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[0]["target_notional_usdt"]), 500)

    def test_bad_input_does_not_replace_previous_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "broken.jsonl"
            source.write_text("{broken\n")
            output = directory / "analysis.csv"
            output.write_text("previous result\n")
            with self.assertRaisesRegex(ValueError, "broken.jsonl:1"):
                analyze_files([source], output, [100])
            self.assertEqual(output.read_text(), "previous result\n")
            self.assertFalse(list(directory.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
