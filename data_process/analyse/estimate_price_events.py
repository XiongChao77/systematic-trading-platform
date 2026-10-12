"""估计 K 线收盘价累计涨跌达到阈值的日均次数（仅使用标准库）。

规则：首个收盘价为基准；close/base - 1 达到正/负阈值时计一次，
并以触发时的实际收盘价重设基准。跨天不重设；单根即使涨跌 3% 也只计一次。
不累加相邻收益率的绝对值，不推测 OHLC 内部路径，因此会遗漏盘中往返。
默认从 CSV 文件名识别 K 线间隔（如 DOGEUSDT_1m.csv），也可用 --interval 指定。
支持固定时长间隔 s、m、h、d、w；月线间隔不支持。
默认按 UTC 自然日统计，包含零事件日；另报仅完整数据日的平均值。
输出每日次数及已完成事件耗时的百分位数（线性插值）。耗时从基准收盘
时刻到触发收盘时刻计算，单位为分钟；末尾未完成事件不纳入百分位数。

用法：py estimate_price_events.py QuantData/Cryptocurrency/binance_public_data/um/DOGEUSDT_1m.csv
      py estimate_price_events.py input.csv --interval 1m --threshold 0.01
      py estimate_price_events.py --threshold 0.01
默认将 JSON 结果保存到当前目录的 output/<输入文件名>_price_events.json；
可用 --output 指定其他路径。
"""

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

DEFAULT_INTERVAL = "15m"
DEFAULT_CSV = Path(__file__).resolve().parent / "Cryptocurrency/binance_public_data/um/DOGEUSDT_15m.csv"


def interval_to_ms(interval):
    """Convert a fixed-duration Binance interval to milliseconds."""
    match = re.fullmatch(r"([1-9]\d*)([smhdw])", interval)
    if not match:
        raise ValueError("interval 必须是正数加 s、m、h、d 或 w，如 1m、4h、1d")
    value, unit = int(match.group(1)), match.group(2)
    scale = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}[unit]
    return value * scale


def infer_interval(path):
    """Read an interval suffix such as ``_1m`` from a CSV filename."""
    match = re.search(r"_([1-9]\d*[smhdwM])$", Path(path).stem)
    return match.group(1) if match else DEFAULT_INTERVAL


def interval_open_is_aligned(open_ms, interval, interval_ms):
    if interval.endswith("w"):
        monday_epoch_ms = 4 * 86_400_000
        return (open_ms - monday_epoch_ms) % interval_ms == 0
    return open_ms % interval_ms == 0


def percentiles(values):
    """使用 (n-1)*p/100 排位线性插值；无样本时返回 None。"""
    ordered = sorted(values)
    result = {}
    for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
        if not ordered:
            result[f"p{p}"] = None
            continue
        rank = (len(ordered) - 1) * p / 100
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        result[f"p{p}"] = ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
    return result


def event_direction(price, base, threshold):
    """Decimal 比较保证恰好达到阈值也计数。"""
    if price >= base * (1 + threshold):
        return "up"
    if price <= base * (1 - threshold):
        return "down"
    return None


def analyze(path, threshold=Decimal("0.01"), utc_offset=0, interval=None):
    if not threshold.is_finite() or not 0 < threshold < 1:
        raise ValueError("threshold 必须介于 0 和 1 之间，如 0.01 表示 1%")
    interval = interval or infer_interval(path)
    interval_ms = interval_to_ms(interval)
    bars_per_day = 86_400_000 // interval_ms if interval_ms <= 86_400_000 and 86_400_000 % interval_ms == 0 else None
    tz = timezone(timedelta(hours=utc_offset))
    daily = {}
    events = []
    base_ms = None
    base = previous_ms = first_ms = last_ms = None
    rows = missing_bars = gap_count = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"open_time_ms_utc", "close_time_ms_utc", "close"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV 缺少字段：{sorted(required - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, 2):
            try:
                open_ms = int(row["open_time_ms_utc"])
                close_ms = int(row["close_time_ms_utc"])
                price = Decimal(row["close"])
                if not price.is_finite() or price <= 0:
                    raise ValueError("收盘价必须为有限正数")
                if not interval_open_is_aligned(open_ms, interval, interval_ms) or close_ms != open_ms + interval_ms - 1:
                    raise ValueError(f"时间戳不是标准的 {interval} K 线")
                if previous_ms is not None and open_ms <= previous_ms:
                    raise ValueError("时间戳重复或未严格升序")
            except (ValueError, ArithmeticError) as exc:
                raise ValueError(f"第 {line} 行无效：{exc}") from exc
            if previous_ms is not None and open_ms - previous_ms > interval_ms:
                gap_count += 1
                missing_bars += (open_ms - previous_ms) // interval_ms - 1
            previous_ms = open_ms
            if first_ms is None:
                first_ms = open_ms
            last_ms = close_ms
            rows += 1
            day = datetime.fromtimestamp(close_ms / 1000, tz).date()
            counts = daily.setdefault(day, Counter(bars=0, up=0, down=0))
            counts["bars"] += 1
            if base is None:
                base = price
                base_ms = close_ms
                continue
            direction = event_direction(price, base, threshold)
            if direction:
                counts[direction] += 1
                events.append(
                    {
                        "start_time": datetime.fromtimestamp(base_ms / 1000, tz).isoformat(),
                        "end_time": datetime.fromtimestamp(close_ms / 1000, tz).isoformat(),
                        "base_price": str(base),
                        "end_price": str(price),
                        "direction": direction,
                        "return": float(price / base - 1),
                        "duration_minutes": (close_ms - base_ms) / 60000,
                    }
                )
                base = price
                base_ms = close_ms

    if rows < 2:
        raise ValueError("至少需要两根 K 线")
    first_day, last_day = min(daily), max(daily)
    records = []
    day = first_day
    while day <= last_day:
        counts = daily.get(day, Counter())
        records.append(
            {
                "date": day.isoformat(),
                "bars": counts["bars"],
                "up": counts["up"],
                "down": counts["down"],
                "total": counts["up"] + counts["down"],
                "complete_day": bars_per_day is not None and counts["bars"] == bars_per_day,
            }
        )
        day += timedelta(days=1)
    up = sum(r["up"] for r in records)
    down = sum(r["down"] for r in records)
    complete = [r for r in records if r["complete_day"]]
    summary = {
        "source": str(Path(path).resolve()),
        "interval": interval,
        "interval_ms": interval_ms,
        "method": "close; reset to observed close after each event; carry base across days",
        "threshold": str(threshold),
        "utc_offset_hours": utc_offset,
        "first_bar_open": datetime.fromtimestamp(first_ms / 1000, tz).isoformat(),
        "last_bar_close": datetime.fromtimestamp(last_ms / 1000, tz).isoformat(),
        "bars": rows,
        "gap_count": gap_count,
        "missing_bars": missing_bars,
        "calendar_days": len(records),
        "complete_days": len(complete),
        "up_events": up,
        "down_events": down,
        "total_events": up + down,
        "average_events_per_calendar_day": (up + down) / len(records),
        "average_up_per_calendar_day": up / len(records),
        "average_down_per_calendar_day": down / len(records),
        "average_events_per_complete_day": (sum(r["total"] for r in complete) / len(complete) if complete else None),
        "percentile_method": "linear interpolation at rank (n-1)*p/100",
        "daily_count_percentiles_calendar_days": percentiles(r["total"] for r in records),
        "daily_count_percentiles_complete_days": percentiles(r["total"] for r in complete),
        "completed_event_duration_minutes_percentiles": percentiles(e["duration_minutes"] for e in events),
        "completed_event_duration_sample_size": len(events),
        "unfinished_event_elapsed_minutes": (last_ms - base_ms) / 60000,
        "limitations": f"{interval} close sampling misses intrabar paths; gaps retain the previous baseline.",
    }
    return {"summary": summary, "daily": records, "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--interval", help="K 线间隔；默认从文件名识别，无法识别时按 15m")
    parser.add_argument("--threshold", type=Decimal, default=Decimal("0.01"))
    parser.add_argument("--utc-offset", type=int, default=0, choices=range(-12, 15), help="自然日时区偏移，默认 UTC；北京时间用 8")
    parser.add_argument("--output", type=Path, help="JSON 输出路径；默认保存到当前目录的 output/ 文件夹")
    args = parser.parse_args()
    output = args.output or Path.cwd() / "output" / f"{args.csv.stem}_price_events.json"
    if output.resolve() == args.csv.resolve():
        parser.error("输出路径不能覆盖输入 CSV")
    try:
        result = analyze(args.csv, args.threshold, args.utc_offset, args.interval)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

"""
python estimate_price_events.py QuantData/Cryptocurrency/binance_public_data/um/DOGEUSDT_1m.csv \
  --threshold 0.01
"""
