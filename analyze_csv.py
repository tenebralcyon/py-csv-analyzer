from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MISSING_VALUES = {"", "na", "n/a", "null", "none", "nan"}


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_VALUES


def try_parse_float(value: str) -> Optional[float]:
    s = value.strip().replace(",", "")
    if is_missing(s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class ColumnStats:
    name: str
    total: int
    missing: int
    inferred_type: str  # "numeric" or "text" or "mixed/empty"
    numeric_count: int
    text_count: int
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    stdev: Optional[float] = None
    top_values: Optional[List[Tuple[str, int]]] = None


def summarize_numeric(values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None, None
    mn = min(values)
    mx = max(values)
    mean = sum(values) / len(values)
    if len(values) >= 2:
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        stdev = math.sqrt(var)
    else:
        stdev = 0.0
    return mn, mx, mean, stdev


def analyze_csv(path: Path, top_k: int = 5, sample_rows: int = 0) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header row (could not detect columns).")

        fieldnames = reader.fieldnames
        rows_read = 0

        # Per-column collectors
        numeric_values: Dict[str, List[float]] = {c: [] for c in fieldnames}
        text_values: Dict[str, Counter[str]] = {c: Counter() for c in fieldnames}
        missing_counts: Dict[str, int] = {c: 0 for c in fieldnames}
        numeric_counts: Dict[str, int] = {c: 0 for c in fieldnames}
        text_counts: Dict[str, int] = {c: 0 for c in fieldnames}

        # Optional sampling of first N rows (for huge files)
        for row in reader:
            rows_read += 1
            if sample_rows and rows_read > sample_rows:
                break

            for c in fieldnames:
                raw = (row.get(c) or "").strip()
                if is_missing(raw):
                    missing_counts[c] += 1
                    continue

                num = try_parse_float(raw)
                if num is not None:
                    numeric_values[c].append(num)
                    numeric_counts[c] += 1
                else:
                    # keep text value counts (cap super-long strings)
                    val = raw if len(raw) <= 80 else raw[:77] + "..."
                    text_values[c][val] += 1
                    text_counts[c] += 1

        col_stats: List[ColumnStats] = []
        for c in fieldnames:
            total = rows_read
            missing = missing_counts[c]
            n_num = numeric_counts[c]
            n_txt = text_counts[c]

            if total == 0:
                inferred = "empty"
            elif n_num > 0 and n_txt == 0:
                inferred = "numeric"
            elif n_txt > 0 and n_num == 0:
                inferred = "text"
            elif n_num == 0 and n_txt == 0 and missing == total:
                inferred = "empty"
            else:
                inferred = "mixed"

            mn, mx, mean, stdev = summarize_numeric(numeric_values[c])
            top_vals = None
            if n_txt > 0:
                top_vals = text_values[c].most_common(top_k)

            col_stats.append(
                ColumnStats(
                    name=c,
                    total=total,
                    missing=missing,
                    inferred_type=inferred,
                    numeric_count=n_num,
                    text_count=n_txt,
                    min_val=mn,
                    max_val=mx,
                    mean=mean,
                    stdev=stdev,
                    top_values=top_vals,
                )
            )

    return {
        "file": str(path),
        "rows_analyzed": rows_read,
        "columns": fieldnames,
        "column_stats": col_stats,
    }


def format_report(result: Dict[str, Any], top_k: int) -> str:
    lines: List[str] = []
    lines.append(f"CSV Report")
    lines.append(f"File: {result['file']}")
    lines.append(f"Rows analyzed: {result['rows_analyzed']}")
    lines.append(f"Columns: {len(result['columns'])}")
    lines.append("")

    for cs in result["column_stats"]:
        missing_pct = (cs.missing / cs.total * 100) if cs.total else 0.0
        lines.append(f"Column: {cs.name}")
        lines.append(f"  Type: {cs.inferred_type}")
        lines.append(f"  Missing: {cs.missing}/{cs.total} ({missing_pct:.1f}%)")
        lines.append(f"  Numeric values: {cs.numeric_count}")
        lines.append(f"  Text values: {cs.text_count}")

        if cs.inferred_type in {"numeric", "mixed"} and cs.numeric_count > 0:
            lines.append(f"  Min: {cs.min_val}")
            lines.append(f"  Max: {cs.max_val}")
            lines.append(f"  Mean: {cs.mean}")
            lines.append(f"  Std dev: {cs.stdev}")

        if cs.top_values:
            lines.append(f"  Top {top_k} text values:")
            for v, cnt in cs.top_values:
                lines.append(f"    - {v} ({cnt})")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a CSV and print a simple summary report.")
    parser.add_argument("csv_path", type=str, help="Path to a CSV file")
    parser.add_argument("--top", type=int, default=5, help="Top K text values to show per text column (default: 5)")
    parser.add_argument("--sample-rows", type=int, default=0, help="Analyze only the first N rows (0 = all)")
    parser.add_argument("--out", type=str, default="", help="Optional output file path for the report (.txt)")

    args = parser.parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()

    result = analyze_csv(csv_path, top_k=args.top, sample_rows=args.sample_rows)
    report = format_report(result, top_k=args.top)

    print(report)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.write_text(report, encoding="utf-8")
        print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()