from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
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


def normalize_text(value: str) -> str:
    """Normalization for 'near duplicate' detection."""
    v = value.strip().lower()
    v = re.sub(r"\s+", " ", v)  # collapse whitespace
    return v


@dataclass
class ColumnStats:
    name: str
    total: int
    missing: int
    inferred_type: str  # numeric / text / mixed / empty
    numeric_count: int
    text_count: int
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    stdev: Optional[float] = None
    top_values: Optional[List[Tuple[str, int]]] = None
    outliers_zscore: Optional[Dict[str, Any]] = None
    outliers_iqr: Optional[Dict[str, Any]] = None
    text_cleanup_suggestions: Optional[Dict[str, Any]] = None


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


def percentile(sorted_vals: List[float], p: float) -> float:
    """p in [0,1]."""
    if not sorted_vals:
        raise ValueError("Empty list")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def outliers_by_zscore(values: List[float], z_thresh: float = 3.0) -> Dict[str, Any]:
    if len(values) < 3:
        return {"threshold": z_thresh, "count": 0, "examples": []}
    _, _, mean, stdev = summarize_numeric(values)
    if stdev is None or stdev == 0:
        return {"threshold": z_thresh, "count": 0, "examples": []}

    flagged = []
    for x in values:
        z = (x - mean) / stdev
        if abs(z) >= z_thresh:
            flagged.append((x, z))

    flagged.sort(key=lambda t: abs(t[1]), reverse=True)
    return {
        "threshold": z_thresh,
        "count": len(flagged),
        "examples": [{"value": v, "z": z} for v, z in flagged[:5]],
    }


def outliers_by_iqr(values: List[float], k: float = 1.5) -> Dict[str, Any]:
    if len(values) < 4:
        return {"k": k, "count": 0, "bounds": None, "examples": []}
    s = sorted(values)
    q1 = percentile(s, 0.25)
    q3 = percentile(s, 0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr

    flagged = [x for x in values if x < lo or x > hi]
    flagged_sorted = sorted(flagged, key=lambda x: (x < lo, abs(x - (lo if x < lo else hi))), reverse=True)

    return {
        "k": k,
        "bounds": {"low": lo, "high": hi, "q1": q1, "q3": q3, "iqr": iqr},
        "count": len(flagged),
        "examples": flagged_sorted[:5],
    }


def cleanup_suggestions_for_text(counter: Counter[str], min_count: int = 1) -> Dict[str, Any]:
    """
    Suggest merges like:
      'Vancouver', 'vancouver', 'Vancouver ' -> 'vancouver'
    """
    originals = [(k, v) for k, v in counter.items() if v >= min_count]
    groups: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for val, cnt in originals:
        groups[normalize_text(val)].append((val, cnt))

    suggestions = []
    for norm, items in groups.items():
        # "interesting" if multiple spellings OR any item has leading/trailing whitespace or weird spacing/case
        if len(items) >= 2:
            total = sum(c for _, c in items)
            suggestions.append(
                {
                    "recommended": norm,
                    "variants": sorted([{"value": v, "count": c} for v, c in items], key=lambda d: d["count"], reverse=True),
                    "total": total,
                }
            )

    # Also flag values with leading/trailing whitespace (even if no duplicates)
    whitespace_flags = []
    for val, cnt in originals:
        if val != val.strip():
            whitespace_flags.append({"value": val, "count": cnt, "recommended": val.strip()})

    suggestions.sort(key=lambda s: s["total"], reverse=True)
    return {"merge_groups": suggestions[:10], "whitespace_issues": whitespace_flags[:10]}


def iter_rows(path: Path, sample_rows: int = 0) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header row (could not detect columns).")
        rows: List[Dict[str, str]] = []
        for i, row in enumerate(reader, start=1):
            if sample_rows and i > sample_rows:
                break
            rows.append({k: (row.get(k) or "").strip() for k in reader.fieldnames})
    return reader.fieldnames, rows


def analyze_csv(path: Path, top_k: int = 5, sample_rows: int = 0, z_thresh: float = 3.0, iqr_k: float = 1.5) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"File not found: {path}")

    columns, rows = iter_rows(path, sample_rows=sample_rows)
    rows_read = len(rows)

    numeric_values: Dict[str, List[float]] = {c: [] for c in columns}
    text_values: Dict[str, Counter[str]] = {c: Counter() for c in columns}
    missing_counts: Dict[str, int] = {c: 0 for c in columns}
    numeric_counts: Dict[str, int] = {c: 0 for c in columns}
    text_counts: Dict[str, int] = {c: 0 for c in columns}

    for row in rows:
        for c in columns:
            raw = row.get(c, "")
            if is_missing(raw):
                missing_counts[c] += 1
                continue

            num = try_parse_float(raw)
            if num is not None:
                numeric_values[c].append(num)
                numeric_counts[c] += 1
            else:
                val = raw if len(raw) <= 80 else raw[:77] + "..."
                text_values[c][val] += 1
                text_counts[c] += 1

    col_stats: List[ColumnStats] = []
    for c in columns:
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
        top_vals = text_values[c].most_common(top_k) if n_txt > 0 else None

        out_z = outliers_by_zscore(numeric_values[c], z_thresh=z_thresh) if n_num > 0 else None
        out_iqr = outliers_by_iqr(numeric_values[c], k=iqr_k) if n_num > 0 else None
        clean_sug = cleanup_suggestions_for_text(text_values[c]) if n_txt > 0 else None

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
                outliers_zscore=out_z,
                outliers_iqr=out_iqr,
                text_cleanup_suggestions=clean_sug,
            )
        )

    return {
        "file": str(path),
        "rows_analyzed": rows_read,
        "columns": columns,
        "rows": rows,  # keep rows for group-by
        "column_stats": col_stats,
    }


def groupby_summary(rows: List[Dict[str, str]], group_col: str, value_col: str) -> Dict[str, Any]:
    groups: Dict[str, List[float]] = defaultdict(list)
    missing_group = 0
    missing_value = 0

    for r in rows:
        g_raw = (r.get(group_col) or "").strip()
        v_raw = (r.get(value_col) or "").strip()

        if is_missing(g_raw):
            missing_group += 1
            continue
        v = try_parse_float(v_raw)
        if v is None:
            missing_value += 1
            continue

        groups[g_raw].append(v)

    summary = []
    for g, vals in groups.items():
        mn, mx, mean, stdev = summarize_numeric(vals)
        summary.append(
            {
                "group": g,
                "count": len(vals),
                "sum": sum(vals),
                "mean": mean,
                "min": mn,
                "max": mx,
            }
        )

    summary.sort(key=lambda d: d["count"], reverse=True)
    return {
        "group_col": group_col,
        "value_col": value_col,
        "groups": len(summary),
        "missing_group": missing_group,
        "missing_value": missing_value,
        "top_groups": summary[:20],
    }


def format_report(result: Dict[str, Any], top_k: int, group_summary: Optional[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("CSV Report")
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

            if cs.outliers_zscore and cs.outliers_zscore["count"] > 0:
                lines.append(f"  Outliers (z-score >= {cs.outliers_zscore['threshold']}): {cs.outliers_zscore['count']}")
                for ex in cs.outliers_zscore["examples"]:
                    lines.append(f"    - {ex['value']} (z={ex['z']:.2f})")

            if cs.outliers_iqr and cs.outliers_iqr["count"] > 0:
                b = cs.outliers_iqr["bounds"]
                lines.append(f"  Outliers (IQR k={cs.outliers_iqr['k']}): {cs.outliers_iqr['count']}")
                lines.append(f"    Bounds: low={b['low']:.3f}, high={b['high']:.3f} (Q1={b['q1']:.3f}, Q3={b['q3']:.3f})")
                for ex in cs.outliers_iqr["examples"]:
                    lines.append(f"    - {ex}")

        if cs.top_values:
            lines.append(f"  Top {top_k} text values:")
            for v, cnt in cs.top_values:
                lines.append(f"    - {v} ({cnt})")

            # cleanup suggestions
            if cs.text_cleanup_suggestions:
                merges = cs.text_cleanup_suggestions.get("merge_groups", [])
                whites = cs.text_cleanup_suggestions.get("whitespace_issues", [])
                if merges:
                    lines.append("  Cleanup suggestions (possible merges):")
                    for g in merges[:3]:
                        lines.append(f"    -> recommended '{g['recommended']}' from variants:")
                        for vv in g["variants"][:5]:
                            lines.append(f"       - {vv['value']} ({vv['count']})")
                if whites:
                    lines.append("  Cleanup suggestions (trim whitespace):")
                    for w in whites[:3]:
                        lines.append(f"    - '{w['value']}' -> '{w['recommended']}' ({w['count']})")

        lines.append("")

    if group_summary:
        lines.append("Group-by Summary")
        lines.append(f"Group column: {group_summary['group_col']}")
        lines.append(f"Value column: {group_summary['value_col']}")
        lines.append(f"Groups: {group_summary['groups']}")
        lines.append(f"Rows skipped (missing group): {group_summary['missing_group']}")
        lines.append(f"Rows skipped (missing/non-numeric value): {group_summary['missing_value']}")
        lines.append("")
        lines.append("Top groups (by count):")
        for g in group_summary["top_groups"]:
            lines.append(
                f"  - {g['group']}: count={g['count']}, sum={g['sum']:.3f}, mean={g['mean']:.3f}, min={g['min']:.3f}, max={g['max']:.3f}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a CSV and print a summary report.")
    parser.add_argument("csv_path", type=str, help="Path to a CSV file")
    parser.add_argument("--top", type=int, default=5, help="Top K text values to show per text column (default: 5)")
    parser.add_argument("--sample-rows", type=int, default=0, help="Analyze only the first N rows (0 = all)")
    parser.add_argument("--out", type=str, default="", help="Optional output file path for the report (.txt)")

    # New: outlier controls
    parser.add_argument("--z", type=float, default=3.0, help="Z-score threshold for numeric outliers (default: 3.0)")
    parser.add_argument("--iqr-k", type=float, default=1.5, help="IQR multiplier for numeric outliers (default: 1.5)")

    # New: group-by
    parser.add_argument("--group-by", type=str, default="", help="Column name to group by (optional)")
    parser.add_argument("--value", type=str, default="", help="Numeric column name to summarize per group (optional)")

    args = parser.parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()

    result = analyze_csv(
        csv_path,
        top_k=args.top,
        sample_rows=args.sample_rows,
        z_thresh=args.z,
        iqr_k=args.iqr_k,
    )

    group_sum = None
    if args.group_by and args.value:
        group_sum = groupby_summary(result["rows"], args.group_by, args.value)

    report = format_report(result, top_k=args.top, group_summary=group_sum)

    print(report)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.write_text(report, encoding="utf-8")
        print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()
