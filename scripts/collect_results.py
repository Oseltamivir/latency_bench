#!/usr/bin/env python3
"""
Utility to collect benchmarking results:

- summary: Generate Markdown tables for GitHub Job Summary
- append-history: Append run results into history JSON files for Pages

This consolidates logic that was previously embedded in workflow YAML.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple, Optional


def _to_ms(x):
    # Match existing behavior: convert seconds to ms when values are small (<50)
    return (x * 1000.0 if isinstance(x, (int, float)) and x < 50 else x)


def _read_json_safe(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _find_latency_files(results_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(results_dir, "latency_bs*.json")))


def _parse_bs_from_path(path: str) -> Optional[int]:
    m = re.search(r"bs(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _collect_latency_rows(results_dir: str) -> List[Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]]:
    rows = []
    for f in _find_latency_files(results_dir):
        d = _read_json_safe(f, {})
        bs = _parse_bs_from_path(f)
        if bs is None:
            continue
        avg = _to_ms(d.get("avg_latency"))
        p = d.get("percentiles", {}) or {}
        rows.append(
            (
                bs,
                avg,
                *(_to_ms(p.get(k)) for k in ("10", "25", "50", "75", "90", "99")),
            )
        )
    return sorted(rows, key=lambda t: t[0])


def _find_gsm_result_paths(results_dir: str) -> List[str]:
    # Primary: lm-eval v0.4.x nested results like results/gsm8k_bsX/<sanitized_model>/results_*.json
    nested = sorted(
        glob.glob(os.path.join(results_dir, "gsm8k_bs*/**/results_*.json"), recursive=True)
    )
    if nested:
        return nested

    # Fallbacks: older/flat naming inside each gsm8k_bs* directory
    paths: List[str] = []
    for ddir in sorted(glob.glob(os.path.join(results_dir, "gsm8k_bs*"))):
        cand = [
            os.path.join(ddir, n)
            for n in ("eval_results.json", "results.json", "metrics.json")
        ]
        paths.extend([p for p in cand if os.path.exists(p)])
    return sorted(paths)


def _latest_gsm_by_bs(results_dir: str) -> Dict[int, str]:
    by_bs: Dict[int, Tuple[float, str]] = {}
    for p in _find_gsm_result_paths(results_dir):
        bs = _parse_bs_from_path(p)
        if bs is None:
            continue
        mt = os.path.getmtime(p)
        cur = by_bs.get(bs)
        if cur is None or mt > cur[0]:
            by_bs[bs] = (mt, p)
    # keep only the path
    return {bs: tup[1] for bs, tup in by_bs.items()}


def _extract_gsm_scores(doc: dict) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    # Get the gsm8k metrics dict across formats
    res = doc.get("results", {})
    if isinstance(res, dict) and "gsm8k" in res and isinstance(res["gsm8k"], dict):
        g = res["gsm8k"]
    elif isinstance(res, dict):
        g = res  # older single-task dumps
    else:
        g = {}

    # Prefer explicit strict/flex if present; fall back to plain exact_match/acc
    strict = (
        g.get("exact_match,strict-match")
        or g.get("exact_match_strict")
        or g.get("em_strict")
    )
    flex = g.get("exact_match,flexible-extract") or g.get("exact_match") or g.get("acc")

    # Effective sample count if available
    n_eff = None
    ns = doc.get("n-samples") or doc.get("n_samples")
    if isinstance(ns, dict):
        ge = ns.get("gsm8k")
        if isinstance(ge, dict):
            n_eff = ge.get("effective") or ge.get("n_effective") or ge.get("n")

    return strict, flex, n_eff


def generate_summary_markdown(results_dir: str) -> str:
    out_lines: List[str] = []

    # Latency table
    rows = _collect_latency_rows(results_dir)
    out_lines.append("## vLLM Latency — Llama 1B\n")
    out_lines.append("| Batch | Avg (ms) | P10 | P25 | P50 | P75 | P90 | P99 |")
    out_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bs, avg, p10, p25, p50, p75, p90, p99 in rows:
        fm = lambda x: ("" if x is None else f"{x:.1f}")
        out_lines.append(
            f"| {bs} | {fm(avg)} | {fm(p10)} | {fm(p25)} | {fm(p50)} | {fm(p75)} | {fm(p90)} | {fm(p99)} |"
        )

    # GSM8K summary
    by_bs = _latest_gsm_by_bs(results_dir)
    if by_bs:
        out_lines.append("\n## GSM8K (lm-eval)\n")
        out_lines.append("| Batch | EM (strict) | EM (flex) | n(eff) |")
        out_lines.append("|---:|---:|---:|---:|")
        for bs in sorted(by_bs.keys()):
            d = _read_json_safe(by_bs[bs], {})
            strict, flex, n_eff = _extract_gsm_scores(d)
            fmt = (
                lambda x: ""
                if x is None
                else (f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
            )
            out_lines.append(
                f"| {bs} | {fmt(strict)} | {fmt(flex)} | {fmt(n_eff)} |"
            )
    else:
        out_lines.append("\n_**No gsm8k JSON results found**_\n")

    return "\n".join(out_lines) + "\n"


def append_history(results_dir: str, hist_path: str, gsm_path: str) -> None:
    ts = int(time.time() * 1000)

    # Ensure parent dirs exist
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    os.makedirs(os.path.dirname(gsm_path), exist_ok=True)

    hist = _read_json_safe(hist_path, []) or []
    gsm = _read_json_safe(gsm_path, []) or []

    meta = _read_json_safe(os.path.join(results_dir, "env.json"), {})
    gpu_txt_path = os.path.join(results_dir, "gpu.txt")
    gpu = ""
    if os.path.exists(gpu_txt_path):
        try:
            with open(gpu_txt_path, "r", encoding="utf-8") as f:
                gpu = f.read().strip()
        except Exception:
            gpu = ""

    # Latency history
    for f in _find_latency_files(results_dir):
        d = _read_json_safe(f, {})
        bs = _parse_bs_from_path(f)
        if bs is None:
            continue
        hist.append(
            {
                "ts": ts,
                "batch_size": bs,
                "metrics": d,
                "meta": meta,
                "gpu": gpu,
            }
        )

    # GSM8K history (one per batch size, latest file)
    latest = _latest_gsm_by_bs(results_dir)
    for bs, p in sorted(latest.items()):
        d = _read_json_safe(p, {})
        gsm.append(
            {
                "ts": ts,
                "batch_size": bs,
                "metrics": d,
                "meta": meta,
                "gpu": gpu,
            }
        )

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    with open(gsm_path, "w", encoding="utf-8") as f:
        json.dump(gsm, f, indent=2)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect and summarize benchmark results")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summary", help="Generate Markdown summary")
    p_sum.add_argument("--results-dir", default="results", help="Directory with results")
    p_sum.add_argument(
        "--out",
        default="-",
        help="Output file path or '-' for stdout",
    )

    p_hist = sub.add_parser("append-history", help="Append results to history JSONs")
    p_hist.add_argument("--results-dir", default="results", help="Directory with results")
    p_hist.add_argument("--hist-path", required=True, help="Latency history JSON path")
    p_hist.add_argument("--gsm-path", required=True, help="GSM8K history JSON path")

    args = parser.parse_args(argv)

    if args.cmd == "summary":
        md = generate_summary_markdown(args.results_dir)
        if args.out == "-":
            sys.stdout.write(md)
        else:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(md)
        return 0

    if args.cmd == "append-history":
        append_history(args.results_dir, args.hist_path, args.gsm_path)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

