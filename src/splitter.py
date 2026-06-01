"""
╔══════════════════════════════════════════════════════════════════════╗
║                   Dataset Splitter v1.0                             ║
║         Shuffle → Split → train.jsonl  +  val.jsonl                 ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
  python splitter.py --input merged.jsonl
  python splitter.py --input merged.jsonl --val-ratio 0.05 --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime


# ─────────────────────────── CONFIG ──────────────────────────────────

DEFAULT_VAL_RATIO = 0.05      # 5 % → val,  95 % → train
DEFAULT_SEED      = 42        # reproducible shuffle
REPORT_FILE       = "split_report.txt"


# ──────────────────────────── SPLITTER ───────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records, bad = [], 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  [WARN] {bad} malformed lines skipped.")
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split(
    input_path: Path,
    val_ratio: float,
    seed: int,
    train_path: Path,
    val_path: Path,
) -> None:

    # ── 1. load ───────────────────────────────────────────────────────
    print(f"\n📂  Loading {input_path.name} …")
    records = load_jsonl(input_path)
    total   = len(records)
    print(f"  {total:,} records loaded.")

    # ── 2. shuffle ────────────────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(records)
    print(f"  Shuffled with seed={seed}.")

    # ── 3. split ──────────────────────────────────────────────────────
    val_size   = round(total * val_ratio)
    train_size = total - val_size

    train_records = records[:train_size]
    val_records   = records[train_size:]

    # ── 4. write ──────────────────────────────────────────────────────
    write_jsonl(train_records, train_path)
    print(f"\n✅  train.jsonl → {train_path}  ({train_size:,} records)")

    write_jsonl(val_records, val_path)
    print(f"✅  val.jsonl   → {val_path}   ({val_size:,} records)")

    # ── 5. report ─────────────────────────────────────────────────────
    lines = [
        "=" * 60,
        "  Dataset Split Report",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "INPUT:",
        f"  File       : {input_path}",
        f"  Total rows : {total:,}",
        f"  Seed       : {seed}",
        "",
        "SPLIT:",
        f"  val ratio  : {val_ratio:.1%}",
        f"  train.jsonl: {train_size:,}  ({train_size/total:.1%})",
        f"  val.jsonl  : {val_size:,}   ({val_size/total:.1%})",
        "",
        "OUTPUT:",
        f"  {train_path}",
        f"  {val_path}",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    Path(REPORT_FILE).write_text(report, encoding="utf-8")
    print(f"\n📄  Report saved → {REPORT_FILE}")


# ──────────────────────────── CLI ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle & split a JSONL dataset into train / val.")
    parser.add_argument("--input",     required=True,              help="Source .jsonl file")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Fraction for val set (default: 0.05)")
    parser.add_argument("--seed",      type=int,   default=DEFAULT_SEED,      help="Random seed (default: 42)")
    parser.add_argument("--out-dir",   default=".",                help="Directory for output files (default: current dir)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌  File not found: {input_path}")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split(
        input_path  = input_path,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
        train_path  = out_dir / "train.jsonl",
        val_path    = out_dir / "val.jsonl",
    )


if __name__ == "__main__":
    main()