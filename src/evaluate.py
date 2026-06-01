"""
╔══════════════════════════════════════════════════════════════╗
║              SwaggerLM — Model Evaluation Script             ║
║  Compares base Qwen2.5-Coder-3B vs fine-tuned SwaggerLM     ║
║  on OpenAPI documentation generation quality                 ║
╚══════════════════════════════════════════════════════════════╝

USAGE:
  # Make sure both models are available in Ollama:
  ollama pull qwen2.5-coder:3b        # base model
  ollama list                          # check swaggerlm is there

  # Run evaluation on validation set:
  python evaluate.py --data val.jsonl --samples 50

  # Full evaluation (all 425 val records — takes a while):
  python evaluate.py --data val.jsonl

  # Save results to file:
  python evaluate.py --data val.jsonl --samples 50 --output results.json

REQUIREMENTS:
  pip install requests nltk tabulate
"""

import json
import re
import argparse
import time
import sys
from pathlib import Path
from collections import defaultdict

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


# ─────────────────────────── CONFIG ──────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"

BASE_MODEL = "qwen2.5-coder:3b"
FINETUNED_MODEL = "swaggerlm"

SYSTEM_PROMPT = (
    "You are an expert API developer. "
    "Given a Python FastAPI endpoint, generate a complete and valid "
    "OpenAPI JSON documentation object. "
    "Include summary, description, parameters, requestBody if applicable, "
    "and responses with status codes. "
    "Output ONLY the JSON object, nothing else."
)

INSTRUCTION = (
    "Generate a complete OpenAPI JSON documentation object "
    "for this FastAPI endpoint."
)

# Fields to check for structural completeness
REQUIRED_FIELDS = ["summary", "responses"]
OPTIONAL_FIELDS = ["description", "parameters", "requestBody", "operationId", "tags"]
ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS


# ─────────────────────────── OLLAMA CLIENT ───────────────────

def generate(model: str, code: str) -> str:
    """Send a prompt to Ollama and return raw response text."""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{INSTRUCTION}\n\n"
        f"```python\n{code}\n```<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1500,
                    "repeat_penalty": 1.1,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"    ⚠️  Generation error ({model}): {e}")
        return ""


# ─────────────────────────── PARSING ─────────────────────────

def parse_json(raw: str) -> dict | None:
    """Try to extract and parse JSON from model output."""
    # Remove markdown fences
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def clean_operation(op: dict) -> dict:
    """Unwrap nested 'operation' field and remove non-standard fields."""
    if "operation" in op and isinstance(op["operation"], dict):
        op = op["operation"]
    for field in ("path", "method", "pathParameters"):
        op.pop(field, None)
    return op


# ─────────────────────────── METRICS ─────────────────────────

def compute_json_validity(parsed: dict | None) -> bool:
    """Check if the output is valid JSON."""
    return parsed is not None


def compute_structural_fields(parsed: dict | None) -> dict:
    """Check which OpenAPI fields are present in the output."""
    if parsed is None:
        return {f: False for f in ALL_FIELDS}

    op = clean_operation(parsed)
    result = {}
    for field in ALL_FIELDS:
        result[field] = field in op and op[field] is not None
    return result


def compute_bleu(reference_json: str, generated_json: str) -> float:
    """Compute BLEU score between reference and generated JSON strings."""
    if not HAS_NLTK:
        return -1.0

    ref_tokens = reference_json.split()
    gen_tokens = generated_json.split()

    if not ref_tokens or not gen_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    try:
        return sentence_bleu(
            [ref_tokens],
            gen_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )
    except Exception:
        return 0.0


def compute_openapi_validity(parsed: dict | None) -> bool:
    """Check if parsed JSON looks like a valid OpenAPI operation."""
    if parsed is None:
        return False

    op = clean_operation(parsed)

    # Must have at least 'responses' or 'summary' to be a valid operation
    has_responses = "responses" in op and isinstance(op.get("responses"), dict)
    has_summary = "summary" in op and isinstance(op.get("summary"), str)

    return has_responses or has_summary


# ─────────────────────────── EVALUATION ──────────────────────

def evaluate_sample(record: dict, models: list[str]) -> dict:
    """Evaluate one sample across all models."""
    code = record.get("input", "")
    reference = record.get("output", "")

    results = {}

    for model in models:
        start = time.time()
        raw = generate(model, code)
        elapsed = time.time() - start

        parsed = parse_json(raw)
        fields = compute_structural_fields(parsed)

        # Compute BLEU against reference
        if parsed is not None:
            gen_str = json.dumps(clean_operation(parsed), sort_keys=True)
        else:
            gen_str = raw
        bleu = compute_bleu(reference, gen_str)

        results[model] = {
            "raw_output": raw[:500],  # truncate for storage
            "json_valid": compute_json_validity(parsed),
            "openapi_valid": compute_openapi_validity(parsed),
            "fields": fields,
            "bleu": bleu,
            "time_seconds": round(elapsed, 2),
        }

    return results


def aggregate_results(all_results: list[dict], models: list[str]) -> dict:
    """Aggregate metrics across all samples."""
    agg = {}
    n = len(all_results)

    for model in models:
        model_results = [r[model] for r in all_results if model in r]
        count = len(model_results)
        if count == 0:
            continue

        json_valid = sum(1 for r in model_results if r["json_valid"])
        openapi_valid = sum(1 for r in model_results if r["openapi_valid"])

        field_counts = defaultdict(int)
        for r in model_results:
            for field, present in r["fields"].items():
                if present:
                    field_counts[field] += 1

        bleu_scores = [r["bleu"] for r in model_results if r["bleu"] >= 0]
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

        avg_time = sum(r["time_seconds"] for r in model_results) / count

        # Structural completeness = average fraction of all fields present
        structural_scores = []
        for r in model_results:
            present = sum(1 for v in r["fields"].values() if v)
            structural_scores.append(present / len(ALL_FIELDS))
        avg_structural = sum(structural_scores) / len(structural_scores)

        agg[model] = {
            "total": count,
            "json_valid": json_valid,
            "json_valid_pct": round(100 * json_valid / count, 1),
            "openapi_valid": openapi_valid,
            "openapi_valid_pct": round(100 * openapi_valid / count, 1),
            "field_rates": {
                f: round(100 * field_counts[f] / count, 1) for f in ALL_FIELDS
            },
            "structural_completeness_pct": round(100 * avg_structural, 1),
            "avg_bleu": round(avg_bleu, 4),
            "avg_time_seconds": round(avg_time, 2),
        }

    return agg


# ─────────────────────────── DISPLAY ─────────────────────────

def print_results(agg: dict, models: list[str]) -> None:
    """Print formatted comparison table."""
    print("\n" + "═" * 60)
    print("  SwaggerLM — Evaluation Results")
    print("═" * 60)

    # Main metrics table
    headers = ["Metric"] + [m for m in models]
    rows = []

    metrics = [
        ("Samples", "total"),
        ("JSON validity (%)", "json_valid_pct"),
        ("OpenAPI validity (%)", "openapi_valid_pct"),
        ("Structural completeness (%)", "structural_completeness_pct"),
        ("Average BLEU", "avg_bleu"),
        ("Avg time (sec)", "avg_time_seconds"),
    ]

    for label, key in metrics:
        row = [label]
        for model in models:
            if model in agg:
                row.append(agg[model].get(key, "—"))
            else:
                row.append("—")
        rows.append(row)

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print(f"\n{'Metric':<35} {models[0]:<15} {models[1]:<15}")
        print("-" * 65)
        for row in rows:
            print(f"{row[0]:<35} {str(row[1]):<15} {str(row[2]):<15}")

    # Field-level breakdown
    print(f"\n{'Field':<25} ", end="")
    for m in models:
        if m in agg:
            print(f"{m:<15} ", end="")
    print()
    print("-" * 55)

    for field in ALL_FIELDS:
        print(f"{field:<25} ", end="")
        for m in models:
            if m in agg:
                pct = agg[m]["field_rates"].get(field, 0)
                print(f"{pct:>6.1f}%         ", end="")
        print()

    print()


# ─────────────────────────── MAIN ────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SwaggerLM vs base model on OpenAPI generation.",
    )
    parser.add_argument("--data", required=True, help="Path to val.jsonl")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help=f"Base model name (default: {BASE_MODEL})")
    parser.add_argument("--finetuned-model", default=FINETUNED_MODEL,
                        help=f"Fine-tuned model name (default: {FINETUNED_MODEL})")
    args = parser.parse_args()

    if not HAS_REQUESTS:
        print("❌ requests not installed. Run: pip install requests")
        sys.exit(1)

    if not HAS_NLTK:
        print("⚠️  nltk not installed — BLEU scores will be skipped.")
        print("   Install with: pip install nltk")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        sys.exit(1)

    # Load data
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.samples:
        records = records[:args.samples]

    models = [args.base_model, args.finetuned_model]
    print(f"\n📊 Evaluating {len(records)} samples")
    print(f"   Models: {' vs '.join(models)}")
    print(f"   Data: {data_path}\n")

    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
        for model in models:
            found = any(model in m for m in available)
            status = "✅" if found else "❌ NOT FOUND"
            print(f"   {model}: {status}")
        print()
    except requests.ConnectionError:
        print("❌ Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    # Evaluate
    all_results = []
    for i, record in enumerate(records):
        endpoint_name = record.get("input", "")[:60].replace("\n", " ")
        print(f"  [{i+1}/{len(records)}] {endpoint_name}...")

        result = evaluate_sample(record, models)
        all_results.append(result)

        # Show inline progress
        for model in models:
            r = result[model]
            valid = "✅" if r["json_valid"] else "❌"
            print(f"    {model}: {valid} JSON | "
                  f"{sum(v for v in r['fields'].values())}/{len(ALL_FIELDS)} fields | "
                  f"{r['time_seconds']}s")

    # Aggregate and display
    agg = aggregate_results(all_results, models)
    print_results(agg, models)

    # Save results
    if args.output:
        output = {
            "config": {
                "base_model": args.base_model,
                "finetuned_model": args.finetuned_model,
                "data_file": str(data_path),
                "num_samples": len(records),
            },
            "aggregated": agg,
            "per_sample": all_results,
        }
        Path(args.output).write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
