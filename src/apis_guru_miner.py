"""
╔══════════════════════════════════════════════════════════════════════╗
║                    APIs.guru Miner v1.0                             ║
║   Extracts real OpenAPI operations from APIs.guru directory and     ║
║   generates synthetic FastAPI Python endpoints as input             ║
╚══════════════════════════════════════════════════════════════════════╝

STRATEGY:
  APIs.guru contains 4138 real OpenAPI specs from Stripe, GitHub,
  Google, AWS, etc. For each operation we:
    1. Extract the real OpenAPI operation object (output)
    2. Synthesize a matching FastAPI Python endpoint (input)

  This gives us high-quality OUTPUT (real docs) paired with
  realistic INPUT (synthetic but correct Python code).

USAGE:
  python apis_guru_miner.py
  python apis_guru_miner.py --apis-dir repos/apis-guru/APIs
  python apis_guru_miner.py --limit 2000 --out apis_guru.jsonl
"""

import json
import re
import hashlib
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


# ─────────────────────────── CONFIG ──────────────────────────────────

DEFAULT_APIS_DIR = "repos/apis-guru/APIs"
OUTPUT_FILE      = f"apis_guru_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
REPORT_FILE      = "apis_guru_report.txt"

# Quality gates
MIN_SUMMARY_WORDS    = 2     # operation must have a real summary
MIN_PARAMS_OR_DESC   = 1     # must have params OR description
MAX_PARAMS           = 15    # skip overly complex operations
MAX_FILES_PER_API    = 1     # only take latest version per API

# How many records to collect (None = all)
DEFAULT_LIMIT = None

INSTRUCTION = (
    "Generate a complete OpenAPI JSON documentation object for this "
    "FastAPI endpoint. Include summary, description, parameters, "
    "requestBody if applicable, and responses with status codes."
)

# HTTP methods we care about
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


# ─────────────────────────── DATA MODEL ──────────────────────────────

@dataclass
class OperationRecord:
    instruction:    str
    input:          str     # synthetic FastAPI Python code
    output:         str     # real OpenAPI operation JSON
    source_file:    str = ""
    provider:       str = ""
    path:           str = ""
    method:         str = ""
    operation_id:   str = ""
    sha256:         str = ""

    def to_jsonl_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input":       self.input,
            "output":      self.output,
            "_meta": {
                "source":       self.source_file,
                "provider":     self.provider,
                "path":         self.path,
                "method":       self.method,
                "operation_id": self.operation_id,
            }
        }


@dataclass
class MiningStats:
    files_scanned:       int = 0
    files_failed:        int = 0
    operations_found:    int = 0
    rejected_no_summary: int = 0
    rejected_too_simple: int = 0
    rejected_too_complex: int = 0
    rejected_duplicate:  int = 0
    accepted:            int = 0
    providers:           list = field(default_factory=list)


# ─────────────────────────── YAML LOADER ─────────────────────────────

def load_spec(path: Path) -> dict | None:
    """
    Load an OpenAPI/Swagger YAML or JSON specification file.

    Args:
        path: Path to the spec file.

    Returns:
        Parsed dict of the spec, or None if loading fails.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(content)
        else:
            return json.loads(content)
    except Exception:
        return None


# ─────────────────────────── CODE GENERATOR ──────────────────────────

def _openapi_type_to_python(schema: dict) -> str:
    if not isinstance(schema, dict):
        return "Any"
    t = schema.get("type", "")
    
    # some specs have type as a list: ["string", "null"]
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    
    fmt = schema.get("format", "")
    mapping = {
        "integer": "int",
        "number":  "float",
        "boolean": "bool",
        "string":  "str",
        "array":   "list",
        "object":  "dict",
    }
    if t == "string" and fmt in ("date", "date-time"):
        return "str"
    if t == "string" and fmt == "uuid":
        return "str"
    return mapping.get(t, "Any")


def _sanitize_name(name) -> str:
    # convert to string first — some specs have non-string names
    name = str(name) if name is not None else "param"
    name = re.sub(r'[-.\s]', '_', name)
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    if name and name[0].isdigit():
        name = f"param_{name}"
    return name or "param"


def generate_fastapi_endpoint(
    path:        str,
    method:      str,
    operation:   dict,
    provider:    str,
) -> str:
    """
    Generate a synthetic FastAPI Python endpoint from an OpenAPI operation.

    Args:
        path:      URL path string (e.g. '/v1/charges/{charge_id}').
        method:    HTTP method string (e.g. 'get').
        operation: OpenAPI operation dict.
        provider:  API provider name for router variable naming.

    Returns:
        Python source code string of the FastAPI endpoint.
    """
    operation_id = operation.get("operationId", "")
    summary      = operation.get("summary", "")
    parameters   = operation.get("parameters", [])
    has_body     = "requestBody" in operation

    # generate function name
    if operation_id:
        func_name = re.sub(r'[^a-zA-Z0-9_]', '_', operation_id).lower()
        func_name = re.sub(r'_+', '_', func_name).strip('_')
    else:
        # derive from path + method
        path_clean = re.sub(r'[{}/ ]', '_', path).strip('_')
        func_name  = f"{method}_{path_clean}".lower()
        func_name  = re.sub(r'_+', '_', func_name).strip('_')

    # build parameter list
    router_var = "router"
    path_params: list[str] = re.findall(r'\{(\w+)\}', path)
    func_params: list[str] = []

    # add path parameters first
    for param in parameters:
        if not isinstance(param, dict):
            continue
        name     = _sanitize_name(param.get("name", "param"))
        location = param.get("in", "query")
        schema   = param.get("schema", {})
        py_type  = _openapi_type_to_python(schema)
        required = param.get("required", location == "path")

        if location == "path":
            func_params.append(f"{name}: {py_type}")
        elif location == "query":
            if required:
                func_params.append(f"{name}: {py_type}")
            else:
                default = param.get("schema", {}).get("default")
                if default is not None:
                    func_params.append(f"{name}: {py_type} = {repr(default)}")
                else:
                    func_params.append(f"{name}: Optional[{py_type}] = None")

    # add body parameter for POST/PUT/PATCH
    if has_body and method in ("post", "put", "patch"):
        func_params.append("body: dict")

    # add db dependency (common pattern)
    func_params.append("db: Session = Depends(get_db)")

    params_str = ", ".join(func_params)

    # build docstring from operation
    desc = operation.get("description", "").strip()
    if desc:
        # take first sentence only
        first_sentence = desc.split(".")[0].strip()
        # remove HTML tags
        first_sentence = re.sub(r'<[^>]+>', '', first_sentence).strip()
        if len(first_sentence) > 200:
            first_sentence = first_sentence[:200] + "..."
    else:
        first_sentence = summary

    # determine response type hint
    responses     = operation.get("responses", {})
    success_codes = [k for k in responses if str(k).startswith("2")]
    status_code   = success_codes[0] if success_codes else "200"

    # build the endpoint
    decorator  = f'@{router_var}.{method}("{path}"'
    if status_code != "200":
        decorator += f", status_code={status_code}"
    decorator += ")"

    is_async   = method in ("get", "post", "put", "patch", "delete")
    async_kw   = "async " if is_async else ""

    lines = [
        decorator,
        f"{async_kw}def {func_name}({params_str}):",
    ]

    # add body
    if first_sentence:
        lines.append(f'    """')
        lines.append(f'    {first_sentence}')
        lines.append(f'    """')

    # minimal implementation
    if method == "get":
        lines.append(f"    return db.query({_guess_model(path)}).all()")
    elif method == "post":
        lines.append(f"    return db.add({_guess_model(path)}(**body))")
    elif method == "delete":
        lines.append(f"    db.delete(db.query({_guess_model(path)}).first())")
    else:
        lines.append(f"    return {{}}")

    return "\n".join(lines)


def _guess_model(path: str) -> str:
    """
    Guess a Pydantic model name from a URL path.

    Args:
        path: URL path string.

    Returns:
        CamelCase model name string.
    """
    parts = [p for p in path.split("/") if p and not p.startswith("{")]
    if parts:
        last = parts[-1].replace("-", "_").replace(".", "_")
        return last.title().replace("_", "")
    return "Model"


# ─────────────────────────── OPERATION CLEANER ───────────────────────

def clean_operation(operation: dict, path: str, method: str) -> dict:
    """
    Clean and normalize an OpenAPI operation for use as training output.

    Removes vendor extensions, simplifies $ref schemas, and ensures
    the output is a clean self-contained operation object.

    Args:
        operation: Raw OpenAPI operation dict from the spec.
        path:      URL path string.
        method:    HTTP method string.

    Returns:
        Cleaned operation dict ready for JSON serialization.
    """
    clean = {}

    if "summary" in operation:
        clean["summary"] = str(operation["summary"])[:200]

    if "description" in operation:
        desc = re.sub(r'<[^>]+>', '', str(operation["description"]))
        clean["description"] = desc[:500]

    if "operationId" in operation:
        clean["operationId"] = operation["operationId"]

    if "tags" in operation:
        clean["tags"] = operation["tags"][:3]

    # parameters — keep only relevant fields
    if "parameters" in operation:
        clean_params = []
        for param in operation["parameters"][:MAX_PARAMS]:
            if not isinstance(param, dict):
                continue
            if "$ref" in param:
                continue
            cp = {
                "name":     param.get("name", ""),
                "in":       param.get("in", "query"),
                "required": param.get("required", False),
            }
            if "description" in param:
                cp["description"] = str(param["description"])[:200]
            if "schema" in param:
                schema = param["schema"]
                if isinstance(schema, dict) and "$ref" not in schema:
                    cp["schema"] = _simplify_schema(schema)
                else:
                    cp["schema"] = {"type": "string"}
            clean_params.append(cp)
        if clean_params:
            clean["parameters"] = clean_params

    # requestBody
    if "requestBody" in operation:
        rb = operation["requestBody"]
        if isinstance(rb, dict) and "content" in rb:
            clean["requestBody"] = {
                "required": rb.get("required", True),
                "content": {
                    "application/json": {
                        "schema": {"type": "object"}
                    }
                }
            }

    # responses — keep top 3 status codes
    if "responses" in operation:
        clean_responses = {}
        for code, resp in list(operation["responses"].items())[:3]:
            if not isinstance(resp, dict):
                continue
            cr = {"description": str(resp.get("description", "Response"))[:100]}
            if "content" in resp:
                cr["content"] = {"application/json": {"schema": {"type": "object"}}}
            clean_responses[str(code)] = cr
        if clean_responses:
            clean["responses"] = clean_responses

    # add path and method for context
    clean["path"]   = path
    clean["method"] = method

    return clean


def _simplify_schema(schema: dict) -> dict:
    if not isinstance(schema, dict):
        return {"type": "string"}
    simple = {}
    if "type" in schema:
        simple["type"] = schema["type"]
    if "format" in schema:
        simple["format"] = schema["format"]
    if "enum" in schema:
        simple["enum"] = [str(e) for e in schema["enum"][:10]]
    if "default" in schema:
        val = schema["default"]
        # convert non-JSON-serializable types to string
        if isinstance(val, (str, int, float, bool, type(None))):
            simple["default"] = val
        else:
            simple["default"] = str(val)
    if schema.get("type") == "array" and "items" in schema:
        items = schema["items"]
        if isinstance(items, dict) and "$ref" not in items:
            simple["items"] = {"type": items.get("type", "string")}
        else:
            simple["items"] = {"type": "string"}
    return simple or {"type": "string"}


# ─────────────────────────── QUALITY GATES ───────────────────────────

def passes_quality(operation: dict, path: str) -> tuple[bool, str]:
    """
    Check whether an operation meets quality requirements.

    Args:
        operation: OpenAPI operation dict to evaluate.
        path:      URL path string for context.

    Returns:
        Tuple of (passes, rejection_reason). passes is True if the
        operation meets all quality gates.
    """
    summary = operation.get("summary", "").strip()
    desc    = operation.get("description", "").strip()
    params  = operation.get("parameters", [])

    # must have a real summary
    if len(summary.split()) < MIN_SUMMARY_WORDS and not desc:
        return False, "no_summary"

    # must have some content
    has_params  = len(params) >= MIN_PARAMS_OR_DESC
    has_desc    = len(desc) > 20
    has_body    = "requestBody" in operation
    has_resp    = bool(operation.get("responses"))

    if not any([has_params, has_desc, has_body, has_resp]):
        return False, "too_simple"

    # skip if too many parameters
    if len(params) > MAX_PARAMS:
        return False, "too_complex"

    return True, ""


# ─────────────────────────── MAIN MINER ──────────────────────────────

class ApisGuruMiner:
    """
    Extracts OpenAPI operations from APIs.guru directory and generates
    synthetic FastAPI endpoint pairs for fine-tuning.

    Args:
        apis_dir:    Path to the APIs.guru APIs directory.
        output_path: Destination JSONL file path.
        limit:       Maximum number of records to collect.
    """

    def __init__(
        self,
        apis_dir:    str,
        output_path: str,
        limit:       int | None = None,
    ) -> None:
        self.apis_dir    = Path(apis_dir)
        self.output_path = Path(output_path)
        self.limit       = limit
        self.stats       = MiningStats()
        self._seen:      set[str] = set()

    def run(self) -> MiningStats:
        """
        Execute the full mining pipeline.

        Returns:
            MiningStats with counts for every pipeline stage.
        """
        records: list[OperationRecord] = []

        # find all spec files
        spec_files = list(self.apis_dir.rglob("openapi.yaml"))
        spec_files += list(self.apis_dir.rglob("swagger.yaml"))
        spec_files += list(self.apis_dir.rglob("openapi.json"))

        # deduplicate — one per provider (take latest version)
        spec_files = self._deduplicate_specs(spec_files)

        print(f"  Found {len(spec_files):,} unique API specs\n")

        for spec_file in spec_files:
            if self.limit and len(records) >= self.limit:
                break

            self.stats.files_scanned += 1
            provider = self._extract_provider(spec_file)

            spec = load_spec(spec_file)
            if not spec:
                self.stats.files_failed += 1
                continue

            file_records = self._process_spec(spec, spec_file, provider)
            records.extend(file_records)

            if file_records:
                self.stats.providers.append(provider)

        # apply limit
        if self.limit:
            records = records[:self.limit]

        self._write(records)
        self.stats.accepted = len(records)
        return self.stats

    def _deduplicate_specs(self, spec_files: list[Path]) -> list[Path]:
        """
        Keep only the latest version spec per API provider.

        Args:
            spec_files: All discovered spec file paths.

        Returns:
            Deduplicated list keeping one spec per provider directory.
        """
        provider_map: dict[str, Path] = {}
        for f in spec_files:
            # key = parent of version dir = provider dir
            provider_key = str(f.parent.parent)
            if provider_key not in provider_map:
                provider_map[provider_key] = f
            else:
                # keep the one with the "higher" version path
                existing = provider_map[provider_key]
                if str(f) > str(existing):
                    provider_map[provider_key] = f
        return list(provider_map.values())

    def _extract_provider(self, spec_file: Path) -> str:
        """
        Extract provider name from the spec file path.

        Args:
            spec_file: Path to the spec file.

        Returns:
            Provider name string (e.g. 'stripe.com').
        """
        parts = spec_file.parts
        try:
            apis_idx = parts.index("APIs")
            return parts[apis_idx + 1]
        except (ValueError, IndexError):
            return spec_file.parent.parent.name

    def _process_spec(
        self,
        spec:      dict,
        spec_file: Path,
        provider:  str,
    ) -> list[OperationRecord]:
        """
        Extract all valid operations from a single OpenAPI spec.

        Args:
            spec:      Parsed OpenAPI spec dict.
            spec_file: Path to the spec file (for metadata).
            provider:  Provider name string.

        Returns:
            List of OperationRecord objects from this spec.
        """
        paths = spec.get("paths", {})
        if not isinstance(paths, dict):
            return []

        records = []

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            for method in HTTP_METHODS:
                operation = path_item.get(method)
                if not isinstance(operation, dict):
                    continue

                self.stats.operations_found += 1

                # quality check
                passes, reason = passes_quality(operation, path)
                if not passes:
                    if reason == "no_summary":
                        self.stats.rejected_no_summary += 1
                    elif reason == "too_simple":
                        self.stats.rejected_too_simple += 1
                    elif reason == "too_complex":
                        self.stats.rejected_too_complex += 1
                    continue

                # build output — clean real OpenAPI operation
                clean_op = clean_operation(operation, path, method)
                output_str = json.dumps(clean_op, indent=2, ensure_ascii=False)

                # deduplication on output
                digest = hashlib.sha256(output_str.encode()).hexdigest()
                if digest in self._seen:
                    self.stats.rejected_duplicate += 1
                    continue
                self._seen.add(digest)

                # build input — synthetic FastAPI endpoint
                input_code = generate_fastapi_endpoint(
                    path      = path,
                    method    = method,
                    operation = operation,
                    provider  = provider,
                )

                records.append(OperationRecord(
                    instruction  = INSTRUCTION,
                    input        = input_code,
                    output       = output_str,
                    source_file  = str(spec_file),
                    provider     = provider,
                    path         = path,
                    method       = method,
                    operation_id = operation.get("operationId", ""),
                    sha256       = digest,
                ))

        return records

    def _write(self, records: list[OperationRecord]) -> None:
        """
        Write all records to the output JSONL file.

        Args:
            records: List of OperationRecord objects to serialize.

        Returns:
            None
        """
        with self.output_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec.to_jsonl_dict(), ensure_ascii=False) + "\n")
        print(f"\n✅  Dataset → {self.output_path}  ({len(records):,} records)")


# ─────────────────────────── REPORT ──────────────────────────────────

def write_report(stats: MiningStats, output_path: str) -> None:
    """
    Print and save a mining report.

    Args:
        stats:       MiningStats from the mining run.
        output_path: Path to the output JSONL file.

    Returns:
        None
    """
    total_rej = (
        stats.rejected_no_summary
        + stats.rejected_too_simple
        + stats.rejected_too_complex
        + stats.rejected_duplicate
    )
    rate = stats.accepted / max(stats.operations_found, 1) * 100

    lines = [
        "=" * 60,
        "  APIs.guru Miner — Report",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "FILE STATISTICS:",
        f"  Specs scanned     : {stats.files_scanned:>8,}",
        f"  Specs failed      : {stats.files_failed:>8,}",
        f"  Providers         : {len(set(stats.providers)):>8,}",
        "",
        "OPERATION PIPELINE:",
        f"  Operations found  : {stats.operations_found:>8,}",
        f"  Rejected (no sum) : {stats.rejected_no_summary:>8,}",
        f"  Rejected (simple) : {stats.rejected_too_simple:>8,}",
        f"  Rejected (complex): {stats.rejected_too_complex:>8,}",
        f"  Rejected (dup)    : {stats.rejected_duplicate:>8,}",
        f"  Total rejected    : {total_rej:>8,}",
        f"  Accepted          : {stats.accepted:>8,}  ({rate:.1f}%)",
        "",
        "OUTPUT:",
        f"  {output_path}",
        "=" * 60,
    ]

    text = "\n".join(lines)
    print("\n" + text)
    Path(REPORT_FILE).write_text(text, encoding="utf-8")
    print(f"\n📄  Report → {REPORT_FILE}")

    # show sample
    try:
        first = json.loads(Path(output_path).open().readline())
        print("\n--- Sample input (synthetic FastAPI endpoint) ---")
        print(first["input"])
        print("\n--- Sample output (real OpenAPI operation) ---")
        print(first["output"][:500])
    except Exception:
        pass


# ──────────────────────────── CLI ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine APIs.guru for real OpenAPI operations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--apis-dir", default=DEFAULT_APIS_DIR,
        help=f"Path to APIs.guru APIs directory (default: {DEFAULT_APIS_DIR})"
    )
    parser.add_argument(
        "--out", default=OUTPUT_FILE,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help="Maximum records to collect (default: all)"
    )
    args = parser.parse_args()

    apis_dir = Path(args.apis_dir)
    if not apis_dir.exists():
        print(
            f"❌  APIs directory not found: {apis_dir}\n"
            f"    Clone APIs.guru first:\n"
            f"    git clone https://github.com/APIs-guru/openapi-directory repos/apis-guru"
        )
        raise SystemExit(1)

    # check yaml is available
    try:
        import yaml
    except ImportError:
        print("❌  PyYAML not installed. Run: pip install pyyaml")
        raise SystemExit(1)

    print(__doc__)
    print("─" * 60)
    print(f"📂  Mining: {apis_dir}\n")

    miner  = ApisGuruMiner(
        apis_dir    = str(apis_dir),
        output_path = args.out,
        limit       = args.limit,
    )
    stats  = miner.run()
    write_report(stats, args.out)


if __name__ == "__main__":
    main()
