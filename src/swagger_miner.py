"""
╔══════════════════════════════════════════════════════════════════════╗
║                    Swagger Miner v2.0                               ║
║   Extracts (endpoint code → OpenAPI JSON) pairs from FastAPI        ║
║   repositories. Auto-discovers repos via GitHub API.                ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
  # Auto-discover and clone top FastAPI repos from GitHub, then mine
  python swagger_miner.py --auto-discover --clone

  # Mine already-cloned repos
  python swagger_miner.py

  # Mine specific repos
  python swagger_miner.py --repos repos/fastapi repos/fastapi-template
"""

import ast
import json
import re
import hashlib
import argparse
import textwrap
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─────────────────────────── CONFIG ──────────────────────────────────

DEFAULT_REPOS = [
    "repos/fastapi",
    "repos/fastapi-template",
    "repos/realworld",
    "repos/fastapi-crud",
    "repos/fastapi-users",
    "repos/fastapi-production",
    "repos/fastapi-microservices",
    "repos/fastapi-postgresql",
    "repos/nicegui",
    "repos/autogpt",
    "repos/langserve",
    "repos/litestar",
    "repos/dispatch",
    "repos/hummingbot",
]

OUTPUT_FILE  = f"swagger_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
REPORT_FILE  = "swagger_mining_report.txt"
REPOS_DIR    = "repos"

# GitHub API search — top FastAPI repos by stars
GITHUB_SEARCH_QUERIES = [
    "fastapi router language:python",
    "fastapi endpoints language:python",
    "fastapi rest api language:python",
]
GITHUB_REPOS_PER_QUERY = 10   # top N repos per query
GITHUB_MIN_STARS       = 50   # skip tiny repos

MIN_ENDPOINT_LINES = 2
MAX_ENDPOINT_LINES = 100

INSTRUCTION = (
    "Generate a complete OpenAPI JSON documentation object for this "
    "FastAPI endpoint. Include summary, description, parameters, "
    "requestBody if applicable, and responses with status codes."
)


# ─────────────────────────── DATA MODELS ─────────────────────────────

@dataclass
class EndpointRecord:
    instruction:   str
    input:         str
    output:        str
    source_file:   str = ""
    function_name: str = ""
    http_method:   str = ""
    path:          str = ""
    sha256:        str = ""

    def to_jsonl_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input":       self.input,
            "output":      self.output,
            "_meta": {
                "source":      self.source_file,
                "function":    self.function_name,
                "http_method": self.http_method,
                "path":        self.path,
            }
        }


@dataclass
class MiningStats:
    files_scanned:      int  = 0
    files_failed:       int  = 0
    endpoints_found:    int  = 0
    rejected_too_short: int  = 0
    rejected_too_long:  int  = 0
    rejected_duplicate: int  = 0
    accepted:           int  = 0
    repos_processed:    list = field(default_factory=list)


# ─────────────────────────── GITHUB DISCOVERY ────────────────────────

def discover_repos_from_github(
    queries:       list[str],
    per_query:     int = 10,
    min_stars:     int = 50,
    github_token:  str | None = None,
) -> list[dict]:
    """
    Discover top FastAPI repositories from GitHub search API.

    Args:
        queries:      List of search query strings.
        per_query:    Number of results to fetch per query.
        min_stars:    Minimum star count to include a repo.
        github_token: Optional GitHub personal access token for
                      higher rate limits (5000 req/hr vs 60).

    Returns:
        Deduplicated list of repo dicts with 'name' and 'clone_url'.
    """
    if not HAS_REQUESTS:
        print("⚠️  requests not installed. Run: pip install requests")
        return []

    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    seen_ids: set[int] = set()
    repos: list[dict] = []

    for query in queries:
        url = "https://api.github.com/search/repositories"
        params = {
            "q":        f"{query}&sort=stars&order=desc",
            "per_page": per_query,
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)

            # handle rate limiting
            if resp.status_code == 403:
                reset_time = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset_time - int(time.time()), 1)
                print(f"  ⏳ GitHub rate limit hit — waiting {wait}s...")
                time.sleep(wait)
                resp = requests.get(url, headers=headers, params=params, timeout=10)

            if resp.status_code != 200:
                print(f"  [WARN] GitHub API error {resp.status_code} for query: {query}")
                continue

            data = resp.json()
            for repo in data.get("items", []):
                if repo["id"] in seen_ids:
                    continue
                if repo["stargazers_count"] < min_stars:
                    continue
                if repo.get("archived"):
                    continue
                seen_ids.add(repo["id"])
                repos.append({
                    "name":       repo["name"],
                    "full_name":  repo["full_name"],
                    "clone_url":  repo["clone_url"],
                    "stars":      repo["stargazers_count"],
                    "language":   repo.get("language", ""),
                })

            # small delay to be polite to GitHub API
            time.sleep(0.5)

        except requests.RequestException as e:
            print(f"  [WARN] Request failed for query '{query}': {e}")

    # sort by stars descending
    repos.sort(key=lambda r: r["stars"], reverse=True)
    return repos


def clone_repos(repos: list[dict], repos_dir: str = REPOS_DIR) -> list[str]:
    """
    Clone a list of GitHub repositories into the repos directory.

    Args:
        repos:     List of repo dicts with 'name' and 'clone_url'.
        repos_dir: Local directory to clone into.

    Returns:
        List of local directory paths that were successfully cloned.
    """
    Path(repos_dir).mkdir(exist_ok=True)
    cloned_paths: list[str] = []

    for repo in repos:
        dest = Path(repos_dir) / repo["name"]

        if dest.exists():
            print(f"  ✅ Already exists: {dest}")
            cloned_paths.append(str(dest))
            continue

        print(f"  📥 Cloning {repo['full_name']} ({repo['stars']:,} ⭐) → {dest}")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth=1", repo["clone_url"], str(dest)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                cloned_paths.append(str(dest))
                print(f"     ✓ Done")
            else:
                print(f"     ✗ Failed: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            print(f"     ✗ Timeout cloning {repo['name']}")
        except Exception as e:
            print(f"     ✗ Error: {e}")

    return cloned_paths


# ─────────────────────────── AST HELPERS ─────────────────────────────

def extract_decorator_info(decorator: ast.expr) -> tuple[str, str] | None:
    """
    Extract HTTP method and path from a FastAPI route decorator.

    Args:
        decorator: AST expression node of the decorator.

    Returns:
        Tuple of (http_method, path) or None if not a route decorator.
    """
    try:
        unparsed = ast.unparse(decorator)
    except Exception:
        return None

    pattern = re.compile(
        r'(\w+)\.(get|post|put|patch|delete|head|options|trace)\s*\(\s*["\']([^"\']+)["\']',
        re.IGNORECASE
    )
    match = pattern.search(unparsed)
    if match:
        return match.group(2).lower(), match.group(3)
    return None


def extract_response_model(decorator: ast.expr) -> str | None:
    """
    Extract response_model from a FastAPI route decorator if present.

    Args:
        decorator: AST expression node of the decorator.

    Returns:
        Response model name string or None.
    """
    try:
        unparsed = ast.unparse(decorator)
        match = re.search(r'response_model\s*=\s*(\w+)', unparsed)
        return match.group(1) if match else None
    except Exception:
        return None


def extract_status_code(decorator: ast.expr) -> int:
    """
    Extract status_code from a FastAPI route decorator.

    Args:
        decorator: AST expression node of the decorator.

    Returns:
        HTTP status code integer, defaults to 200.
    """
    try:
        unparsed = ast.unparse(decorator)
        match = re.search(r'status_code\s*=\s*(\d+)', unparsed)
        return int(match.group(1)) if match else 200
    except Exception:
        return 200


def _python_type_to_openapi(type_str: str) -> str:
    """
    Map a Python type annotation string to an OpenAPI type string.

    Args:
        type_str: Python type annotation as string.

    Returns:
        OpenAPI type string.
    """
    mapping = {
        "int":   "integer",
        "float": "number",
        "bool":  "boolean",
        "str":   "string",
        "list":  "array",
        "dict":  "object",
        "List":  "array",
        "Dict":  "object",
        "UUID":  "string",
    }
    for py_type, oa_type in mapping.items():
        if py_type in type_str:
            return oa_type
    return "string"


def extract_parameters(node: ast.FunctionDef, path: str) -> list[dict]:
    """
    Extract FastAPI path and query parameters from function signature.

    Args:
        node: AST FunctionDef node of the endpoint function.
        path: URL path string to determine path vs query params.

    Returns:
        List of OpenAPI parameter objects.
    """
    SKIP = {
        "db", "request", "response", "background_tasks",
        "current_user", "token", "session", "settings",
        "commons", "pagination",
    }
    path_params = set(re.findall(r'\{(\w+)\}', path))
    params = []

    for arg in node.args.args:
        name = arg.arg
        if name in ("self", "cls") or name in SKIP:
            continue

        type_str = "string"
        if arg.annotation:
            try:
                type_str = ast.unparse(arg.annotation)
            except Exception:
                pass

        openapi_type = _python_type_to_openapi(type_str)
        in_location  = "path" if name in path_params else "query"
        required     = name in path_params

        params.append({
            "name":     name,
            "in":       in_location,
            "required": required,
            "schema":   {"type": openapi_type},
        })

    return params


def build_openapi_object(
    node:           ast.FunctionDef,
    http_method:    str,
    path:           str,
    response_model: str | None,
    status_code:    int,
) -> dict:
    """
    Build an OpenAPI operation object from a FastAPI endpoint node.

    Args:
        node:           AST function node.
        http_method:    HTTP method string.
        path:           URL path string.
        response_model: Pydantic response model name or None.
        status_code:    HTTP success status code.

    Returns:
        OpenAPI operation dict.
    """
    parameters = extract_parameters(node, path)

    if response_model:
        response_content = {
            "application/json": {
                "schema": {"$ref": f"#/components/schemas/{response_model}"}
            }
        }
    else:
        response_content = {
            "application/json": {"schema": {"type": "object"}}
        }

    summary = node.name.replace("_", " ").title()

    operation: dict = {
        "summary":     summary,
        "operationId": node.name,
        "parameters":  parameters,
        "responses": {
            str(status_code): {
                "description": "Successful response",
                "content":     response_content,
            },
            "422": {"description": "Validation Error"},
        }
    }

    if http_method in ("post", "put", "patch"):
        operation["requestBody"] = {
            "required": True,
            "content": {
                "application/json": {"schema": {"type": "object"}}
            }
        }

    parts = [p for p in path.split("/") if p and not p.startswith("{")]
    if parts:
        operation["tags"] = [parts[0].title()]

    return operation


# ─────────────────────────── MAIN MINER ──────────────────────────────

class SwaggerMiner:
    """
    Scans FastAPI repositories and builds endpoint→OpenAPI JSONL dataset.

    Args:
        repo_dirs:   List of repository directory paths to scan.
        output_path: Destination JSONL file path.
    """

    def __init__(self, repo_dirs: list[str], output_path: str) -> None:
        self.repo_dirs   = [Path(d) for d in repo_dirs]
        self.output_path = Path(output_path)
        self.stats       = MiningStats()
        self._seen:      set[str] = set()

    def run(self) -> MiningStats:
        """
        Execute the full mining pipeline.

        Returns:
            MiningStats with counts for every stage.
        """
        records: list[EndpointRecord] = []

        for repo_dir in self.repo_dirs:
            if not repo_dir.exists():
                print(f"  [SKIP] {repo_dir} — not found.")
                continue

            print(f"\n📂  Scanning: {repo_dir.name}")
            self.stats.repos_processed.append(str(repo_dir))
            repo_records = self._process_repo(repo_dir)
            records.extend(repo_records)
            print(f"  → {len(repo_records):,} endpoint records")

        self._write(records)
        self.stats.accepted = len(records)
        return self.stats

    def _process_repo(self, repo_dir: Path) -> list[EndpointRecord]:
        records = []
        for py_file in sorted(repo_dir.rglob("*.py")):
            if any(p.startswith("test") for p in py_file.parts):
                continue
            self.stats.files_scanned += 1
            try:
                records.extend(self._process_file(py_file))
            except Exception:
                self.stats.files_failed += 1
        return records

    def _process_file(self, py_file: Path) -> list[EndpointRecord]:
        """
        Parse a Python file and extract all FastAPI endpoints.

        Args:
            py_file: Path to the Python source file.

        Returns:
            List of EndpointRecord objects.
        """
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            self.stats.files_failed += 1
            return []

        source_lines = source.splitlines()
        records = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                info = extract_decorator_info(decorator)
                if info is None:
                    continue
                http_method, path = info
                self.stats.endpoints_found += 1
                record = self._build_record(
                    node, source_lines, str(py_file),
                    http_method, path, decorator
                )
                if record:
                    records.append(record)
                break

        return records

    def _build_record(
        self,
        node:         ast.FunctionDef,
        source_lines: list[str],
        filepath:     str,
        http_method:  str,
        path:         str,
        decorator:    ast.expr,
    ) -> EndpointRecord | None:
        """
        Build a single EndpointRecord from an AST node.

        Args:
            node:         AST function definition node.
            source_lines: All source lines of the file.
            filepath:     Path to the source file.
            http_method:  HTTP method string.
            path:         URL path string.
            decorator:    AST decorator node.

        Returns:
            EndpointRecord or None if quality gates fail.
        """
        func_lines  = source_lines[node.lineno - 1 : node.end_lineno]
        func_source = textwrap.dedent("\n".join(func_lines)).strip()
        line_count  = len(func_lines)

        if line_count < MIN_ENDPOINT_LINES:
            self.stats.rejected_too_short += 1
            return None
        if line_count > MAX_ENDPOINT_LINES:
            self.stats.rejected_too_long += 1
            return None

        digest = hashlib.sha256(func_source.encode()).hexdigest()
        if digest in self._seen:
            self.stats.rejected_duplicate += 1
            return None
        self._seen.add(digest)

        response_model = extract_response_model(decorator)
        status_code    = extract_status_code(decorator)
        openapi_obj    = build_openapi_object(
            node, http_method, path, response_model, status_code
        )
        full_schema = {
            "path":      path,
            "method":    http_method,
            "operation": openapi_obj,
        }

        return EndpointRecord(
            instruction   = INSTRUCTION,
            input         = func_source,
            output        = json.dumps(full_schema, indent=2, ensure_ascii=False),
            source_file   = filepath,
            function_name = node.name,
            http_method   = http_method,
            path          = path,
            sha256        = digest,
        )

    def _write(self, records: list[EndpointRecord]) -> None:
        """
        Write all records to the output JSONL file.

        Args:
            records: List of EndpointRecord objects.

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
    Print and save a human-readable mining report.

    Args:
        stats:       MiningStats from the mining run.
        output_path: Path to the output JSONL file.

    Returns:
        None
    """
    total_rej = (
        stats.rejected_too_short
        + stats.rejected_too_long
        + stats.rejected_duplicate
    )
    rate = stats.accepted / max(stats.endpoints_found, 1) * 100

    lines = [
        "=" * 60,
        "  Swagger Miner — Report",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "REPOSITORIES:",
        *[f"  • {r}" for r in stats.repos_processed],
        "",
        "FILE STATISTICS:",
        f"  Files scanned     : {stats.files_scanned:>8,}",
        f"  Files failed      : {stats.files_failed:>8,}",
        "",
        "ENDPOINT PIPELINE:",
        f"  Endpoints found   : {stats.endpoints_found:>8,}",
        f"  Rejected (short)  : {stats.rejected_too_short:>8,}",
        f"  Rejected (long)   : {stats.rejected_too_long:>8,}",
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

    try:
        first = json.loads(Path(output_path).open().readline())
        print("\n--- Sample input ---")
        print(first["input"][:400])
        print("\n--- Sample output ---")
        print(first["output"][:400])
    except Exception:
        pass


# ──────────────────────────── CLI ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine FastAPI repos for endpoint→OpenAPI training pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repos", nargs="+", default=DEFAULT_REPOS,
        help="Repo directories to scan"
    )
    parser.add_argument(
        "--out", default=OUTPUT_FILE,
        help="Output JSONL path"
    )
    parser.add_argument(
        "--auto-discover", action="store_true",
        help="Discover top FastAPI repos from GitHub API"
    )
    parser.add_argument(
        "--clone", action="store_true",
        help="Clone discovered repos automatically (use with --auto-discover)"
    )
    parser.add_argument(
        "--github-token", default=None,
        help="GitHub personal access token for higher API rate limits"
    )
    parser.add_argument(
        "--repos-per-query", type=int, default=GITHUB_REPOS_PER_QUERY,
        help=f"Repos to fetch per GitHub search query (default: {GITHUB_REPOS_PER_QUERY})"
    )
    args = parser.parse_args()

    print(__doc__)
    print("─" * 60)

    repo_dirs = args.repos

    # ── auto-discover from GitHub ─────────────────────────────────────
    if args.auto_discover:
        print("🔍  Discovering FastAPI repos from GitHub...\n")
        discovered = discover_repos_from_github(
            queries      = GITHUB_SEARCH_QUERIES,
            per_query    = args.repos_per_query,
            min_stars    = GITHUB_MIN_STARS,
            github_token = args.github_token,
        )

        print(f"\n  Found {len(discovered)} unique repos:\n")
        for r in discovered:
            print(f"  ⭐ {r['stars']:>6,}  {r['full_name']}")

        if args.clone:
            print(f"\n📥  Cloning {len(discovered)} repos...\n")
            cloned = clone_repos(discovered, REPOS_DIR)
            repo_dirs = cloned
        else:
            # just print clone commands
            print("\n  Run these to clone:\n")
            for r in discovered:
                dest = f"{REPOS_DIR}/{r['name']}"
                print(f"  git clone --depth=1 {r['clone_url']} {dest}")
            return

    # ── mine ─────────────────────────────────────────────────────────
    # if no --repos passed and not from auto-discover,
    # automatically scan everything inside the repos/ folder
    if not args.auto_discover and args.repos == DEFAULT_REPOS:
        repos_path = Path(REPOS_DIR)
        if repos_path.exists():
            found = sorted([str(p) for p in repos_path.iterdir() if p.is_dir()])
            if found:
                print(f"📂  Auto-detected {len(found)} repos in ./{REPOS_DIR}/\n")
                for r in found:
                    print(f"    {r}")
                print()
                repo_dirs = found

    existing = [r for r in repo_dirs if Path(r).exists()]
    if not existing:
        print(
            "⚠️  No repo directories found.\n"
            "    Run with --auto-discover --clone to fetch repos automatically.\n"
            "    Or clone manually:\n\n"
            "    git clone https://github.com/tiangolo/fastapi repos/fastapi\n"
        )
        return

    miner = SwaggerMiner(repo_dirs=existing, output_path=args.out)
    stats = miner.run()
    write_report(stats, args.out)


if __name__ == "__main__":
    main()
