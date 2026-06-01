"""
╔══════════════════════════════════════════════════════════════════════╗
║                    SwaggerLM — Swagger UI Server                    ║
║   Generates OpenAPI JSON from Python code and displays it in        ║
║   Swagger UI at http://localhost:8000                               ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
  # Generate docs for a Python file and open Swagger UI
  python serve_swagger.py --input my_api.py

  # Use an already-generated openapi.json
  python serve_swagger.py --json openapi.json

  # Use a specific port
  python serve_swagger.py --input my_api.py --port 8080

REQUIREMENTS:
  pip install requests

  Ollama running locally with your fine-tuned model:
  ollama run swaggerlm
"""

import json
import argparse
import ast
import re
import textwrap
import http.server
import threading
import webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─────────────────────────── CONFIG ──────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "swaggerlm"      # your fine-tuned model name in Ollama
DEFAULT_PORT  = 8000

SYSTEM_PROMPT = (
    "You are an expert API developer. "
    "Given a Python FastAPI endpoint, generate a complete and valid "
    "OpenAPI JSON documentation object. "
    "Include summary, description, parameters with types, "
    "requestBody with detailed schema and properties if applicable, "
    "and responses with status codes and detailed response schema including properties. "
    "Never use empty schemas like {\"type\": \"object\"} — always list the actual properties. "
    "Output ONLY the JSON object, nothing else."
)

INSTRUCTION = (
    "Generate a complete OpenAPI JSON documentation object "
    "for this FastAPI endpoint."
)


# ─────────────────────────── ENDPOINT EXTRACTOR ──────────────────────

def extract_endpoints(source: str) -> list[dict]:
    """
    Extract all FastAPI route endpoints from Python source code.

    Args:
        source: Python source code string to parse.

    Returns:
        List of dicts with 'method', 'path', and 'source' keys.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  ⚠️  Syntax error in file: {e}")
        return []

    source_lines = source.splitlines()
    endpoints    = []

    pattern = re.compile(
        r'(\w+)\.(get|post|put|patch|delete|head|options)\s*\(\s*["\']([^"\']+)["\']',
        re.IGNORECASE
    )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            try:
                unparsed = ast.unparse(decorator)
            except Exception:
                continue
            match = pattern.search(unparsed)
            if not match:
                continue

            method = match.group(2).lower()
            path   = match.group(3)
            lines  = source_lines[node.lineno - 1 : node.end_lineno]
            func_source = textwrap.dedent("\n".join(lines)).strip()

            endpoints.append({
                "method":   method,
                "path":     path,
                "name":     node.name,
                "source":   func_source,
            })
            break

    return endpoints


# ─────────────────────────── OLLAMA CLIENT ───────────────────────────

def generate_openapi(endpoint: dict, model: str) -> dict | None:
    """
    Call the local Ollama model to generate OpenAPI JSON for an endpoint.

    Args:
        endpoint: Dict with 'method', 'path', 'name', 'source' keys.
        model:    Ollama model name to use.

    Returns:
        Parsed OpenAPI operation dict, or None if generation fails.
    """
    if not HAS_REQUESTS:
        print("❌  requests not installed. Run: pip install requests")
        return None

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{INSTRUCTION}\n\n"
        f"```python\n{endpoint['source']}\n```"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature":    0.1,
                    "num_predict":    500,
                    "repeat_penalty": 1.1,
                }
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # clean markdown fences if model added them
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*",     "", raw)
        raw = raw.strip()

        # try to extract JSON object
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"  ⚠️  Invalid JSON for {endpoint['name']}: {e}")
        return None
    except requests.RequestException as e:
        print(f"  ⚠️  Ollama error for {endpoint['name']}: {e}")
        return None


# ─────────────────────────── OPERATION CLEANER ───────────────────

def clean_operation(operation: dict) -> dict:
    """Remove non-standard fields from generated OpenAPI operation."""
    if "operation" in operation and isinstance(operation["operation"], dict):
        operation = operation["operation"]
    for field in ("path", "method", "pathParameters"):
        operation.pop(field, None)
    return operation


# ─────────────────────────── OPENAPI BUILDER ─────────────────────────

def build_openapi_spec(
    endpoints:     list[dict],
    operations:    list[dict | None],
    title:         str = "SwaggerLM Generated API",
    source_file:   str = "",
) -> dict:
    """
    Build a complete OpenAPI 3.0 specification from endpoints and operations.

    Args:
        endpoints:   List of extracted endpoint dicts.
        operations:  List of generated OpenAPI operation dicts (or None).
        title:       API title for the spec info section.
        source_file: Source file name for description.

    Returns:
        Complete OpenAPI 3.0 spec dict ready for JSON serialization.
    """
    paths: dict = {}

    for endpoint, operation in zip(endpoints, operations):
        if operation is None:
            continue
        operation = clean_operation(operation)

        # ensure operationId is set
        if "operationId" not in operation:
            operation["operationId"] = endpoint["name"]

        path   = endpoint["path"]
        method = endpoint["method"]

        if path not in paths:
            paths[path] = {}
        paths[path][method] = operation

    return {
        "openapi": "3.0.0",
        "info": {
            "title":       title,
            "version":     "1.0.0",
            "description": (
                f"API documentation generated by SwaggerLM from `{source_file}`. "
                "Powered by fine-tuned Qwen2.5-Coder-3B."
            ),
        },
        "paths": paths,
    }


# ─────────────────────────── SWAGGER UI HTML ─────────────────────────

def render_swagger_ui(openapi_json: str) -> str:
    """
    Render a complete Swagger UI HTML page with the OpenAPI spec embedded.

    Args:
        openapi_json: JSON string of the OpenAPI specification.

    Returns:
        Complete HTML string for the Swagger UI page.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SwaggerLM — Generated API Docs</title>
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui.min.css">
  <style>
    body {{ margin: 0; background: #fafafa; }}
    .topbar {{ background: #1a1a2e !important; }}
    .topbar-wrapper .link {{ display: flex; align-items: center; gap: 12px; }}
    .topbar-wrapper .link::before {{
      content: '⚡ SwaggerLM';
      color: #e94560;
      font-size: 18px;
      font-weight: bold;
      font-family: monospace;
    }}
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-bundle.min.js"></script>
  <script>
    const spec = {openapi_json};
    SwaggerUIBundle({{
      spec:            spec,
      dom_id:          '#swagger-ui',
      presets:         [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
      layout:          'BaseLayout',
      deepLinking:     true,
      displayOperationId: false,
      defaultModelsExpandDepth: 1,
    }});
  </script>
</body>
</html>"""


# ─────────────────────────── HTTP SERVER ─────────────────────────────

class SwaggerHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler that serves Swagger UI and the OpenAPI spec."""

    openapi_spec: dict = {}

    def do_GET(self):
        if self.path in ("/", "/docs"):
            html = render_swagger_ui(
                json.dumps(self.openapi_spec, ensure_ascii=False)
            )
            self._respond(200, "text/html", html.encode())

        elif self.path == "/openapi.json":
            body = json.dumps(self.openapi_spec, indent=2, ensure_ascii=False)
            self._respond(200, "application/json", body.encode())

        else:
            self._respond(404, "text/plain", b"Not found")

    def _respond(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass   # suppress default access logs


# ──────────────────────────── CLI ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and serve Swagger UI for a Python API file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", default=None,
        help="Python source file or project directory to generate docs for"
    )
    parser.add_argument(
        "--json", default=None,
        help="Use an existing openapi.json file instead of generating"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port to serve on (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--save", default=None,
        help="Save generated openapi.json to this path"
    )
    args = parser.parse_args()

    if not args.input and not args.json:
        parser.print_help()
        raise SystemExit(1)

    print(__doc__)
    print("─" * 60)

    # ── load or generate OpenAPI spec ────────────────────────────────
    if args.json:
        print(f"📄  Loading existing spec: {args.json}")
        spec = json.loads(Path(args.json).read_text(encoding="utf-8"))

    else:
        source_path = Path(args.input)
        if not source_path.exists():
            print(f"❌  Path not found: {source_path}")
            raise SystemExit(1)

        endpoints = []
        if source_path.is_dir():
            py_files = sorted(source_path.rglob("*.py"))
            print(f"📁  Scanning project: {source_path}/  ({len(py_files)} Python files)\n")
            for py_file in py_files:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                file_endpoints = extract_endpoints(source)
                if file_endpoints:
                    rel_path = py_file.relative_to(source_path)
                    print(f"    📄  {rel_path} — {len(file_endpoints)} endpoints")
                    endpoints.extend(file_endpoints)
            print()
        else:
            source = source_path.read_text(encoding="utf-8", errors="ignore")
            print(f"📂  Parsing: {source_path.name}")
            endpoints = extract_endpoints(source)

        if not endpoints:
            print("⚠️  No FastAPI endpoints found.")
            raise SystemExit(1)

        source_name = source_path.name if source_path.is_file() else source_path.name + "/"
        print(f"📂  Total: {len(endpoints)} endpoints\n")

        # check Ollama
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(args.model in m for m in models):
                print(f"⚠️  Model '{args.model}' not found in Ollama.")
                print(f"   Available: {', '.join(models) or 'none'}")
                print(f"   Falling back to basic generation.\n")
                operations = [None] * len(endpoints)
            else:
                # generate OpenAPI for each endpoint
                operations = []
                for ep in endpoints:
                    print(f"  🤖  Generating docs for {ep['method'].upper()} {ep['path']}  [{ep['name']}]")
                    op = generate_openapi(ep, args.model)
                    operations.append(op)
                    status = "✅" if op else "⚠️ "
                    print(f"       {status} {'Valid JSON' if op else 'Used fallback'}")

        except requests.ConnectionError:
            print("⚠️  Ollama not running — using fallback generation.\n")
            operations = [None] * len(endpoints)

        spec = build_openapi_spec(
            endpoints   = endpoints,
            operations  = operations,
            title       = f"{source_name} API",
            source_file = source_name,
        )

        # save if requested
        if args.save:
            Path(args.save).write_text(
                json.dumps(spec, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"\n💾  Saved → {args.save}")

    # ── start server ─────────────────────────────────────────────────
    SwaggerHandler.openapi_spec = spec

    server = HTTPServer(("localhost", args.port), SwaggerHandler)
    url    = f"http://localhost:{args.port}"

    print(f"\n🚀  Swagger UI running at {url}")
    print(f"    OpenAPI JSON at {url}/openapi.json")
    print(f"\n    Press Ctrl+C to stop\n")

    # open browser automatically
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋  Server stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
