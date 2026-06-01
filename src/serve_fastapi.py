"""
╔══════════════════════════════════════════════════════════════════════╗
║                 SwaggerLM — FastAPI Live Demo                       ║
║   Reads a Python file, generates OpenAPI docs via fine-tuned LLM,  ║
║   and proxies requests to a real running API server                 ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
  pip install fastapi uvicorn requests httpx

  # Terminal 1 — start the real API on port 8001
  cd repos/demo && uvicorn main:app --port 8001

  # Terminal 2 — start SwaggerLM docs on port 8000
  python serve_fastapi.py --input repos/demo/main.py --model swaggerlm

RESULT:
  http://localhost:8000/docs        → Swagger UI with AI-generated docs
  http://localhost:8000/openapi.json → Raw OpenAPI JSON
  http://localhost:8000/            → Summary page
  "Try it out" → proxied to real API on port 8001
"""

import ast
import json
import re
import textwrap
import argparse
import webbrowser
import threading
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ─────────────────────────── CONFIG ──────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "swaggerlm"
DEFAULT_PORT  = 8000
REAL_SERVER   = "http://localhost:8001"   # real API to proxy requests to

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


# ─────────────────────────── ENDPOINT EXTRACTOR ──────────────────────

def extract_endpoints(source: str) -> list[dict]:
    """
    Extract all FastAPI route endpoints from Python source code.

    Args:
        source: Python source code string to parse.

    Returns:
        List of dicts with method, path, name and source keys.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  ⚠️  Syntax error: {e}")
        return []

    source_lines = source.splitlines()
    endpoints    = []
    pattern      = re.compile(
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
            source_code = textwrap.dedent("\n".join(lines)).strip()
            endpoints.append({
                "method":  method,
                "path":    path,
                "name":    node.name,
                "source":  source_code,
            })
            break

    return endpoints


# ─────────────────────────── OLLAMA CLIENT ───────────────────────────

def generate_openapi(endpoint: dict, model: str) -> dict | None:
    """
    Call the local Ollama model to generate OpenAPI JSON for one endpoint.

    Args:
        endpoint: Dict with method, path, name, source keys.
        model:    Ollama model name to use for generation.

    Returns:
        Parsed OpenAPI operation dict, or None if generation fails.
    """
    if not HAS_REQUESTS:
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

        # clean markdown fences
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*",     "", raw)
        raw = raw.strip()

        # extract first valid JSON object
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw)

    except Exception as e:
        print(f"  ⚠️  Generation failed for {endpoint['name']}: {e}")
        return None


# ─────────────────────────── OPERATION CLEANER ───────────────────────

def clean_operation(operation: dict) -> dict:
    """
    Clean and normalize a generated OpenAPI operation object.

    Handles cases where the model wraps parameters in an 'operation'
    field or adds non-standard fields like 'path' and 'method'.

    Args:
        operation: Raw OpenAPI operation dict from the model.

    Returns:
        Clean OpenAPI operation dict conforming to the 3.0 standard.
    """
    # unwrap nested 'operation' field if model added it
    if "operation" in operation and isinstance(operation["operation"], dict):
        operation = operation["operation"]

    # remove non-standard fields
    for field in ("path", "method", "pathParameters"):
        operation.pop(field, None)

    return operation


# ─────────────────────────── FASTAPI BUILDER ─────────────────────────

def build_fastapi_app(
    endpoints:   list[dict],
    operations:  list[dict | None],
    source_file: str,
    model:       str,
) -> FastAPI:
    """
    Build a FastAPI application with AI-generated OpenAPI documentation.

    Each endpoint proxies requests to the real API server running on
    REAL_SERVER (default: localhost:8001).

    Args:
        endpoints:   List of extracted endpoint dicts.
        operations:  List of generated OpenAPI operation dicts.
        source_file: Name of the source file being documented.
        model:       Model name used for generation.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title       = f"{Path(source_file).stem} API",
        description = (
            f"API documentation generated by **SwaggerLM** from `{source_file}`.  \n"
            f"Powered by fine-tuned **Qwen2.5-Coder-3B** via QLoRA.  \n\n"
            f"Model: `{model}` · Real API: `{REAL_SERVER}`"
        ),
        version = "1.0.0",
    )

    # build custom OpenAPI spec
    custom_paths = {}

    for endpoint, operation in zip(endpoints, operations):
        path   = endpoint["path"]
        method = endpoint["method"]
        name   = endpoint["name"]

        if operation:
            op = clean_operation(operation)
        else:
            op = {
                "summary":     name.replace("_", " ").title(),
                "operationId": name,
                "responses":   {"200": {"description": "Successful response"}}
            }

        op.setdefault("operationId", name)
        op.setdefault("responses", {"200": {"description": "Successful response"}})

        if path not in custom_paths:
            custom_paths[path] = {}
        custom_paths[path][method] = op

        # register proxy endpoint
        _register_endpoint(app, method, path, name, op)

    # override OpenAPI schema with our generated one
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        app.openapi_schema = {
            "openapi": "3.0.0",
            "info": {
                "title":       app.title,
                "version":     app.version,
                "description": app.description,
            },
            "paths": custom_paths,
        }
        return app.openapi_schema

    app.openapi = custom_openapi

    # root endpoint — summary page
    @app.get("/", include_in_schema=False)
    async def root():
        html = _render_summary(endpoints, operations, source_file, model)
        return HTMLResponse(html)

    return app


def _register_endpoint(
    app:       FastAPI,
    method:    str,
    path:      str,
    name:      str,
    operation: dict,
) -> None:
    """
    Register a proxy endpoint that forwards requests to the real API server.

    Args:
        app:       FastAPI application instance.
        method:    HTTP method string.
        path:      URL path string.
        name:      Function name for the endpoint.
        operation: OpenAPI operation dict for this endpoint.

    Returns:
        None
    """
    path_params      = re.findall(r'\{(\w+)\}', path)
    summary          = operation.get("summary", name)
    fastapi_path     = path
    route_decorator  = getattr(app, method, None)
    if route_decorator is None:
        return

    func_name = f"{method}_{name}_{hash(path) % 10000}"

    if path_params:
        async def handler(**kwargs):
            import httpx
            real_path = path
            for key, val in kwargs.items():
                real_path = real_path.replace(f"{{{key}}}", str(val))
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.request(
                        method.upper(),
                        f"{REAL_SERVER}{real_path}",
                        timeout=10.0,
                    )
                return JSONResponse(
                    content    = resp.json(),
                    status_code = resp.status_code,
                )
            except Exception as e:
                return JSONResponse(
                    {"error": f"Could not reach real server: {e}"},
                    status_code=502,
                )
        handler.__name__ = func_name
        route_decorator(
            fastapi_path,
            summary           = summary,
            operation_id      = func_name,
            include_in_schema = False,
        )(handler)
    else:
        async def handler():
            import httpx
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.request(
                        method.upper(),
                        f"{REAL_SERVER}{path}",
                        timeout=10.0,
                    )
                return JSONResponse(
                    content     = resp.json(),
                    status_code = resp.status_code,
                )
            except Exception as e:
                return JSONResponse(
                    {"error": f"Could not reach real server: {e}"},
                    status_code=502,
                )
        handler.__name__ = func_name
        route_decorator(
            fastapi_path,
            summary           = summary,
            operation_id      = func_name,
            include_in_schema = False,
        )(handler)


# ─────────────────────────── SUMMARY PAGE ────────────────────────────

def _render_summary(
    endpoints:   list[dict],
    operations:  list[dict | None],
    source_file: str,
    model:       str,
) -> str:
    """
    Render an HTML summary page for the SwaggerLM demo.

    Args:
        endpoints:   List of extracted endpoint dicts.
        operations:  List of generated operations.
        source_file: Source file name.
        model:       Model name used.

    Returns:
        HTML string for the summary page.
    """
    rows = ""
    for ep, op in zip(endpoints, operations):
        method  = ep["method"].upper()
        path    = ep["path"]
        summary = op.get("summary", ep["name"]) if op else ep["name"]
        color   = {
            "GET":    "#61affe",
            "POST":   "#49cc90",
            "PUT":    "#fca130",
            "PATCH":  "#50e3c2",
            "DELETE": "#f93e3e",
        }.get(method, "#999")
        rows += f"""
        <tr>
            <td><span style="background:{color};color:white;padding:3px 8px;
                border-radius:3px;font-weight:bold;font-size:12px">{method}</span></td>
            <td style="font-family:monospace">{path}</td>
            <td>{summary}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>SwaggerLM</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 900px;
            margin: 40px auto; padding: 0 20px; background: #fafafa; }}
    h1 {{ color: #1a1a2e; }}
    .badge {{ background: #e94560; color: white; padding: 4px 10px;
              border-radius: 12px; font-size: 13px; }}
    .info {{ background: #e8f4fd; border-left: 4px solid #2196F3;
             padding: 12px 16px; border-radius: 4px; margin: 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th {{ text-align: left; padding: 10px; background: #f0f0f0; }}
    td {{ padding: 10px; border-bottom: 1px solid #eee; }}
    .btn {{ display: inline-block; background: #1a1a2e; color: white;
            padding: 10px 20px; border-radius: 6px; text-decoration: none;
            margin-top: 20px; }}
  </style>
</head>
<body>
  <h1>⚡ SwaggerLM <span class="badge">Live Demo</span></h1>
  <p>Generated documentation for <code>{source_file}</code> using
     fine-tuned <strong>Qwen2.5-Coder-3B</strong> (QLoRA).</p>
  <div class="info">
    🔗 Requests are proxied to real API at <strong>{REAL_SERVER}</strong>
  </div>
  <p>Model: <code>{model}</code> · Endpoints: <strong>{len(endpoints)}</strong></p>
  <a class="btn" href="/docs">📖 Open Swagger UI →</a>
  <a class="btn" style="margin-left:10px;background:#49cc90" href="/openapi.json">
    📄 openapi.json →</a>
  <table>
    <tr><th>Method</th><th>Path</th><th>Summary</th></tr>
    {rows}
  </table>
</body>
</html>"""


# ──────────────────────────── CLI ─────────────────────────────────────

def main() -> None:
    global REAL_SERVER
    
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI docs and proxy to a real FastAPI server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",       required=True,              help="Python source file or project directory")
    parser.add_argument("--model",       default=DEFAULT_MODEL,      help="Ollama model name")
    parser.add_argument("--port",        type=int, default=DEFAULT_PORT, help="Docs server port")
    parser.add_argument("--real-server", default=REAL_SERVER,        help=f"Real API URL (default: {REAL_SERVER})")
    parser.add_argument("--save",        default=None,               help="Save openapi.json to this path")
    args = parser.parse_args()

    # allow overriding real server from CLI
    
    REAL_SERVER = args.real_server

    if not HAS_FASTAPI:
        print("❌  FastAPI not installed. Run: pip install fastapi uvicorn httpx")
        raise SystemExit(1)
    if not HAS_REQUESTS:
        print("❌  requests not installed. Run: pip install requests")
        raise SystemExit(1)

    source_path = Path(args.input)
    if not source_path.exists():
        print(f"❌  Path not found: {source_path}")
        raise SystemExit(1)

    print(__doc__)
    print("─" * 60)

    # Support both single file and directory scanning
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
        endpoints = extract_endpoints(source)

    if not endpoints:
        print("⚠️  No FastAPI endpoints found.")
        raise SystemExit(1)

    source_name = source_path.name if source_path.is_file() else source_path.name + "/"
    print(f"📂  Total: {len(endpoints)} endpoints\n")

    # check Ollama
    try:
        r      = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(args.model in m for m in models):
            print(f"⚠️  Model '{args.model}' not found. Available: {', '.join(models)}")
            print("   Using fallback generation.\n")
            operations = [None] * len(endpoints)
        else:
            print(f"✅  Connected to Ollama — model: {args.model}\n")
            operations = []
            for ep in endpoints:
                print(f"  🤖  {ep['method'].upper()} {ep['path']}  [{ep['name']}]")
                op = generate_openapi(ep, args.model)
                operations.append(op)
                print(f"       {'✅ Valid JSON' if op else '⚠️  Fallback'}")
    except requests.ConnectionError:
        print("⚠️  Ollama not running — using fallback.\n")
        operations = [None] * len(endpoints)

    # save openapi.json if requested
    if args.save:
        spec = {
            "openapi": "3.0.0",
            "info":    {"title": source_path.stem, "version": "1.0.0"},
            "paths":   {}
        }
        for ep, op in zip(endpoints, operations):
            clean = clean_operation(op) if op else {
                "summary": ep["name"],
                "responses": {"200": {"description": "OK"}}
            }
            spec["paths"].setdefault(ep["path"], {})[ep["method"]] = clean
        Path(args.save).write_text(
            json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n💾  Saved → {args.save}")

    app = build_fastapi_app(
        endpoints   = endpoints,
        operations  = operations,
        source_file = source_name,
        model       = args.model,
    )

    url = f"http://localhost:{args.port}"
    print(f"\n🚀  Docs server: {url}/docs")
    print(f"    Real API:    {REAL_SERVER}")
    print(f"    Press Ctrl+C to stop\n")

    threading.Timer(1.5, lambda: webbrowser.open(f"{url}/docs")).start()
    uvicorn.run(app, host="localhost", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
