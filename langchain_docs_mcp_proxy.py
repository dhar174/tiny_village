#!/usr/bin/env python3
"""Local stdio MCP proxy for the hosted LangChain docs server.

This bypasses the remote server's broken OAuth authorize flow by exposing a
plain stdio MCP server that forwards the useful tool calls upstream.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


REMOTE_MCP_URL = os.environ.get("LANGCHAIN_DOCS_MCP_URL", "https://docs.langchain.com/mcp")
PROTOCOL_VERSION = "2025-03-26"
SERVER_INFO = {
    "name": "langchain-docs-local-proxy",
    "version": "1.0.0",
}

_CACHED_TOOLS: list[dict[str, Any]] | None = None


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def send_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}

    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break

        decoded = line.decode("ascii", errors="replace").strip()
        if ":" not in decoded:
            continue

        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    content_length = headers.get("content-length")
    if not content_length:
        raise ValueError("Missing Content-Length header")

    body = sys.stdin.buffer.read(int(content_length))
    if not body:
        return None

    return json.loads(body.decode("utf-8"))


def parse_sse_payload(raw: bytes) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    data_lines: list[str] = []

    for line in raw.decode("utf-8").splitlines():
        if not line:
            if data_lines:
                events.append(json.loads("\n".join(data_lines)))
                data_lines = []
            continue

        if line.startswith(":"):
            continue

        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]

        if field == "data":
            data_lines.append(value)

    if data_lines:
        events.append(json.loads("\n".join(data_lines)))

    if not events:
        raise ValueError("Remote MCP server returned no SSE payload")

    return events[-1]


def remote_rpc(method: str, params: dict[str, Any] | None = None, request_id: int = 1) -> dict[str, Any]:
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        REMOTE_MCP_URL,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "langchain-docs-local-proxy/1.0.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Remote MCP HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach remote MCP server: {exc}") from exc

    if "text/event-stream" in content_type:
        return parse_sse_payload(payload)
    return json.loads(payload.decode("utf-8"))


def load_tools() -> list[dict[str, Any]]:
    global _CACHED_TOOLS

    if _CACHED_TOOLS is not None:
        return _CACHED_TOOLS

    payload = remote_rpc(
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": SERVER_INFO,
        },
    )

    result = payload.get("result", {})
    tools_map = result.get("capabilities", {}).get("tools", {})
    tool_list: list[dict[str, Any]] = []

    if isinstance(tools_map, dict):
        for name, spec in tools_map.items():
            if name == "listChanged" or not isinstance(spec, dict):
                continue
            tool = {"name": name}
            tool.update(spec)
            tool_list.append(tool)

    if not tool_list:
        raise RuntimeError("Remote MCP server did not advertise any tools")

    _CACHED_TOOLS = tool_list
    return tool_list


def send_success(request_id: Any, result: dict[str, Any]) -> None:
    send_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def send_error(request_id: Any, code: int, message: str) -> None:
    send_message({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def handle_request(message: dict[str, Any]) -> None:
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params", {})

    if method == "initialize":
        send_success(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            },
        )
        return

    if method == "ping":
        send_success(request_id, {})
        return

    if method == "tools/list":
        send_success(request_id, {"tools": load_tools()})
        return

    if method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        payload = remote_rpc(
            "tools/call",
            {
                "name": tool_name,
                "arguments": tool_args,
            },
            request_id=2,
        )

        if "error" in payload:
            error = payload["error"]
            send_error(request_id, error.get("code", -32000), error.get("message", "Remote tool error"))
            return

        send_success(request_id, payload.get("result", {}))
        return

    if request_id is None:
        return

    send_error(request_id, -32601, f"Method not found: {method}")


def main() -> int:
    try:
        while True:
            message = read_message()
            if message is None:
                return 0

            try:
                handle_request(message)
            except Exception as exc:  # pragma: no cover - manual integration script
                request_id = message.get("id")
                log(f"langchain_docs_mcp_proxy error: {exc}")
                if request_id is not None:
                    send_error(request_id, -32000, str(exc))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
