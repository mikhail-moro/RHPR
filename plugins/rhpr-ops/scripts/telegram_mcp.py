#!/usr/bin/env python3
"""Minimal dependency-free MCP stdio server for one Telegram write tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.notify_telegram import send_message  # noqa: E402


TOOL = {
    "name": "send_notification",
    "description": (
        "Send one concise outbound RHPR status notification to the configured Telegram chat. "
        "Never include secrets, datasets, private records, stack traces, or full logs."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "minLength": 1,
                "maxLength": 4096,
                "description": "Plain-text notification body.",
            },
            "dry_run": {
                "type": "boolean",
                "default": False,
                "description": "Validate and preview without sending.",
            },
        },
        "required": ["message"],
        "additionalProperties": False,
    },
    "annotations": {
        "title": "Send Telegram notification",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
}


def response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    if request_id is None:
        return None

    if method == "initialize":
        requested_version = request.get("params", {}).get("protocolVersion", "2025-06-18")
        return response(
            request_id,
            {
                "protocolVersion": requested_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "rhpr-telegram", "version": "0.1.0"},
                "instructions": (
                    "Outbound notifications only. Ask for approval before sending and never expose secrets."
                ),
            },
        )
    if method == "ping":
        return response(request_id, {})
    if method == "tools/list":
        return response(request_id, {"tools": [TOOL]})
    if method == "tools/call":
        params = request.get("params", {})
        if params.get("name") != "send_notification":
            return error(request_id, -32602, "unknown tool")
        arguments = params.get("arguments", {})
        message = arguments.get("message")
        if not isinstance(message, str) or not 1 <= len(message) <= 4096:
            return error(request_id, -32602, "message must contain 1 to 4096 characters")
        try:
            dry_run = bool(arguments.get("dry_run", False))
            if not dry_run:
                send_message(message)
            result_text = "Validated notification (dry run)." if dry_run else "Telegram notification sent."
            return response(
                request_id,
                {
                    "content": [{"type": "text", "text": result_text}],
                    "structuredContent": {"sent": not dry_run, "characters": len(message)},
                    "isError": False,
                },
            )
        except Exception as exc:
            return response(
                request_id,
                {
                    "content": [{"type": "text", "text": f"Notification failed: {exc}"}],
                    "isError": True,
                },
            )
    return error(request_id, -32601, f"method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            result = handle(message)
        except Exception as exc:
            result = error(None, -32700, f"invalid request: {exc}")
        if result is not None:
            print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
