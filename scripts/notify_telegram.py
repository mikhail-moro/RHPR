from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def build_message(status: str, experiment: str, run_url: str, metrics_path: Path | None) -> str:
    icon = "✅" if status.lower() == "success" else "❌"
    lines = [f"{icon} RHPR Kaggle run: {status}", f"Experiment: {experiment}", run_url]
    if metrics_path and metrics_path.is_file():
        metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
        lines.append("Metrics:")
        for key, value in sorted(metrics.items()):
            lines.append(f"• {key}: {value}")
    return "\n".join(lines)[:4096]


def send_message(message: str, *, timeout: float = 15.0) -> dict[str, Any]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")

    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True,
    }
    thread_id = os.environ.get("TELEGRAM_THREAD_ID", "").strip()
    if thread_id:
        payload["message_thread_id"] = int(thread_id)

    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Telegram API returned HTTP {error.code}: {detail}") from error
    if not result.get("ok"):
        raise RuntimeError(f"Telegram rejected the message: {result.get('description', 'unknown error')}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a compact workflow notification")
    parser.add_argument("--status", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    message = build_message(args.status, args.experiment, args.run_url, args.metrics)
    if args.dry_run:
        print(message)
        return
    send_message(message)
    print("Telegram notification sent")


if __name__ == "__main__":
    main()
