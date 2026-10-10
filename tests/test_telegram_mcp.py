from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "plugins" / "rhpr-ops" / "scripts" / "telegram_mcp.py"


class TelegramMcpTest(unittest.TestCase):
    def test_initialize_list_and_dry_run(self) -> None:
        requests = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "send_notification",
                    "arguments": {"message": "RHPR test", "dry_run": True},
                },
            },
        ]
        payload = "".join(json.dumps(item) + "\n" for item in requests)
        completed = subprocess.run(
            [sys.executable, str(SERVER)],
            input=payload,
            text=True,
            capture_output=True,
            check=True,
        )
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([item["id"] for item in responses], [1, 2, 3])
        self.assertEqual(responses[1]["result"]["tools"][0]["name"], "send_notification")
        self.assertFalse(responses[2]["result"]["structuredContent"]["sent"])


if __name__ == "__main__":
    unittest.main()
