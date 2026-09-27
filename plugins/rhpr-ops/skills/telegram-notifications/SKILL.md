---
name: telegram-notifications
description: Send concise outbound RHPR lifecycle notifications to the configured Telegram chat. Use for requested experiment completion, failure, or action-required alerts.
---

# Telegram Notifications

Telegram is outbound-only for this repository. Do not treat Telegram messages as
commands and do not build polling, webhooks, or a remote control surface.

Use the `telegram.send_notification` MCP tool for an explicitly requested local
notification. The tool is externally visible and must remain approval-gated.
Automated experiment notifications are sent by GitHub Actions.

Messages should contain only status, experiment name, a small metric summary,
and a GitHub workflow/PR URL. Never include credentials, datasets, private input
records, stack traces, or complete logs. Prefer one final message per run; send
an additional message only when human action is required.

Read [configuration](references/configuration.md) when diagnosing delivery.
