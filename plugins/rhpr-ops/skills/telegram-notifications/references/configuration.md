# Telegram configuration

Required environment variables or GitHub Actions secrets:

- `TELEGRAM_BOT_TOKEN`: BotFather token
- `TELEGRAM_CHAT_ID`: user, group, or channel destination
- `TELEGRAM_THREAD_ID`: optional forum topic identifier

For a channel, the bot must be an administrator. For a group, privacy settings
do not prevent the bot from sending messages. A `400` error usually means an
invalid chat/thread ID; `401` means an invalid token; `403` usually means the bot
was blocked or lacks access. Never print the request URL because it contains the
bot token.
