"""Monthly Telegram backup reminders with persistent acknowledgements."""

from datetime import datetime, time
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import threading
from uuid import uuid4
from zoneinfo import ZoneInfo

from trade.notification.telegram_notify import TelegramNotify

ZONE = ZoneInfo("Asia/Shanghai")
NOON = time(12)


class BackupReminder:
    def __init__(self, credentials, directory, logger):
        self.credentials = credentials
        self.directory = Path(directory)
        self.state_path = self.directory / "backup_reminder.json"
        self.logger = logger
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="backup-reminder", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=6)

    def _save(self, state):
        with NamedTemporaryFile(mode="w", dir=self.directory, encoding="utf-8", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(state, handle, indent=2)
                handle.write("\n")
                handle.close()
                temporary.replace(self.state_path)
            finally:
                temporary.unlink(missing_ok=True)

    def _send_due(self, bot, state, now):
        if now.time() < NOON:
            return
        month = now.strftime("%Y-%m")
        day = now.date().isoformat()
        task = state.get(month)
        if task is None:
            task = state[month] = {"token": f"backup:{uuid4().hex}"}
            self._save(state)
        if task.get("completed_at") or task.get("last_sent_day") == day:
            return
        message = bot.call(
            "sendMessage",
            chat_id=bot.chat_id,
            text=f"Backup reminder for {month}\n\nBack up this live runner directory:\n{self.directory}",
            reply_markup={"inline_keyboard": [[
                {"text": "✅ 已完成", "callback_data": task["token"]},
            ]]},
        )
        task.update(last_sent_day=day, chat_id=message["chat"]["id"])
        self._save(state)

    def _confirm(self, bot, state, callback, now):
        data = callback.get("data") or ""
        if not data.startswith("backup:"):
            return
        task = state.get(now.strftime("%Y-%m"), {})
        message = callback.get("message") or {}
        chat = message.get("chat", {})
        sender = callback.get("from", {}).get("id")
        # The destination restricts group acknowledgements to users in that chat.
        if task.get("token") != data or task.get("chat_id") != chat.get("id") or sender is None:
            try:
                bot.call("answerCallbackQuery", callback_query_id=callback["id"], text="此提醒已失效")
            except RuntimeError as exc:
                self.logger.warning("Backup expiry display failed | %s", exc)
            return
        if chat.get("type") == "private" and sender != chat.get("id"):
            return
        if not task.get("completed_at"):
            completed = {**task, "completed_at": now.isoformat(), "completed_by": sender}
            month = now.strftime("%Y-%m")
            self._save({**state, month: completed})
            state[month] = completed
        try:
            bot.call("answerCallbackQuery", callback_query_id=callback["id"], text="已记录备份完成")
            bot.call(
                "editMessageText", chat_id=chat["id"], message_id=message["message_id"],
                text=f"✅ {now:%Y-%m} 备份已完成\n\n{self.directory}",
                reply_markup={"inline_keyboard": []},
            )
        except RuntimeError as exc:
            self.logger.warning("Backup confirmation display failed | %s", exc)

    def _run(self):
        try:
            bot = TelegramNotify(self.credentials, logger=self.logger, timeout=5)
            state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        except Exception:
            self.logger.exception("Backup reminder could not start")
            return
        offset = 0
        initial_poll = True
        while not self.stop.is_set():
            try:
                now = datetime.now(ZONE)
                seconds_to_noon = (datetime.combine(now.date(), NOON, ZONE) - now).total_seconds()
                timeout = 20
                if initial_poll:
                    timeout = 0
                elif seconds_to_noon > 0:
                    timeout = max(1, min(timeout, int(seconds_to_noon)))
                updates = bot.call(
                    "getUpdates", offset=offset, timeout=timeout,
                    request_timeout=timeout + 5, allowed_updates=["callback_query"],
                )
                if self.stop.is_set():
                    return
                for update in updates:
                    if self.stop.is_set():
                        return
                    if "callback_query" in update:
                        self._confirm(bot, state, update["callback_query"], datetime.now(ZONE))
                    offset = update["update_id"] + 1
                initial_poll = False
                if not self.stop.is_set():
                    self._send_due(bot, state, datetime.now(ZONE))
            except Exception:
                self.logger.exception("Backup reminder update failed")
                self.stop.wait(60)
