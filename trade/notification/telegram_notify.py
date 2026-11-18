"""Telegram implementation of the live-trading notification interface."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import Any

import requests

from trade.notification.notify import Notify


class TelegramNotify(Notify):
    """Send messages through a Telegram bot using file-based credentials."""

    TOKEN_FILES = ("bot_token", "telegram_bot_token", "token")
    CHAT_ID_FILES = ("chat_id", "telegram_chat_id")

    def __init__(
        self,
        credential_path: str,
        *,
        logger: logging.Logger | None = None,
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._logger = logger or logging.getLogger("trade.notification.telegram")
        self._session = session or requests.Session()
        self._timeout = float(timeout)
        self._bot_token, self._chat_id = self._load_credentials(credential_path)

    @staticmethod
    def _read_first(directory: str, candidates: Iterable[str]) -> str:
        for filename in candidates:
            path = os.path.join(directory, filename)
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                value = handle.read().strip()
            if value:
                return value
        raise FileNotFoundError(
            f"Telegram credential was not found in {directory}: {tuple(candidates)}"
        )

    @classmethod
    def _load_credentials(cls, credential_path: str) -> tuple[str, str]:
        path = os.path.realpath(str(credential_path))
        if os.path.isdir(path):
            return (
                cls._read_first(path, cls.TOKEN_FILES),
                cls._read_first(path, cls.CHAT_ID_FILES),
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Telegram credential path not found: {path}")

        with open(path, "r", encoding="utf-8") as handle:
            raw_credential = handle.read().strip()

        if raw_credential.startswith("{"):
            payload: Any = json.loads(raw_credential)
            if not isinstance(payload, dict):
                raise TypeError("Telegram credential JSON must contain an object")
            token = str(payload.get("bot_token") or payload.get("token") or "").strip()
            chat_id = str(payload.get("chat_id") or "").strip()
        else:
            first_separator = raw_credential.find(":")
            second_separator = raw_credential.find(":", first_separator + 1)
            if first_separator > 0 and second_separator > first_separator:
                chat_id = raw_credential[:first_separator].strip()
                token = raw_credential[first_separator + 1 :].strip()
            else:
                token = raw_credential
                chat_id = cls._read_first(
                    os.path.dirname(path),
                    cls.CHAT_ID_FILES,
                )

        if not token or not chat_id:
            raise ValueError("Telegram bot_token and chat_id must not be empty")
        return token, chat_id

    def send(self, message: str) -> bool:
        text = str(message).strip()
        if not text:
            raise ValueError("Telegram notification message must not be empty")
        try:
            self.call("sendMessage", chat_id=self._chat_id, text=text)
            return True
        except Exception as exc:
            # Do not log the request URL because it contains the bot token.
            self._logger.warning(
                "Telegram notification delivery failed | error_type=%s",
                type(exc).__name__,
            )
            return False

    @property
    def chat_id(self) -> str:
        return self._chat_id

    def call(self, method: str, *, request_timeout=None, **params):
        try:
            response = self._session.post(
                f"https://api.telegram.org/bot{self._bot_token}/{method}",
                json=params,
                timeout=self._timeout if request_timeout is None else request_timeout,
                allow_redirects=False,
            )
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"Telegram {method}: {type(exc).__name__}") from None
        if not response.ok or not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RuntimeError(f"Telegram {method}: HTTP {response.status_code}")
        return payload.get("result")
