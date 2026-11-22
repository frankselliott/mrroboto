"""Spotify: find a track through the Web API and play it in the Spotify app on this Mac.

Search uses the client-credentials flow (a free developer app's ID and secret, no user login).
Playback goes through the Spotify desktop app via AppleScript, so the music comes out of the
Mac's selected output — the robot's speaker. Tempo and section data are not available from
Spotify any more, so dances to Spotify tracks use the themed routines at a steady tempo.
"""

from __future__ import annotations

import base64
import logging
import subprocess
import time
from dataclasses import dataclass

import httpx

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Track:
    uri: str
    name: str
    artist: str
    duration_ms: int

    @property
    def title(self) -> str:
        return f"{self.name} by {self.artist}" if self.artist else self.name


class SpotifySearch:
    def __init__(self, client_id: str, client_secret: str, client: httpx.Client | None = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self._client = client or httpx.Client(timeout=10)
        self._token = ""
        self._token_until = 0.0

    def _auth(self) -> str:
        if self._token and time.monotonic() < self._token_until - 30:
            return self._token
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        r = self._client.post(
            "https://accounts.spotify.com/api/token",
            headers={"Authorization": f"Basic {basic}"},
            data={"grant_type": "client_credentials"},
        )
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        self._token_until = time.monotonic() + float(data.get("expires_in", 3600))
        return self._token

    def search(self, query: str) -> Track | None:
        r = self._client.get(
            "https://api.spotify.com/v1/search",
            params={"q": query, "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {self._auth()}"},
        )
        r.raise_for_status()
        items = r.json().get("tracks", {}).get("items", [])
        if not items:
            return None
        t = items[0]
        return Track(t["uri"], t["name"], ", ".join(a["name"] for a in t.get("artists", [])), int(t.get("duration_ms", 0)))


# --- the Spotify app, via AppleScript ------------------------------------------------------


def _osascript(script: str, timeout: float = 8.0) -> str:
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "osascript failed")
    return r.stdout.strip()


class SpotifyApp:
    """Thin control of the Spotify desktop app. Every call is one AppleScript round-trip."""

    def __init__(self, run=_osascript):
        self._run = run

    def available(self) -> bool:
        try:
            self._run('tell application "System Events" to (name of processes) contains "Spotify"')
            return True
        except Exception:  # noqa: BLE001
            return False

    def play(self, uri: str) -> None:
        self._run(f'tell application "Spotify"\nactivate\nplay track "{uri}"\nend tell')

    def pause(self) -> None:
        try:
            self._run('tell application "Spotify" to pause')
        except Exception as exc:  # noqa: BLE001
            _log.debug("spotify pause failed: %s", exc)

    def state(self) -> str:
        try:
            return self._run('tell application "Spotify" to player state as string')
        except Exception:  # noqa: BLE001
            return "unknown"

    def position(self) -> float:
        try:
            return float(self._run('tell application "Spotify" to player position'))
        except Exception:  # noqa: BLE001
            return 0.0

    def now_playing(self) -> Track | None:
        try:
            out = self._run(
                'tell application "Spotify" to (name of current track) & "\\n" & (artist of current track) '
                '& "\\n" & (duration of current track) & "\\n" & (spotify url of current track)'
            )
            name, artist, duration, url = out.split("\n")[:4]
            return Track(url, name, artist, int(float(duration)))
        except Exception:  # noqa: BLE001
            return None
