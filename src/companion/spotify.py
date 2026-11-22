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


# --- Spotify Connect through the Web API (needs the user's one-time sign-in) -----------------

import hashlib  # noqa: E402
import secrets as _secrets  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import urllib.parse  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

REDIRECT_URI = "http://127.0.0.1:8899/callback"
SCOPES = "user-modify-playback-state user-read-playback-state"


def _pkce_pair() -> tuple[str, str]:
    verifier = _secrets.token_urlsafe(64)[:96]
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


class SpotifyUser:
    """Playback control as the signed-in user: devices, play on a device, pause, state."""

    def __init__(self, client_id: str, refresh_token: str = "", client: httpx.Client | None = None, on_refresh=None):
        self.client_id = client_id
        self.refresh_token = refresh_token
        self._client = client or httpx.Client(timeout=10)
        self._access = ""
        self._until = 0.0
        self._on_refresh = on_refresh  # callback(refresh_token) to persist a rotated token
        self.device_id = ""
        self.device_name = ""

    # --- sign-in ----------------------------------------------------------------------------

    def login(self, open_browser, timeout: float = 240.0) -> bool:
        """Run the PKCE flow: open the consent page, catch the redirect, store the refresh token."""
        verifier, challenge = _pkce_pair()
        state = _secrets.token_urlsafe(16)
        url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode({
            "client_id": self.client_id, "response_type": "code", "redirect_uri": REDIRECT_URI,
            "scope": SCOPES, "code_challenge_method": "S256", "code_challenge": challenge, "state": state,
        })
        got: dict = {}
        done = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if self.path.startswith("/callback") and q.get("state", [""])[0] == state and q.get("code"):
                    got["code"] = q["code"][0]
                    body = b"<h2>Mr Roboto is connected to Spotify. You can close this tab.</h2>"
                else:
                    body = b"<h2>Something went wrong; try again from the app.</h2>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body)
                done.set()

            def log_message(self, *args):  # noqa: A002
                pass

        server = HTTPServer(("127.0.0.1", 8899), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            open_browser(url)
            done.wait(timeout)
        finally:
            server.shutdown()
            server.server_close()
        if "code" not in got:
            return False
        r = self._client.post("https://accounts.spotify.com/api/token", data={
            "client_id": self.client_id, "grant_type": "authorization_code", "code": got["code"],
            "redirect_uri": REDIRECT_URI, "code_verifier": verifier,
        })
        r.raise_for_status()
        self._store(r.json())
        return True

    def _store(self, data: dict) -> None:
        self._access = data["access_token"]
        self._until = time.monotonic() + float(data.get("expires_in", 3600))
        if data.get("refresh_token"):
            self.refresh_token = data["refresh_token"]
            if self._on_refresh:
                self._on_refresh(self.refresh_token)

    def _auth(self) -> str:
        if self._access and time.monotonic() < self._until - 30:
            return self._access
        if not self.refresh_token:
            raise RuntimeError("Spotify account not connected")
        r = self._client.post("https://accounts.spotify.com/api/token", data={
            "client_id": self.client_id, "grant_type": "refresh_token", "refresh_token": self.refresh_token,
        })
        r.raise_for_status()
        self._store(r.json())
        return self._access

    def _req(self, method: str, path: str, **kw) -> httpx.Response:
        r = self._client.request(method, "https://api.spotify.com/v1" + path, headers={"Authorization": f"Bearer {self._auth()}"}, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Spotify {method} {path} -> {r.status_code}: {r.text[:200]}")
        return r

    # --- playback ---------------------------------------------------------------------------

    def devices(self) -> list[dict]:
        return self._req("GET", "/me/player/devices").json().get("devices", [])

    def pick_device(self) -> str:
        """This Mac if Spotify lists it, else the active device, else the first one."""
        devs = self.devices()
        if not devs:
            raise RuntimeError("Spotify sees no devices: open the Spotify app on this Mac and press play once")
        host = socket.gethostname().split(".")[0].lower().replace("-", " ")
        for d in devs:
            if d.get("type") == "Computer" and (host in d.get("name", "").lower().replace("-", " ") or d.get("is_active")):
                break
        else:
            d = next((x for x in devs if x.get("is_active")), devs[0])
        self.device_id, self.device_name = d["id"], d.get("name", "")
        return self.device_id

    def play(self, uri: str) -> None:
        device = self.device_id or self.pick_device()
        self._req("PUT", "/me/player/play", params={"device_id": device}, json={"uris": [uri]})

    def pause(self) -> None:
        try:
            self._req("PUT", "/me/player/pause")
        except RuntimeError as exc:
            _log.debug("spotify pause: %s", exc)

    def state(self) -> dict:
        """{'playing': bool, 'position': seconds, 'track': title or ''}"""
        r = self._req("GET", "/me/player")
        if r.status_code == 204 or not r.content:
            return {"playing": False, "position": 0.0, "track": ""}
        d = r.json()
        item = d.get("item") or {}
        return {
            "playing": bool(d.get("is_playing")),
            "position": float(d.get("progress_ms", 0)) / 1000.0,
            "track": item.get("name", ""),
        }
