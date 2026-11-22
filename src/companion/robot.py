"""The robot side: supervising Carle's two processes and talking to them.

- ``CarleProcesses`` starts ``carle daemon`` (BLE link + servo-safe queue) and
  ``carle speak-server --no-animate`` (audio → JT_Speaker) as child processes of the
  companion, so they inherit the launcher bundle's Bluetooth permission, and restarts them if
  they die.
- ``DaemonClient`` speaks the daemon's newline-JSON Unix-socket protocol (reimplemented here in
  a few lines rather than importing Carle into this venv).
- ``SpeakClient`` posts PCM to the speak server and can stop playback.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from companion.config import CARLE_DIR, LOG_DIR, ROOT

_log = logging.getLogger(__name__)

SOCKET_PATH = ROOT / ".carle" / "daemon.sock"


def find_uv() -> str:
    for candidate in (
        shutil.which("uv"),
        str(Path.home() / "Library" / "Python" / "3.9" / "bin" / "uv"),
        str(Path.home() / ".local" / "bin" / "uv"),
        "/opt/homebrew/bin/uv",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("uv not found; install it with `python3 -m pip install --user uv`")


class CarleProcesses:
    """Start, watch and stop the Carle daemon and speak server."""

    def __init__(self, address: str, speak_port: int = 8081, speaker_device: str = "JT_Speaker"):
        self.address = address
        self.speak_port = speak_port
        self.speaker_device = speaker_device
        self._daemon: subprocess.Popen | None = None
        self._speak: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.restarts = {"daemon": 0, "speak": 0}
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._launch_daemon()
        self._launch_speak()
        self._thread = threading.Thread(target=self._supervise, name="carle-supervisor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for name, proc in (("speak", self._speak), ("daemon", self._daemon)):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                _log.info("stopped carle %s", name)
        self._daemon = self._speak = None

    def set_address(self, address: str) -> None:
        """Point the daemon at a different robot and restart it."""
        if address == self.address:
            return
        self.address = address
        if self._daemon and self._daemon.poll() is None:
            self._daemon.terminate()
            try:
                self._daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._daemon.kill()
        self._launch_daemon()

    def restart_daemon(self, reason: str = "") -> None:
        """Kill and relaunch the daemon (its CoreBluetooth state can go stale, e.g. after the
        Mac's Bluetooth is toggled off and on; the robot then never reconnects without this)."""
        _log.warning("restarting carle daemon: %s", reason)
        if self._daemon and self._daemon.poll() is None:
            self._daemon.terminate()
            try:
                self._daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._daemon.kill()
        self.restarts["daemon"] += 1
        self._launch_daemon()

    def restart_speak(self, reason: str = "") -> None:
        """Relaunch the speak server (its audio-device list goes stale when the speaker
        disconnects and comes back; a fresh process sees the reconnected device)."""
        _log.warning("restarting carle speak server: %s", reason)
        if self._speak and self._speak.poll() is None:
            self._speak.terminate()
            try:
                self._speak.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._speak.kill()
        self.restarts["speak"] += 1
        self._launch_speak()

    @property
    def daemon_alive(self) -> bool:
        return self._daemon is not None and self._daemon.poll() is None

    @property
    def speak_alive(self) -> bool:
        return self._speak is not None and self._speak.poll() is None

    # --- internals ------------------------------------------------------------------------

    def _carle(self, *args: str) -> list[str]:
        return [find_uv(), "run", "--project", str(CARLE_DIR), "carle", *args]

    def _spawn(self, name: str, cmd: list[str]) -> subprocess.Popen:
        log = open(LOG_DIR / f"{name}.log", "ab")  # noqa: SIM115 - handed to the child
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)  # our venv is not Carle's; let uv pick carle/.venv
        env["PATH"] = f"{Path(find_uv()).parent}:{env.get('PATH', '')}"
        # The watchdog wrapper kills the child if this process disappears (no orphans).
        wrapped = [sys.executable, "-m", "companion._child", "--", *cmd]
        proc = subprocess.Popen(wrapped, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()
        _log.info("started carle %s (pid %d): %s", name, proc.pid, " ".join(cmd[-4:]))
        return proc

    def _launch_daemon(self) -> None:
        if not self.address:
            _log.warning("no robot address configured; daemon not started")
            return
        # A stale lock from a crashed run would make `daemon start` refuse; Carle checks the
        # lock holder's pid, so a dead holder is fine, but a live orphan must be told to stop.
        try:
            DaemonClient().shutdown()
            time.sleep(0.5)
        except Exception:  # noqa: BLE001 - nothing was running, which is the normal case
            pass
        self._daemon = self._spawn(
            "carle-daemon", self._carle("daemon", "start", self.address, "--foreground")
        )

    def _reap_stale_speak_server(self) -> None:
        """Terminate a leftover `carle speak-server` holding our port from an earlier run."""
        try:
            out = subprocess.run(
                ["lsof", "-t", "-i", f"tcp:{self.speak_port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5,
            ).stdout.split()
        except (OSError, subprocess.TimeoutExpired):
            return
        for pid in out:
            try:
                cmdline = subprocess.run(
                    ["ps", "-o", "command=", "-p", pid], capture_output=True, text=True, timeout=5
                ).stdout
                if "carle" in cmdline and "speak-server" in cmdline:
                    _log.warning("terminating stale speak server pid %s", pid)
                    os.kill(int(pid), 15)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                continue

    def _launch_speak(self) -> None:
        self._reap_stale_speak_server()
        self._speak = self._spawn(
            "carle-speak-server",
            self._carle(
                "speak-server",
                "--no-animate",
                "--port",
                str(self.speak_port),
                "--device",
                self.speaker_device,
            ),
        )

    def _supervise(self) -> None:
        backoff = {"daemon": 2.0, "speak": 2.0}
        while not self._stop.wait(2.0):
            if self.address and self._daemon is not None and self._daemon.poll() is not None:
                _log.warning("carle daemon exited (%s); restarting", self._daemon.returncode)
                self.restarts["daemon"] += 1
                time.sleep(backoff["daemon"])
                backoff["daemon"] = min(backoff["daemon"] * 2, 30)
                self._launch_daemon()
            elif self.daemon_alive:
                backoff["daemon"] = 2.0
            if self._speak is not None and self._speak.poll() is not None:
                _log.warning("carle speak server exited (%s); restarting", self._speak.returncode)
                self.restarts["speak"] += 1
                time.sleep(backoff["speak"])
                backoff["speak"] = min(backoff["speak"] * 2, 30)
                self._launch_speak()
            elif self.speak_alive:
                backoff["speak"] = 2.0


class DaemonError(Exception):
    pass


class DaemonClient:
    """Blocking client for the daemon's control socket (newline-delimited JSON)."""

    def __init__(self, socket_path: Path = SOCKET_PATH, timeout: float = 5.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self._lock = threading.Lock()

    def request(self, req: dict) -> dict:
        with self._lock:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            try:
                s.connect(str(self.socket_path))
                s.sendall((json.dumps(req) + "\n").encode())
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
            except (OSError, socket.timeout) as exc:
                raise DaemonError(f"daemon unreachable: {exc}") from exc
            finally:
                s.close()
        if not buf:
            raise DaemonError("daemon closed the connection without replying")
        resp = json.loads(buf)
        if not resp.get("ok"):
            raise DaemonError(resp.get("error", "unknown daemon error"))
        return resp

    def status(self) -> dict:
        return self.request({"op": "status"})["status"]

    def connected(self) -> bool:
        try:
            return bool(self.status().get("connected"))
        except (DaemonError, json.JSONDecodeError, KeyError):
            return False

    def enqueue(self, items: list[dict]) -> int:
        if not items:
            return 0
        return int(self.request({"op": "enqueue", "items": items}).get("enqueued", 0))

    def clear(self) -> None:
        self.request({"op": "clear"})

    def stop(self) -> None:
        """Abort now and return the robot to neutral (Carle walks the joints home)."""
        self.request({"op": "stop"})

    def shutdown(self) -> None:
        self.request({"op": "shutdown"})


class SpeakClient:
    """Client for Carle's loopback speak server."""

    def __init__(self, port: int = 8081):
        self.base = f"http://127.0.0.1:{port}"
        self._client = httpx.Client(timeout=httpx.Timeout(5.0, read=None, write=None))

    def alive(self) -> bool:
        try:
            # Any GET is a 405 with a JSON body from a live server.
            return self._client.get(self.base + "/speak/stop", timeout=1.0).status_code == 405
        except httpx.HTTPError:
            return False

    def stream_pcm(self, chunks: Iterator[bytes], samplerate: int, channels: int = 1) -> dict:
        """POST a live raw-PCM (16-bit) stream; returns the server's JSON when playback ends."""
        url = (
            f"{self.base}/speak/stream?format=raw&samplerate={samplerate}&channels={channels}"
        )
        r = self._client.post(url, content=chunks, headers={"Content-Type": "application/octet-stream"})
        try:
            body = r.json()
        except ValueError:
            body = {"ok": False, "error": r.text}
        body["http_status"] = r.status_code
        return body

    def clip(self, data: bytes) -> dict:
        r = self._client.post(self.base + "/speak/clip", content=data)
        body = r.json()
        body["http_status"] = r.status_code
        return body

    def stop(self) -> bool:
        try:
            r = self._client.post(self.base + "/speak/stop", timeout=3.0)
            return bool(r.json().get("stopped"))
        except (httpx.HTTPError, ValueError):
            return False
