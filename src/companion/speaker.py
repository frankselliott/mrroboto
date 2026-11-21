"""Keep the robot's Bluetooth speaker (classic A2DP) connected.

macOS does not always reconnect a paired speaker when it comes back; IOBluetooth lets us ask
for the connection ourselves. Nothing here changes pairing or the system default output.
"""

from __future__ import annotations

import logging
import time

_log = logging.getLogger(__name__)


def _paired(name: str):
    import IOBluetooth  # noqa: PLC0415

    for dev in IOBluetooth.IOBluetoothDevice.pairedDevices() or []:
        if (dev.name() or "") == name:
            return dev
    return None


def speaker_connected(name: str = "JT_Speaker") -> bool | None:
    """True/False for a paired speaker; None if it is not paired at all."""
    try:
        dev = _paired(name)
        return None if dev is None else bool(dev.isConnected())
    except Exception as exc:  # noqa: BLE001
        _log.debug("speaker status failed: %s", exc)
        return None


class SpeakerKeeper:
    """Rate-limited reconnect attempts for the speaker."""

    def __init__(self, name: str = "JT_Speaker", min_interval: float = 12.0):
        self.name = name
        self.min_interval = min_interval
        self._last_attempt = 0.0
        self.attempts = 0

    def ensure(self) -> bool | None:
        """Return the speaker's connection state, trying to reconnect it when it is down."""
        state = speaker_connected(self.name)
        if state is not False:
            return state
        now = time.monotonic()
        if now - self._last_attempt < self.min_interval:
            return False
        self._last_attempt = now
        self.attempts += 1
        try:
            dev = _paired(self.name)
            rc = dev.openConnection() if dev is not None else -1
            _log.info("speaker reconnect attempt %d -> %s", self.attempts, rc)
            if rc == 0:
                return True
        except Exception as exc:  # noqa: BLE001
            _log.warning("speaker reconnect failed: %s", exc)
        return False
