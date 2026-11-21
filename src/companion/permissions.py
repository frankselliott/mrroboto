"""Ask macOS for microphone access the proper way (AVFoundation), so the prompt appears.

CoreAudio alone does not always trigger the permission prompt for a scripted app bundle; an
explicit AVCaptureDevice request does, and it also lets us read the current status.
"""

from __future__ import annotations

import logging
import threading

_log = logging.getLogger(__name__)

STATUS = {0: "not-determined", 1: "restricted", 2: "denied", 3: "authorized"}


def microphone_status() -> str:
    try:
        import AVFoundation  # noqa: PLC0415

        code = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeAudio)
        return STATUS.get(int(code), str(code))
    except Exception as exc:  # noqa: BLE001
        _log.warning("cannot query microphone status: %s", exc)
        return "unknown"


def request_microphone(timeout: float = 120.0) -> str:
    """Show the macOS microphone prompt if undecided; return the resulting status."""
    status = microphone_status()
    if status != "not-determined":
        return status
    try:
        import AVFoundation  # noqa: PLC0415

        done = threading.Event()
        result: dict = {}

        def handler(granted: bool) -> None:
            result["granted"] = bool(granted)
            done.set()

        AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVFoundation.AVMediaTypeAudio, handler
        )
        done.wait(timeout)
    except Exception as exc:  # noqa: BLE001
        _log.warning("microphone request failed: %s", exc)
    return microphone_status()
