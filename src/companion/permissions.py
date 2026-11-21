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


BT_STATUS = {0: "not-determined", 1: "restricted", 2: "denied", 3: "authorized"}


def bluetooth_status() -> str:
    try:
        import CoreBluetooth  # noqa: PLC0415

        return BT_STATUS.get(int(CoreBluetooth.CBManager.authorization()), "unknown")
    except Exception as exc:  # noqa: BLE001
        _log.warning("cannot query bluetooth status: %s", exc)
        return "unknown"


def request_bluetooth(timeout: float = 120.0) -> str:
    """Show the macOS Bluetooth prompt if undecided (by creating a central manager)."""
    status = bluetooth_status()
    if status != "not-determined":
        return status
    try:
        import time  # noqa: PLC0415

        import CoreBluetooth  # noqa: PLC0415
        import objc  # noqa: PLC0415
        from Foundation import NSObject, NSRunLoop, NSDate  # noqa: PLC0415

        class _Delegate(NSObject):
            def centralManagerDidUpdateState_(self, central):  # noqa: N802
                pass

        delegate = _Delegate.alloc().init()
        manager = CoreBluetooth.CBCentralManager.alloc().initWithDelegate_queue_(delegate, None)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and bluetooth_status() == "not-determined":
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.5))
        del manager, delegate
        objc.recycleAutoreleasePool()
    except Exception as exc:  # noqa: BLE001
        _log.warning("bluetooth request failed: %s", exc)
    return bluetooth_status()
