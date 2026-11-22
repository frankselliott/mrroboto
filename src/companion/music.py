"""The song library: find songs in a folder and decode them to PCM for the robot's speaker."""

from __future__ import annotations

import difflib
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

AUDIO_EXT = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".aiff", ".aif", ".ogg", ".wma", ".mp4"}
SAMPLE_RATE = 24000
_NOISE = re.compile(
    r"\s*[\(\[][^\)\]]*(official|video|audio|lyrics|hd|4k|remaster)[^\)\]]*[\)\]]|@\S+.*$",
    re.IGNORECASE,
)
#: Downloader prefixes/suffixes such as "YTDown.com_YouTube_Media_<id>_Title_007_48k".
_DOWNLOADER = re.compile(r"^(?:YTDown\.com_)?YouTube_Media_[A-Za-z0-9_-]{11}_|_\d{3}_\d+k$", re.IGNORECASE)
_NOISE_WORDS = re.compile(r"\b(official (?:music )?(?:video|audio|song)|music video|lyric video|lyrics|hd|4k|remastered?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Song:
    title: str
    path: Path

    @property
    def key(self) -> str:
        return re.sub(r"[^a-z0-9 ]", "", self.title.lower()).strip()


def clean_title(stem: str) -> str:
    t = _DOWNLOADER.sub("", stem)
    t = _DOWNLOADER.sub("", t)  # prefix and suffix are separate alternations
    t = _NOISE.sub("", t)
    if "_" in t or ("-" in t and " " not in t):
        t = t.replace("_", " ").replace("-", " ")
    t = _NOISE_WORDS.sub("", t)
    t = re.sub(r"\s+", " ", t).strip(" -_")
    return t or stem


class Library:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder).expanduser()
        self.songs: list[Song] = []
        self.rescan()

    def rescan(self) -> None:
        songs: list[Song] = []
        if self.folder.is_dir():
            for p in sorted(self.folder.iterdir()):
                if p.suffix.lower() in AUDIO_EXT and not p.name.startswith("."):
                    songs.append(Song(clean_title(p.stem), p))
        self.songs = songs

    def titles(self) -> list[str]:
        return [s.title for s in self.songs]

    def find(self, query: str) -> Song | None:
        """Best match for a spoken/typed title: exact, substring, then fuzzy."""
        q = re.sub(r"[^a-z0-9 ]", "", (query or "").lower()).strip()
        if not q or not self.songs:
            return None
        for s in self.songs:
            if s.key == q:
                return s
        subs = [s for s in self.songs if q in s.key or s.key in q]
        if subs:
            return min(subs, key=lambda s: abs(len(s.key) - len(q)))
        # Word overlap next ("the dinosaur song" → the Pinkfong dinosaur track, not "The Robot Song").
        stop = {"the", "a", "song", "play", "to", "dance", "some", "music", "please", "of", "and"}
        qwords = set(q.split()) - stop
        best, score = None, 0.0
        for s in self.songs:
            words = set(s.key.split())
            overlap = len(qwords & words) / max(1, len(qwords))
            if overlap > score:
                best, score = s, overlap
        if best is not None and score >= 0.5:
            return best
        keys = [s.key for s in self.songs]
        close = difflib.get_close_matches(q, keys, n=1, cutoff=0.6)
        if close:
            return self.songs[keys.index(close[0])]
        return None


def decode(song: Song, max_seconds: float = 360.0) -> np.ndarray:
    """Decode any format macOS can read to mono 16-bit PCM at 24 kHz, via afconvert."""
    import soundfile as sf  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "song.wav"
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", str(song.path), str(out)],
            check=True, capture_output=True, timeout=120,
        )
        data, sr = sf.read(out, dtype="int16")
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"unexpected sample rate {sr}")
    pcm = np.asarray(data, dtype=np.int16)
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    pcm = pcm[: int(max_seconds * SAMPLE_RATE)]
    peak = int(np.abs(pcm).max()) if len(pcm) else 0
    if peak:
        pcm = (pcm.astype(np.float32) * (0.9 * 32767 / peak)).astype(np.int16)
    return pcm
