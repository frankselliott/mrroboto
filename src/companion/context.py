"""Live context for the model: the time and the weather, in words a robot can say.

Weather comes from Open-Meteo (free, no key). The location is a place name from settings,
geocoded once; if none is set, the Mac's public connection is looked up once (city only) and
saved so it can be corrected in Settings. Nothing personal is sent anywhere: the only outbound
data is the place name you typed.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime

import httpx

_log = logging.getLogger(__name__)

WMO = {
    0: "clear and sunny", 1: "mostly sunny", 2: "partly cloudy", 3: "cloudy",
    45: "foggy", 48: "foggy", 51: "drizzly", 53: "drizzly", 55: "drizzly",
    56: "icy drizzle", 57: "icy drizzle", 61: "a bit rainy", 63: "rainy", 65: "very rainy",
    66: "icy rain", 67: "icy rain", 71: "a little snowy", 73: "snowy", 75: "very snowy",
    77: "snowy", 80: "showery", 81: "showery", 82: "pouring with rain", 85: "snow showers",
    86: "snow showers", 95: "stormy with thunder", 96: "stormy with hail", 99: "stormy with hail",
}


def describe_code(code: int) -> str:
    return WMO.get(int(code), "a bit of everything")


def birthday_text(owner: str, birthday: str, today: date | None = None) -> str:
    """'Today is Augie's birthday! He is 4.' / 'Augie's birthday is in 3 days.' / ''"""
    if not birthday:
        return ""
    today = today or date.today()
    try:
        y, m, d = (int(x) for x in birthday.strip().split("-"))
    except ValueError:
        return ""
    this_year = date(today.year, m, d) if not (m == 2 and d == 29) else date(today.year, 2, 28)
    nxt = this_year if this_year >= today else this_year.replace(year=today.year + 1)
    days = (nxt - today).days
    age = nxt.year - y
    if days == 0:
        return f"Today is {owner}'s birthday! {owner} is {age} today. Make a big happy fuss and offer a birthday song."
    if days <= 30:
        return f"{owner}'s birthday is in {days} day{'s' if days != 1 else ''} ({owner} will be {age})."
    return ""


def mood_text(now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    if now.hour >= 19 or now.hour < 6:
        return "It is bedtime-ish: be calm, quiet and gentle, offer a slow story or a quiet song rather than a wild dance."
    if now.hour >= 17:
        return "It is the evening: a little calmer than daytime."
    return ""


def time_text(now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    hour = now.hour
    part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening" if hour < 21 else "night"
    return f"{now.strftime('%A %-d %B %Y')}, {now.strftime('%-I:%M %p').lower()} in the {part}"


def lookup_location_by_ip(client: httpx.Client) -> dict | None:
    """City/lat/lon of the Mac's public connection (sends nothing, receives a city)."""
    try:
        r = client.get("https://ipinfo.io/json", timeout=6)
        d = r.json()
        lat, lon = (float(x) for x in d["loc"].split(","))
        name = ", ".join(x for x in (d.get("city"), d.get("region")) if x)
        return {"name": name or d.get("country", "here"), "lat": lat, "lon": lon, "tz": d.get("timezone", "auto")}
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        _log.debug("ip lookup failed: %s", exc)
    return None


def geocode(client: httpx.Client, place: str) -> dict | None:
    try:
        r = client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place, "count": 1, "language": "en", "format": "json"}, timeout=8,
        )
        res = (r.json().get("results") or [None])[0]
        if res:
            return {"name": f"{res['name']}, {res.get('country', '')}".strip(", "), "lat": res["latitude"], "lon": res["longitude"], "tz": res.get("timezone", "auto")}
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        _log.debug("geocode failed: %s", exc)
    return None


def fetch_weather(client: httpx.Client, lat: float, lon: float, tz: str = "auto") -> dict | None:
    try:
        r = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon, "timezone": tz,
                "current": "temperature_2m,weather_code,is_day,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "forecast_days": 2,
            },
            timeout=8,
        )
        return r.json()
    except (httpx.HTTPError, ValueError) as exc:
        _log.debug("weather fetch failed: %s", exc)
        return None


def weather_text(data: dict, place: str) -> str:
    try:
        cur, daily = data["current"], data["daily"]
        now_t = round(cur["temperature_2m"])
        words = describe_code(cur["weather_code"])
        if not cur.get("is_day", 1) and words == "clear and sunny":
            words = "clear"
        hi, lo = round(daily["temperature_2m_max"][0]), round(daily["temperature_2m_min"][0])
        rain = daily["precipitation_probability_max"][0]
        tomorrow = describe_code(daily["weather_code"][1]) if len(daily["weather_code"]) > 1 else None
        text = f"Weather in {place}: {now_t} degrees and {words} right now, high of {hi} and low of {lo} today"
        if rain is not None:
            text += f", {int(rain)} percent chance of rain"
        if tomorrow:
            text += f"; tomorrow looks {tomorrow}"
        return text + "."
    except (KeyError, IndexError, TypeError):
        return ""


class LiveContext:
    """Keeps a fresh one-paragraph context string; refreshes the weather in the background."""

    def __init__(self, place: str = "", refresh_seconds: float = 900, owner: str = "", birthday: str = "", birthdays: str = ""):
        self.place = place
        self.owner = owner
        self.birthday = birthday
        self.birthdays = birthdays  # "Name=YYYY-MM-DD" per line
        self.refresh_seconds = refresh_seconds
        self._client = httpx.Client()
        self._loc: dict | None = None
        self._weather = ""
        self._fetched_at = 0.0
        self._lock = threading.Lock()
        self.resolved_place = ""

    def set_place(self, place: str) -> None:
        if place != self.place:
            self.place = place
            self._loc = None
            self._weather = ""
            self._fetched_at = 0.0

    def refresh(self) -> None:
        """Resolve the location if needed and fetch the weather; safe to call often."""
        with self._lock:
            if time.monotonic() - self._fetched_at < self.refresh_seconds and self._weather:
                return
            if self._loc is None:
                self._loc = geocode(self._client, self.place) if self.place else lookup_location_by_ip(self._client)
                if self._loc:
                    self.resolved_place = self._loc["name"]
            if self._loc:
                data = fetch_weather(self._client, self._loc["lat"], self._loc["lon"], self._loc.get("tz", "auto"))
                if data:
                    self._weather = weather_text(data, self._loc["name"].split(",")[0])
                    self._fetched_at = time.monotonic()

    def text(self) -> str:
        parts = [f"Right now it is {time_text()}."]
        if self._weather:
            parts.append(self._weather)
        for extra in (birthday_text(self.owner, self.birthday), *self._family_birthdays(), mood_text()):
            if extra:
                parts.append(extra)
        return " ".join(parts)

    def _family_birthdays(self) -> list[str]:
        out = []
        for line in (self.birthdays or "").splitlines():
            if "=" in line:
                name, day = (x.strip() for x in line.split("=", 1))
                if name and day:
                    out.append(birthday_text(name, day))
        return out
