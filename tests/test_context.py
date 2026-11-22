from datetime import datetime

from companion.context import LiveContext, describe_code, time_text, weather_text


def test_time_text_reads_naturally():
    t = time_text(datetime(2026, 11, 22, 15, 5))
    assert t == "Sunday 22 November 2026, 3:05 pm in the afternoon"
    assert "morning" in time_text(datetime(2026, 1, 1, 8, 0))
    assert "night" in time_text(datetime(2026, 1, 1, 22, 30))


def test_weather_text_from_open_meteo_shape():
    data = {
        "current": {"temperature_2m": 23.6, "weather_code": 2, "is_day": 1},
        "daily": {"temperature_2m_max": [27.2], "temperature_2m_min": [18.4], "precipitation_probability_max": [10], "weather_code": [2, 61]},
    }
    assert weather_text(data, "Brisbane") == (
        "Weather in Brisbane: 24 degrees and partly cloudy right now, high of 27 and low of 18 today, "
        "10 percent chance of rain; tomorrow looks a bit rainy."
    )
    assert describe_code(95) == "stormy with thunder" and describe_code(12345) == "a bit of everything"
    assert weather_text({}, "x") == ""


def test_live_context_without_network_still_gives_the_time():
    c = LiveContext("")
    c._loc = {"name": "Nowhere", "lat": 0, "lon": 0}
    c._fetched_at = 0
    assert c.text().startswith("Right now it is ")
