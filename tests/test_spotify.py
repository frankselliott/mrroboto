import httpx

from companion.dialogue import strip_all
from companion.spotify import SpotifyApp, SpotifySearch, Track


def test_search_uses_client_credentials_and_parses_a_track():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.host == "accounts.spotify.com":
            assert request.headers["Authorization"].startswith("Basic ")
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert request.headers["Authorization"] == "Bearer tok"
        assert request.url.params["q"] in ("let it go", "again")
        return httpx.Response(200, json={"tracks": {"items": [
            {"uri": "spotify:track:abc", "name": "Let It Go", "artists": [{"name": "Idina Menzel"}], "duration_ms": 224000}
        ]}})

    s = SpotifySearch("id", "secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    t = s.search("let it go")
    assert t == Track("spotify:track:abc", "Let It Go", "Idina Menzel", 224000)
    assert t.title == "Let It Go by Idina Menzel"
    s.search("again")
    assert sum(1 for c in calls if c.url.host == "accounts.spotify.com") == 1  # token cached


def test_search_empty_result():
    handler = lambda r: httpx.Response(200, json={"access_token": "t", "expires_in": 10, "tracks": {"items": []}})  # noqa: E731
    s = SpotifySearch("id", "secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert s.search("nothing") is None


def test_app_wrappers_talk_applescript():
    scripts = []

    def fake_run(script, timeout=8.0):
        scripts.append(script)
        if "player state" in script:
            return "playing"
        if "player position" in script:
            return "12.5"
        if "name of current track" in script:
            return "Intergalactic\nBeastie Boys\n211000\nspotify:track:xyz"
        return ""

    app = SpotifyApp(run=fake_run)
    app.play("spotify:track:xyz")
    assert 'play track "spotify:track:xyz"' in scripts[-1]
    assert app.state() == "playing" and app.position() == 12.5
    assert app.now_playing() == Track("spotify:track:xyz", "Intergalactic", "Beastie Boys", 211000)
    app.pause()
    assert "pause" in scripts[-1]


def test_spotify_tag_is_an_action():
    clean, tags, actions = strip_all("Here it comes! [spotify:Let It Go Idina Menzel]")
    assert clean == "Here it comes!" and actions == [("spotify", "Let It Go Idina Menzel")]
