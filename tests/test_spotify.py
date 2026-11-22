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


def test_user_client_devices_play_pause_state():
    from companion.spotify import SpotifyUser

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.host == "accounts.spotify.com":
            return httpx.Response(200, json={"access_token": "acc", "expires_in": 3600, "refresh_token": "ref2"})
        if request.url.path == "/v1/me/player/devices":
            return httpx.Response(200, json={"devices": [
                {"id": "phone", "name": "iPhone", "type": "Smartphone", "is_active": False},
                {"id": "mac", "name": "Elliott's MacBook Air", "type": "Computer", "is_active": True},
            ]})
        if request.url.path == "/v1/me/player/play":
            return httpx.Response(204)
        if request.url.path == "/v1/me/player/pause":
            return httpx.Response(204)
        if request.url.path == "/v1/me/player":
            return httpx.Response(200, json={"is_playing": True, "progress_ms": 15500, "item": {"name": "Let It Go"}})
        return httpx.Response(404)

    rotated = []
    u = SpotifyUser("cid", "ref1", client=httpx.Client(transport=httpx.MockTransport(handler)), on_refresh=rotated.append)
    assert u.pick_device() == "mac" and u.device_name == "Elliott's MacBook Air"
    assert rotated == ["ref2"]  # a rotated refresh token is persisted
    u.play("spotify:track:x")
    play = next(s for s in seen if s[1] == "/v1/me/player/play")
    assert play[2]["device_id"] == "mac"
    assert u.state() == {"playing": True, "position": 15.5, "track": "Let It Go"}
    u.pause()
    assert seen[-1][1] == "/v1/me/player/pause"
