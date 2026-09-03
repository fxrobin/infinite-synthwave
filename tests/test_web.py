from starlette.testclient import TestClient

from synthwave.web.server import app

client = TestClient(app)


def test_index_and_meta():
    assert "<title>Infinite Synthwave" in client.get("/").text
    meta = client.get("/api/meta").json()
    assert "outrun" in meta["moods"] and "pad_juno" in meta["patches"]
    assert meta["layers"][0] == "drums" and "gate" in meta["effects"]


def test_status_and_commands_without_player():
    assert client.get("/api/status").json()["running"] is False
    r = client.post("/api/tempo", json={"bpm": 120})
    assert r.status_code == 409 and "not running" in r.json()["error"]
    assert client.post("/api/stop").json()["ok"] is False
    assert client.post("/api/auto_tweaks", json={"enabled": False}).status_code == 409
    assert client.get("/api/patch/pad").status_code == 404


def test_start_rejects_bad_config():
    r = client.post("/api/start", json={"bpm": "abc"})
    assert r.status_code == 400


def test_websocket_feed_sends_status():
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["running"] is False and "moods" in msg


def test_master_color_route_and_meta():
    with TestClient(app) as c:
        assert "tape" in c.get("/api/meta").json()["master_colors"]
        bad = c.post("/api/master_color", json={"color": "nope"})
        assert bad.status_code == 400
        off = c.post("/api/master_color", json={"color": "vhs"})
        assert off.status_code == 409  # valid colour, but no player running
