import asyncio
import json

from synthwave import mcp_server


def call(name, **args):
    res = asyncio.run(mcp_server.mcp.call_tool(name, args))
    if getattr(res, "structured_content", None):
        return res.structured_content
    return json.loads(res.content[0].text)


def test_list_patches_tool():
    assert "pad_juno" in call("list_patches")["patches"]


def test_status_when_stopped():
    assert call("status")["running"] is False


def test_export_tool(tmp_path):
    out = tmp_path / "e.wav"
    r = call("export_wav", path=str(out), seconds=2, seed=1, mood="outrun")
    assert r["ok"] and out.exists() and r["seconds"] >= 2


def test_commands_without_player_return_error():
    r = call("set_tempo", bpm=120)
    assert r["ok"] is False and "not running" in r["error"]
