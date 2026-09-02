import pytest
from synthwave.cli import app, parse_duration
from typer.testing import CliRunner


def test_parse_duration():
    assert parse_duration("90") == 90 and parse_duration("90s") == 90
    assert parse_duration("5m") == 300 and parse_duration("1h30m") == 5400
    with pytest.raises(ValueError):
        parse_duration("abc")


def test_patches_command_lists_library():
    r = CliRunner().invoke(app, ["patches"])
    assert r.exit_code == 0 and "pad_juno" in r.output and "drums_808" in r.output


def test_play_export_offline(tmp_path):
    out = tmp_path / "o.wav"
    r = CliRunner().invoke(app, ["play", "--duration", "3s", "--seed", "1", "--export", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists() and out.stat().st_size > 100000
