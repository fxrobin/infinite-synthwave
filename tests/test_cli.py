import pytest
from typer.testing import CliRunner

from synthwave.cli import app, parse_duration


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


def test_parse_fx():
    from synthwave.cli import parse_fx
    assert parse_fx("pad:gate:rate=1/16,depth=0.8") == ("pad", {"type": "gate", "rate": "1/16",
                                                                  "depth": 0.8})
    assert parse_fx("master:lofi:bits=8") == ("master", {"type": "lofi", "bits": 8})
    assert parse_fx("arp:bitcrush") == ("arp", {"type": "bitcrush"})


def test_play_export_with_fx(tmp_path):
    out = tmp_path / "fx.wav"
    r = CliRunner().invoke(app, ["play", "--duration", "2s", "--seed", "1", "--export", str(out),
                                 "--fx", "pad:gate:rate=1/16", "--fx", "master:lofi:bits=9"])
    assert r.exit_code == 0, r.output


def test_parse_bpm_range(tmp_path):
    from synthwave.cli import parse_bpm_range
    assert parse_bpm_range("85-100") == (85.0, 100.0)
    with pytest.raises(ValueError):
        parse_bpm_range("100-85")
    r = CliRunner().invoke(app, ["play", "--duration", "1s", "--seed", "1", "--bpm-range",
                                 "70-72", "--export", str(tmp_path / "r.wav")])
    assert r.exit_code == 0, r.output
    assert "bpm=7" in r.output
