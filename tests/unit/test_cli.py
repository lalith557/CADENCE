from click.testing import CliRunner

from cadence.cli import main


def test_cli_help():
    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "cadence" in r.output.lower()


def test_cli_config_show_default():
    runner = CliRunner()
    r = runner.invoke(main, ["config-show", "-c", "configs/default.yaml"])
    assert r.exit_code == 0
    assert "config_hash" in r.output
