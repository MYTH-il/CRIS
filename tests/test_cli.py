from typer.testing import CliRunner

from cris.cli import app


def test_doctor_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Environment:" in result.stdout
