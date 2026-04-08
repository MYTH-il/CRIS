from pathlib import Path

import typer

from cris.config import settings

app = typer.Typer(help="CRIS command line interface.")


@app.command()
def doctor() -> None:
    """Print the resolved local setup paths."""
    typer.echo(f"Environment: {settings.env}")
    typer.echo(f"Config: {settings.config_path}")
    typer.echo(f"Data dir: {settings.data_dir}")
    typer.echo(f"Artifacts dir: {settings.artifacts_dir}")


@app.command()
def init_dirs() -> None:
    """Create the default project directories."""
    for path in [
        Path("artifacts"),
        Path("data/raw"),
        Path("data/interim"),
        Path("data/processed"),
        Path("data/vector_store"),
        Path("tests"),
    ]:
        path.mkdir(parents=True, exist_ok=True)
        typer.echo(f"created {path}")


if __name__ == "__main__":
    app()
