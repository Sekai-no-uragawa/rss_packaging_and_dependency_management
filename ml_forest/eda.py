from pathlib import Path
from pydoc import cli
import click

import numpy as np
import pandas as pd

import pandas_profiling


@click.command()
@click.option(
    "-d",
    "--csv-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-eda-path",
    default="profile_report.html",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def eda(csv_path: Path, save_eda_path: Path):
    data = pd.read_csv(csv_path)
    click.echo("Data loaded.")
    click.echo("Start creating a report...")
    report = data.profile_report(
        missing_diagrams={"Count": False},
        sort=None,
        html={"style": {"full_width": True}},
        progress_bar=False,
    )
    click.echo("Report created, saving...")
    report.to_file(save_eda_path)
    click.echo(f"Report saved to {save_eda_path}")
