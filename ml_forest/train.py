from pathlib import Path

import click
import pandas as pd

from .ClassifierSwitcher import ClfSwitcher
from .data import get_dataset

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",  
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "-s",
    "--save-model-path",
    default="profile_report.html",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-size",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
def train(
    dataset_path: Path,
    random_state,
    test_size,
    save_model_path: Path,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_size,
    )
