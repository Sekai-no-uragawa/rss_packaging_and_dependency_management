from pathlib import Path

import click
import pandas as pd

from .ClassifierSwitcher import ClfSwitcher
from .data import get_dataset
from .pipeline import create_pipeline
from sklearn.metrics import accuracy_score


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
@click.option(
    '--clf-type',
    type=click.Choice(
        [
            'ExtraTreesClassifier',
            'DecisionTreeClassifier',
            'RandomForestClassifier',
        ]
    ),
    default='ExtraTreesClassifier',
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
def train(
    dataset_path: Path,
    random_state,
    test_size,
    save_model_path: Path,
    clf_type,
    use_scaler: bool,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_size,
    )

    pipeline = create_pipeline(clf_type, use_scaler, random_state)
    pipeline.fit(features_train, target_train)
    print(f'\naccuracy_score: {accuracy_score(target_val, pipeline.predict(features_val))}')