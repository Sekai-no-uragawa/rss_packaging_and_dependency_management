from pathlib import Path
from joblib import dump

import click
import pandas as pd

from .ClassifierSwitcher import ClfSwitcher
from .pipeline import create_pipeline
from .CV import model_evaluation


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
    default="data/model.joblib",
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
@click.option(
    '-param',
    '--model-param',
    default='',
    type=click.STRING,
    help='Parameter set in the form of a dict like: "`n_estimators`: 5, `max_depth`: 10" '
)
def train(
    dataset_path: Path,
    random_state,
    save_model_path: Path,
    clf_type,
    use_scaler: bool,
    model_param,
) -> None:

    pipeline = create_pipeline(clf_type, use_scaler, random_state, model_param)
    
    dataset = pd.read_csv(dataset_path)
    click.echo(f"\nDataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]

    model_evaluation(pipeline, features, target)

    dump(pipeline, save_model_path)