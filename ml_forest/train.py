from calendar import c
from pathlib import Path

import click
from click import Context as ctx
import pandas as pd

from .ClassifierSwitcher import ClfSwitcher
from .data import get_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def create_pipeline(
    clf_type
):
    print(type(clf_type))
#     use_scaler: bool, max_iter: int, logreg_C: float, random_state: int
# ) -> Pipeline:
#     pipeline_steps = []
#     if use_scaler:
#         pipeline_steps.append(("scaler", StandardScaler()))
#     pipeline_steps.append(
#         (
#             "classifier",
#             LogisticRegression(
#                 random_state=random_state, max_iter=max_iter, C=logreg_C
#             ),
#         )
#     )
#     return Pipeline(steps=pipeline_steps)



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
    type=click.Choice(['KNeighborsClassifier', 'DecisionTreeClassifier']),
    default='DecisionTreeClassifier'
)
def train(
    dataset_path: Path,
    random_state,
    test_size,
    save_model_path: Path,
    clf_type
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_size,
    )

    mapping_dict = {
        'KNeighborsClassifier' : KNeighborsClassifier(),
        'DecisionTreeClassifier' : DecisionTreeClassifier(),
    }
    create_pipeline(mapping_dict[clf_type])