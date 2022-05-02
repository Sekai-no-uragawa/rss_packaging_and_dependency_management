from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataset(
    csv_path: Path,
    random_state: int,
    test_split_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"\nDataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
#     click.echo(
#         f"\nfeatures_train shape: {features_train.shape}.\n\
# features_val shape: {features_val.shape}.\n\
# target_train shape: {target_train.shape}.\n\
# target_val shape: {target_val.shape}.\n")
    return features_train, features_val, target_train, target_val