from pathlib import Path
from pyexpat import features
from typing import Tuple

import click
import pandas as pd

def feature_engineering(features: pd.DataFrame) -> pd.DataFrame:
    features['Euclidian_Distance_To_Hydrology'] = (
            features['Horizontal_Distance_To_Hydrology']**2 + 
            features['Vertical_Distance_To_Hydrology']**2
        )**0.5
    features['Mean_Elevation_Vertical_Distance_Hydrology'] = (
        features['Elevation'] + features['Vertical_Distance_To_Hydrology']
    )/2
    features['Mean_Distance_Hydrology_Firepoints'] = (
        features['Horizontal_Distance_To_Hydrology'] + 
        features['Horizontal_Distance_To_Fire_Points']
    )/2
    features['Mean_Distance_Hydrology_Roadways'] = (
        features['Horizontal_Distance_To_Hydrology'] + 
        features['Horizontal_Distance_To_Roadways']
    )/2
    features['Mean_Distance_Firepoints_Roadways'] = (
        features['Horizontal_Distance_To_Fire_Points'] + 
        features['Horizontal_Distance_To_Roadways']
    )/2
    return features

def get_dataset(csv_path: Path, features_eng: bool) -> Tuple[pd.DataFrame, pd.Series]:
    
    dataset = pd.read_csv(csv_path)
    click.echo(f"\nDataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    
    if features_eng:
        features = feature_engineering(features)
    
    return features, target