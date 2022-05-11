from pathlib import Path
from joblib import dump

import click
import mlflow
import pandas as pd

from .data import feature_engineering


@click.command()
@click.option(
    "-id",
    "--run_id",
    default="2eeea5e737674f758800969e6122015a",
    type=str,
    show_default=True,
)
@click.option(
    "-l",
    "--logged_model",
    default="",
    type=str,
    show_default=True,
    help='The location, in URI format, of the MLflow model. if pass it is compiled based on "id"',
)
@click.option(
    "-p",
    "--predict_data_path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-subm-path",
    default="data/submission.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def predict(
    run_id: str,
    logged_model: str,
    predict_data_path: Path,
    save_subm_path: Path,
) -> None:
    run_data_dict = mlflow.get_run(run_id).data.to_dictionary()
    use_feat_engineering = run_data_dict["params"]["use-feat-engineering"]
    test = pd.read_csv(predict_data_path)
    if use_feat_engineering:
        test = feature_engineering(test)

    if not logged_model:
        logged_model = "runs:/" + run_id + "/models"
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    pred = loaded_model.predict(test)

    submission = pd.DataFrame()
    submission["Id"] = test.Id
    submission["Cover_Type"] = pred
    submission.to_csv(save_subm_path, index=False)
    print(f"Model saved to {save_subm_path}")
