from pathlib import Path
from joblib import dump
import configparser
import ast
import click

import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn


from .pipeline import create_pipeline
from .data import get_dataset


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
    "--clf-type",
    type=click.Choice(
        [
            "ExtraTreesClassifier",
            "DecisionTreeClassifier",
            "RandomForestClassifier",
        ]
    ),
    default="ExtraTreesClassifier",
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "-f",
    "--use-feat-engineering",
    default=True,
    type=bool,
    show_default=True,
)
def tuning(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    clf_type: str,
    use_scaler: bool,
    use_feat_engineering: bool,
) -> None:
    with mlflow.start_run(run_name=f"tuning_{clf_type}") as run:

        features, target = get_dataset(dataset_path, use_feat_engineering)
        X, y = features.to_numpy(), target.to_numpy()

        cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
        # enumerate splits
        outer_results = dict()
        num = 0
        for train_ix, test_ix in cv_outer.split(X):
            num += 1
            print(f"Start {num} KFold split...")
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)

            model = create_pipeline(clf_type, use_scaler, random_state, "")

            config = configparser.ConfigParser()
            config.read("ml_forest\grid_param_config.ini")
            space = dict(
                (clf_type + "__" + x, ast.literal_eval(y))
                for x, y in config["PARAMS"].items()
            )

            search = RandomizedSearchCV(
                model, space, scoring="accuracy", cv=cv_inner, refit=True
            )

            result = search.fit(X_train, np.ravel(y_train))

            best_model = result.best_estimator_
            yhat = best_model.predict(X_test)

            acc = accuracy_score(y_test, yhat)
            f1 = f1_score(y_test, yhat, average="macro")
            roc = roc_auc_score(
                y_test,
                best_model.predict_proba(X_test),
                average="macro",
                multi_class="ovr",
            )

            outer_results.setdefault(acc, (result.best_params_, (acc, f1, roc)))
            print("# report progress")
            print(
                ">acc=%.3f, est=%.3f, f1=%.3f, roc_auc=%.3f, cfg=%s"
                % (acc, result.best_score_, f1, roc, result.best_params_)
            )

        best_params = outer_results[max(outer_results.keys())]

        dict_param = best_params[0]
        for key in list(dict_param.keys()):
            new = key.replace(f"{clf_type}__", "")
            dict_param[new] = dict_param.pop(key)
        for key, param in dict_param.items():
            mlflow.log_param(key, param)

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use-feat-engineering", use_feat_engineering)

        result_score = np.array([result[1] for result in outer_results.values()])
        mlflow.log_metric("accuracy", result_score[0])
        mlflow.log_metric("f1_macro", result_score[1])
        mlflow.log_metric("roc_auc_ovr", result_score[2])

        pipeline = create_pipeline(clf_type, use_scaler, random_state, dict_param)
        pipeline = pipeline.fit(features, target)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="models",
        )

        print(best_params)

        dump(pipeline, save_model_path)
        print(f"Model saved to {save_model_path}")
