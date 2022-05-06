# Evaluation-selection

Homework for RS School Machine Learning course.

This work uses [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

## Usage
This package allows you to train model to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables.
1. Clone this repository to your machine.
2. Download [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.8 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
Additional options:

| Command | Type | Description |
| --- | --- | --- |
|-d, --dataset-path | FILE | [default: data/train.csv] |
| -s, --save-model-path | FILE | [default: data/model.joblib] |
| --random-state | INTEGER | [default: 42] |
| --clf-type | TEXT | [ExtraTreesClassifier|DecisionTreeClassifier|RandomForestClassifier] [default: ExtraTreesClassifier] |
| --use-scaler | BOOLEAN | [default: True] |
| -f, --use-feat-engineering | BOOLEAN | [default: False] |
| -param, --model-param | TEXT |  Parameter set in the form of a dict like: "`n_estimators`: 5, `max_depth`: 10" |
| --help | | Show this message and exit. |
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```