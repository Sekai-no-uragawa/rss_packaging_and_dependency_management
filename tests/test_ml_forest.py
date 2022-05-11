from ml_forest import __version__
from click.testing import CliRunner
import click
import pytest
import pandas as pd
from joblib import load

from ml_forest.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()

def test_version():
    assert __version__ == '0.1.0'

def test_train(
    runner: CliRunner
) -> None:
    result = runner.invoke(
        train,
        [
            "--clf-type",
            'LogReg',
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--clf-type'" in result.output

def test_train_isolated(
    runner: CliRunner
) -> None:
    test_data_path = 'tests/test_data.csv'
    pred_data_path = 'tests/test_data_to_predict.csv'
    test_data = pd.read_csv(test_data_path)
    pred_data = pd.read_csv(pred_data_path)
    with runner.isolated_filesystem('C:/test'):
        test_data.to_csv('test_data.csv')
        pred_data.to_csv('pred_data.csv')
        result = runner.invoke(
            train,
            [
                '--dataset-path',
                'test_data.csv'
            ]
        )
        assert result.exit_code == 0
        print(result.output)
        assert 'Model saved to' in result.output

        pred_data_path = 'pred_data.csv'
        loaded_model = load('data/model.joblib')
        pred_data = pd.read_csv(pred_data_path)
        try:
            result = loaded_model.predict(pred_data)
        except Exception:
            raise Exception('Error in model')