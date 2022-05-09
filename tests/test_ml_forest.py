from ml_forest import __version__
from click.testing import CliRunner
import pytest

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