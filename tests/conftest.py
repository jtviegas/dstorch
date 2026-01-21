"""test configurations."""

from pathlib import Path
import sys
import tempfile
import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.joinpath("src").absolute()))  # isort:skip


@pytest.fixture(scope="session")
def resources_folder() -> str:
    """Provides the location of test respource folder.

    Returns:
        test resources folder path

    """
    return str(Path(__file__).parent.joinpath("resources").absolute())


@pytest.fixture(scope="session")
def temporary_folder() -> str:
    """Provides a temporary folder for testing purposes.

    Returns:
        temporary folder

    """
    # pylint: disable=consider-using-with
    _folder = tempfile.TemporaryDirectory("+wb").name
    _path = Path(_folder)
    if not _path.exists():
        _path.mkdir(parents=True)
    return str(_path.absolute())

@pytest.fixture(scope="session", autouse=True)
def configure_pandas():
    """Configure pandas display options for all tests."""
    import pandas as pd
    
    # Set your desired pandas options here
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 128)
    # Add any other options you need