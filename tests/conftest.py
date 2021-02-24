import pytest
from vnoise import Noise


@pytest.fixture(scope="module")
def noise():
    return Noise()
