from numbers import Number

import numpy as np
import pytest
from vnoise import Noise


def test_noise1_dimensions(noise):
    assert isinstance(noise.noise1(1.5), float)
    assert noise.noise1([1]).shape == (1,)
    assert noise.noise1([1, 3, 4]).shape == (3,)


def test_noise2_dimensions(noise):
    # scalar
    assert isinstance(noise.noise2(1.5, 1.0, grid_mode=True), float)
    assert isinstance(noise.noise2(1.5, 1.0, grid_mode=False), float)

    # grid mode on
    assert noise.noise2([0, 1], [2, 3, 4]).shape == (2, 3)
    assert noise.noise2([1], [2, 3, 4]).shape == (1, 3)
    assert noise.noise2([0, 1], 4).shape == (2,)
    assert noise.noise2(4, [0, 1]).shape == (2,)

    # grid mode off
    assert noise.noise2([0, 1, 2], [3, 4, 5], grid_mode=False).shape == (3,)
    assert noise.noise2([0], [3], grid_mode=False).shape == (1,)

    with pytest.raises(ValueError):
        noise.noise2([0, 1, 2], [3, 4, 5, 6], grid_mode=False)


def test_noise3_dimensions(noise):
    # scalar
    assert isinstance(noise.noise3(1.5, 1.0, 1.5, grid_mode=True), float)
    assert isinstance(noise.noise3(1.5, 1.0, 1.5, grid_mode=False), float)

    # grid mode on
    assert noise.noise3([0, 1], [2, 3, 4], 4).shape == (2, 3)
    assert noise.noise3([0, 1], [2, 3, 4], [5, 6, 7, 8]).shape == (2, 3, 4)
    assert noise.noise3([1], [2, 3, 4], [5, 6, 7, 8]).shape == (1, 3, 4)
    assert noise.noise3([0, 1], 4, [5, 6, 7, 8]).shape == (2, 4)
    assert noise.noise3(4, 4, [5, 6, 7, 8]).shape == (4,)
    assert noise.noise3(4, [0, 1], [5, 6, 7, 8]).shape == (2, 4)

    # grid mode off
    assert noise.noise3([0, 1, 2], [3, 4, 5], [5, 6, 7], grid_mode=False).shape == (3,)
    assert noise.noise3([0], [3], [5], grid_mode=False).shape == (1,)

    with pytest.raises(ValueError):
        noise.noise3([0, 1, 2], [3, 4, 5, 6], [5, 6, 7], grid_mode=False)


@pytest.mark.parametrize("seed", [None, 0, 1])
@pytest.mark.parametrize("octaves", list(range(1, 9)))
@pytest.mark.parametrize("x", [10, [1], 0.5, range(5), np.linspace(0, 10, 100)])
def test_noise1_noise2(seed, x, octaves):
    noise = Noise(seed)
    y = 0 if isinstance(x, Number) else np.zeros_like(x)

    n2 = noise.noise2(x, y, octaves=octaves, grid_mode=False)
    n1 = noise.noise1(x, octaves=octaves)
    assert np.all(n1 == n2)


@pytest.mark.parametrize("seed", [None, 0, 1])
@pytest.mark.parametrize("octaves", list(range(1, 9)))
@pytest.mark.parametrize(
    ["x", "y", "grid_mode"],
    [
        (0, 10, False),
        (0.0, 15.0, False),
        (range(5), 0, True),
        (np.linspace(0, 10, 100), np.linspace(10, 20, 100), False),
    ],
)
def test_noise2_noise3(seed, x, y, grid_mode, octaves):
    noise = Noise(seed)
    z = 0 if grid_mode else np.zeros_like(x)
    assert np.all(
        noise.noise2(x, y, octaves=octaves, grid_mode=grid_mode)
        == noise.noise3(x, y, z, octaves=octaves, grid_mode=grid_mode)
    )
