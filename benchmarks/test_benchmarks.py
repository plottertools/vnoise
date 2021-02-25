# noinspection PyPackageRequirements
import noise
import numpy as np
import pnoise
import pytest
import vnoise

OCTAVES = [1, 4, 8]

XX = np.linspace(0, 1, 100)
YY = np.linspace(100, 200, 100)
ZZ = np.linspace(0.5, 0.6, 100)

XX2d = np.linspace(0, 1, 1000)
YY2d = np.linspace(100, 200, 1000)

XX1d = np.linspace(0, 10, 1000000)


def vectorize_func(func, xx, yy, zz, octaves):
    out = np.empty(shape=(len(xx), len(yy), len(zz)))

    for i, x in enumerate(xx):
        for j, y in enumerate(yy):
            for k, z in enumerate(zz):
                out[i, j, k] = func(x, y, z, octaves=octaves)

    return out


def vectorize_func_2d(func, xx, yy, octaves):
    out = np.empty(shape=(len(xx), len(yy)))

    for i, x in enumerate(xx):
        for j, y in enumerate(yy):
            out[i, j] = func(x, y, octaves=octaves)

    return out


def vectorize_func_1d(func, xx, octaves):
    out = np.empty(len(xx))

    for i, x in enumerate(xx):
        out[i] = func(x, octaves=octaves)

    return out


FUNCTIONS = ["noise.snoise", "noise.pnoise", "vnoise", "pnoise"]


@pytest.mark.parametrize("func", FUNCTIONS)
@pytest.mark.parametrize("octaves", OCTAVES)
def test_scalar_3d(benchmark, func, octaves):
    if func == "noise.snoise":
        benchmark(noise.snoise3, 0.2, 0.5, 0.8, octaves=octaves)
    elif func == "noise.pnoise":
        benchmark(noise.pnoise3, 0.2, 0.5, 0.8, octaves=octaves)
    elif func == "vnoise":
        vn = vnoise.Noise()
        benchmark(vn.noise3, 0.2, 0.5, 0.8, octaves=octaves)
    elif func == "pnoise":
        pn = pnoise.Noise()
        pn.octaves = octaves
        benchmark(pn.perlin, 0.2, 0.5, 0.8)
    else:
        assert False


@pytest.mark.parametrize("func", FUNCTIONS)
@pytest.mark.parametrize("octaves", OCTAVES)
def test_vectorized_3d(benchmark, func, octaves):
    xx = XX.copy()
    yy = YY.copy()
    zz = ZZ.copy()

    if func == "noise.snoise":
        benchmark(vectorize_func, noise.snoise3, xx, yy, zz, octaves=octaves)
    elif func == "noise.pnoise":
        benchmark(vectorize_func, noise.pnoise3, xx, yy, zz, octaves=octaves)
    elif func == "vnoise":
        vn = vnoise.Noise()
        benchmark(vn.noise3, xx, yy, zz, octaves=octaves)
    elif func == "pnoise":
        pn = pnoise.Noise()
        pn.octaves = octaves
        benchmark(pn.perlin, xx, yy, zz)
    else:
        assert False


@pytest.mark.parametrize("func", FUNCTIONS)
@pytest.mark.parametrize("octaves", OCTAVES)
def test_vectorized_2d(benchmark, func, octaves):
    xx = XX2d.copy()
    yy = YY2d.copy()

    if func == "noise.snoise":
        benchmark(vectorize_func_2d, noise.snoise2, xx, yy, octaves=octaves)
    elif func == "noise.pnoise":
        benchmark(vectorize_func_2d, noise.pnoise2, xx, yy, octaves=octaves)
    elif func == "vnoise":
        vn = vnoise.Noise()
        benchmark(vn.noise2, xx, yy, octaves=octaves)
    elif func == "pnoise":
        pn = pnoise.Noise()
        pn.octaves = octaves
        benchmark(pn.perlin, xx, yy, 0.0)
    else:
        assert False


@pytest.mark.parametrize("func", FUNCTIONS)
@pytest.mark.parametrize("octaves", OCTAVES)
def test_vectorized_1d(benchmark, func, octaves):
    xx = XX2d.copy()

    if func == "noise.snoise":
        benchmark(vectorize_func_2d, noise.snoise2, xx, [0.0], octaves=octaves)
    elif func == "noise.pnoise":
        benchmark(vectorize_func_1d, noise.pnoise1, xx, octaves=octaves)
    elif func == "vnoise":
        vn = vnoise.Noise()
        benchmark(vn.noise1, xx, octaves=octaves)
    elif func == "pnoise":
        pn = pnoise.Noise()
        pn.octaves = octaves
        benchmark(pn.perlin, xx, 0.0, 0.0)
    else:
        assert False
