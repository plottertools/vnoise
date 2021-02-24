"""This test suite is a port of Casey Duncan original test suite
"""


def test_perlin_1d_range(noise):
    for i in range(-10000, 10000):
        x = i * 0.49
        n = noise.noise1(x)
        assert -1.0 <= n <= 1.0


def test_perlin_1d_octaves_range(noise):
    for i in range(-1000, 1000):
        for o in range(10):
            x = i * 0.49
            n = noise.noise1(x, octaves=o + 1)
            assert -1.0 <= n <= 1.0


def test_perlin_1d_base(noise):
    assert noise.noise1(0.5) == noise.noise1(0.5, base=0)
    assert noise.noise1(0.5) != noise.noise1(0.5, base=5)
    assert noise.noise1(0.5, base=5) != noise.noise1(0.5, base=1)


def test_perlin_2d_range(noise):
    for i in range(-10000, 10000):
        x = i * 0.49
        y = -i * 0.67
        n = noise.noise2(x, y)
        assert -1.0 <= n <= 1.0


def test_perlin_2d_octaves_range(noise):
    for i in range(-1000, 1000):
        for o in range(10):
            x = -i * 0.49
            y = i * 0.67
            n = noise.noise2(x, y, octaves=o + 1)
            assert -1.0 <= n <= 1.0


def test_perlin_2d_base(noise):
    x, y = 0.73, 0.27
    assert noise.noise2(x, y) == noise.noise2(x, y, base=0)
    assert noise.noise2(x, y) != noise.noise2(x, y, base=5)
    assert noise.noise2(x, y, base=5) != noise.noise2(x, y, base=1)


def test_perlin_3d_range(noise):
    for i in range(-10000, 10000):
        x = -i * 0.49
        y = i * 0.67
        z = -i * 0.727
        n = noise.noise3(x, y, z)
        assert -1.0 <= n <= 1.0


def test_perlin_3d_octaves_range(noise):
    for i in range(-1000, 1000):
        x = i * 0.22
        y = -i * 0.77
        z = -i * 0.17
        for o in range(10):
            n = noise.noise3(x, y, z, octaves=o + 1)
            assert -1.0 <= n <= 1.0


def test_perlin_3d_base(noise):
    x, y, z = 0.1, 0.7, 0.33
    assert noise.noise3(x, y, z) == noise.noise3(x, y, z, base=0)
    assert noise.noise3(x, y, z) != noise.noise3(x, y, z, base=5)
    assert noise.noise3(x, y, z, base=5) != noise.noise3(x, y, z, base=1)
