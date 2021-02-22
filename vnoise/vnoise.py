# Code is derived from the C noise code:
# Copyright (c) 2008, Casey Duncan (casey dot duncan at gmail dot com)
# MIT LICENSE.
# This port is also, MIT LICENSE.
from numbers import Number

import numpy as np

M_1_PI = 0.31830988618379067154

GRAD3 = np.array(
    (
        (1, 1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (-1, -1, 0),
        (1, 0, 1),
        (-1, 0, 1),
        (1, 0, -1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, -1, 1),
        (0, 1, -1),
        (0, -1, -1),
        (1, 0, -1),
        (-1, 0, -1),
        (0, -1, 1),
        (0, 1, 1),
    ),
    dtype=int,
)

GRAD4 = (
    (0, 1, 1, 1),
    (0, 1, 1, -1),
    (0, 1, -1, 1),
    (0, 1, -1, -1),
    (0, -1, 1, 1),
    (0, -1, 1, -1),
    (0, -1, -1, 1),
    (0, -1, -1, -1),
    (1, 0, 1, 1),
    (1, 0, 1, -1),
    (1, 0, -1, 1),
    (1, 0, -1, -1),
    (-1, 0, 1, 1),
    (-1, 0, 1, -1),
    (-1, 0, -1, 1),
    (-1, 0, -1, -1),
    (1, 1, 0, 1),
    (1, 1, 0, -1),
    (1, -1, 0, 1),
    (1, -1, 0, -1),
    (-1, 1, 0, 1),
    (-1, 1, 0, -1),
    (-1, -1, 0, 1),
    (-1, -1, 0, -1),
    (1, 1, 1, 0),
    (1, 1, -1, 0),
    (1, -1, 1, 0),
    (1, -1, -1, 0),
    (-1, 1, 1, 0),
    (-1, 1, -1, 0),
    (-1, -1, 1, 0),
    (-1, -1, -1, 0),
)

PERM = np.array(
    (
        151,
        160,
        137,
        91,
        90,
        15,
        131,
        13,
        201,
        95,
        96,
        53,
        194,
        233,
        7,
        225,
        140,
        36,
        103,
        30,
        69,
        142,
        8,
        99,
        37,
        240,
        21,
        10,
        23,
        190,
        6,
        148,
        247,
        120,
        234,
        75,
        0,
        26,
        197,
        62,
        94,
        252,
        219,
        203,
        117,
        35,
        11,
        32,
        57,
        177,
        33,
        88,
        237,
        149,
        56,
        87,
        174,
        20,
        125,
        136,
        171,
        168,
        68,
        175,
        74,
        165,
        71,
        134,
        139,
        48,
        27,
        166,
        77,
        146,
        158,
        231,
        83,
        111,
        229,
        122,
        60,
        211,
        133,
        230,
        220,
        105,
        92,
        41,
        55,
        46,
        245,
        40,
        244,
        102,
        143,
        54,
        65,
        25,
        63,
        161,
        1,
        216,
        80,
        73,
        209,
        76,
        132,
        187,
        208,
        89,
        18,
        169,
        200,
        196,
        135,
        130,
        116,
        188,
        159,
        86,
        164,
        100,
        109,
        198,
        173,
        186,
        3,
        64,
        52,
        217,
        226,
        250,
        124,
        123,
        5,
        202,
        38,
        147,
        118,
        126,
        255,
        82,
        85,
        212,
        207,
        206,
        59,
        227,
        47,
        16,
        58,
        17,
        182,
        189,
        28,
        42,
        223,
        183,
        170,
        213,
        119,
        248,
        152,
        2,
        44,
        154,
        163,
        70,
        221,
        153,
        101,
        155,
        167,
        43,
        172,
        9,
        129,
        22,
        39,
        253,
        19,
        98,
        108,
        110,
        79,
        113,
        224,
        232,
        178,
        185,
        112,
        104,
        218,
        246,
        97,
        228,
        251,
        34,
        242,
        193,
        238,
        210,
        144,
        12,
        191,
        179,
        162,
        241,
        81,
        51,
        145,
        235,
        249,
        14,
        239,
        107,
        49,
        192,
        214,
        31,
        181,
        199,
        106,
        157,
        184,
        84,
        204,
        176,
        115,
        121,
        50,
        45,
        127,
        4,
        150,
        254,
        138,
        236,
        205,
        93,
        222,
        114,
        67,
        29,
        24,
        72,
        243,
        141,
        128,
        195,
        78,
        66,
        215,
        61,
        156,
        180,
        151,
        160,
        137,
        91,
        90,
        15,
        131,
        13,
        201,
        95,
        96,
        53,
        194,
        233,
        7,
        225,
        140,
        36,
        103,
        30,
        69,
        142,
        8,
        99,
        37,
        240,
        21,
        10,
        23,
        190,
        6,
        148,
        247,
        120,
        234,
        75,
        0,
        26,
        197,
        62,
        94,
        252,
        219,
        203,
        117,
        35,
        11,
        32,
        57,
        177,
        33,
        88,
        237,
        149,
        56,
        87,
        174,
        20,
        125,
        136,
        171,
        168,
        68,
        175,
        74,
        165,
        71,
        134,
        139,
        48,
        27,
        166,
        77,
        146,
        158,
        231,
        83,
        111,
        229,
        122,
        60,
        211,
        133,
        230,
        220,
        105,
        92,
        41,
        55,
        46,
        245,
        40,
        244,
        102,
        143,
        54,
        65,
        25,
        63,
        161,
        1,
        216,
        80,
        73,
        209,
        76,
        132,
        187,
        208,
        89,
        18,
        169,
        200,
        196,
        135,
        130,
        116,
        188,
        159,
        86,
        164,
        100,
        109,
        198,
        173,
        186,
        3,
        64,
        52,
        217,
        226,
        250,
        124,
        123,
        5,
        202,
        38,
        147,
        118,
        126,
        255,
        82,
        85,
        212,
        207,
        206,
        59,
        227,
        47,
        16,
        58,
        17,
        182,
        189,
        28,
        42,
        223,
        183,
        170,
        213,
        119,
        248,
        152,
        2,
        44,
        154,
        163,
        70,
        221,
        153,
        101,
        155,
        167,
        43,
        172,
        9,
        129,
        22,
        39,
        253,
        19,
        98,
        108,
        110,
        79,
        113,
        224,
        232,
        178,
        185,
        112,
        104,
        218,
        246,
        97,
        228,
        251,
        34,
        242,
        193,
        238,
        210,
        144,
        12,
        191,
        179,
        162,
        241,
        81,
        51,
        145,
        235,
        249,
        14,
        239,
        107,
        49,
        192,
        214,
        31,
        181,
        199,
        106,
        157,
        184,
        84,
        204,
        176,
        115,
        121,
        50,
        45,
        127,
        4,
        150,
        254,
        138,
        236,
        205,
        93,
        222,
        114,
        67,
        29,
        24,
        72,
        243,
        141,
        128,
        195,
        78,
        66,
        215,
        61,
        156,
        180,
    ),
    dtype=int,
)

SIMPLEX = (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (0, 0, 0, 0),
    (0, 2, 3, 1),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (1, 2, 3, 0),
    (0, 2, 1, 3),
    (0, 0, 0, 0),
    (0, 3, 1, 2),
    (0, 3, 2, 1),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (1, 3, 2, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (1, 2, 0, 3),
    (0, 0, 0, 0),
    (1, 3, 0, 2),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (2, 3, 0, 1),
    (2, 3, 1, 0),
    (1, 0, 2, 3),
    (1, 0, 3, 2),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (2, 0, 3, 1),
    (0, 0, 0, 0),
    (2, 1, 3, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (2, 0, 1, 3),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (3, 0, 1, 2),
    (3, 0, 2, 1),
    (0, 0, 0, 0),
    (3, 1, 2, 0),
    (2, 1, 0, 3),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (3, 1, 0, 2),
    (0, 0, 0, 0),
    (3, 2, 0, 1),
    (3, 2, 1, 0),
)


def lerp(t, a, b):
    return a + t * (b - a)


def grad1(hash: int, x: float):
    g = (hash & 7) + 1.0
    if hash & 8:
        g = -1
    return g * x


def noise1(x: float, repeat: int, base: int) -> float:
    fx = 0.0
    i = int(np.floor(x) % repeat)
    ii = (i + 1) % repeat
    i = (i & 255) + base
    ii = (ii & 255) + base
    x -= np.floor(x)
    fx = x * x * x * (x * (x * 6 - 15) + 10)
    return lerp(fx, grad1(PERM[i], x), grad1(PERM[ii], x - 1)) * 0.4


def grad2(hash: int, x: float, y: float) -> float:
    h = hash & 15
    return x * GRAD3[h][0] + y * GRAD3[h][1]


def noise2(x: float, y: float, repeatx: float, repeaty: float, base: int) -> float:
    i = int(np.floor(np.fmod(x, repeatx)))
    j = int(np.floor(np.fmod(y, repeaty)))
    ii = int(np.fmod(i + 1, repeatx))
    jj = int(np.fmod(j + 1, repeaty))
    i = (i & 255) + base
    j = (j & 255) + base
    ii = (ii & 255) + base
    jj = (jj & 255) + base

    x -= np.floor(x)
    y -= np.floor(y)
    fx = x * x * x * (x * (x * 6 - 15) + 10)
    fy = y * y * y * (y * (y * 6 - 15) + 10)

    A = PERM[i]
    AA = PERM[A + j]
    AB = PERM[A + jj]
    B = PERM[ii]
    BA = PERM[B + j]
    BB = PERM[B + jj]

    return lerp(
        fy,
        lerp(fx, grad2(PERM[AA], x, y), grad2(PERM[BA], x - 1, y)),
        lerp(fx, grad2(PERM[AB], x, y - 1), grad2(PERM[BB], x - 1, y - 1)),
    )


def grad3(hash: int, x: float, y: float, z: float) -> float:
    g = GRAD3[hash & 15]
    return x * g[..., 0] + y * g[..., 1] + z * g[..., 2]


def noise3(
    x: float, y: float, z: float, repeatx: int, repeaty: int, repeatz: int, base: int
) -> float:
    """
    NOTE: modifes x, y z!!
    Args:
        x:
        y:
        z:
        repeatx:
        repeaty:
        repeatz:
        base:

    Returns:

    """
    i = np.array(np.floor(np.fmod(x, repeatx)), dtype=int)
    j = np.array(np.floor(np.fmod(y, repeaty)), dtype=int)
    k = np.array(np.floor(np.fmod(z, repeatz)), dtype=int)
    ii = np.fmod(i + 1, repeatx)
    jj = np.fmod(j + 1, repeaty)
    kk = np.fmod(k + 1, repeatz)

    i = (i & 255) + base
    j = (j & 255) + base
    k = (k & 255) + base
    ii = (ii & 255) + base
    jj = (jj & 255) + base
    kk = (kk & 255) + base

    x -= np.floor(x)
    y -= np.floor(y)
    z -= np.floor(z)

    x1 = x - 1
    y1 = y - 1
    z1 = z - 1

    # TODO: add no grid mode
    if not (isinstance(x, Number) and isinstance(y, Number) and isinstance(z, Number)):
        x, y, z = np.meshgrid(x, y, z, indexing="ij", copy=False)
        x1, y1, z1 = np.meshgrid(x1, y1, z1, indexing="ij", copy=False)
        i, j, k = np.meshgrid(i, j, k, indexing="ij", copy=False)
        ii, jj, kk = np.meshgrid(ii, jj, kk, indexing="ij", copy=False)
        single = False
    else:
        single = True

    fx = x * x * x * (x * (x * 6 - 15) + 10)
    fy = y * y * y * (y * (y * 6 - 15) + 10)
    fz = z * z * z * (z * (z * 6 - 15) + 10)

    A = PERM[i]
    AA = PERM[A + j]
    AB = PERM[A + jj]
    B = PERM[ii]
    BA = PERM[B + j]
    BB = PERM[B + jj]

    return lerp(
        fz,
        lerp(
            fy,
            lerp(fx, grad3(PERM[AA + k], x, y, z), grad3(PERM[BA + k], x1, y, z)),
            lerp(fx, grad3(PERM[AB + k], x, y1, z), grad3(PERM[BB + k], x1, y1, z)),
        ),
        lerp(
            fy,
            lerp(fx, grad3(PERM[AA + kk], x, y, z1), grad3(PERM[BA + kk], x1, y, z1)),
            lerp(
                fx,
                grad3(PERM[AB + kk], x, y1, z1),
                grad3(PERM[BB + kk], x1, y1, z1),
            ),
        ),
    )


def py_noise1(x=None, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024, base=0):
    """1 dimensional perlin improved noise function (see noise3 for more info)"""
    if octaves == 1:
        # Single octave, return simple noise
        return noise1(x, repeat, base)
    elif octaves > 1:
        i = 0
        freq = 1.0
        amp = 1.0
        max = 0.0
        total = 0.0
        for i in range(0, octaves):
            total += noise1(x * freq, int(repeat * freq), base) * amp
            max += amp
            freq *= lacunarity
            amp *= persistence
        return total / max
    else:
        raise ValueError("Expected octaves value > 0")


def py_noise2(
    x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=0
):
    """2 dimensional perlin improved noise function (see noise3 for more info)"""
    if octaves == 1:
        # Single octave, return simple noise
        return noise2(x, y, repeatx, repeaty, base)
    elif octaves > 1:
        freq = 1.0
        amp = 1.0
        max = 0.0
        total = 0.0
        for i in range(0, octaves):
            total += noise2(x * freq, y * freq, repeatx * freq, repeaty * freq, base) * amp
            max += amp
            freq *= lacunarity
            amp *= persistence
        return total / max
    else:
        raise ValueError("Expected octaves value > 0")


def py_noise3(
    x,
    y,
    z,
    octaves=1,
    persistence=0.5,
    lacunarity=2.0,
    repeatx=1024,
    repeaty=1024,
    repeatz=1024,
    base=0,
):
    """
    perlin "improved" noise value for specified coordinate

    octaves -- specifies the number of passes for generating fBm noise,
    defaults to 1 (simple noise).

    persistence -- specifies the amplitude of each successive octave relative
    to the one below it. Defaults to 0.5 (each higher octave's amplitude
    is halved). Note the amplitude of the first pass is always 1.0.

    lacunarity -- specifies the frequency of each successive octave relative
    to the one below it, similar to persistence. Defaults to 2.0.

    repeatx, repeaty, repeatz -- specifies the interval along each axis when
    the noise values repeat. This can be used as the tile size for creating
    tileable textures

    base -- specifies a fixed offset for the input coordinates. Useful for
    generating different noise textures with the same repeat interval"
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    if octaves == 1:
        # Single octave, return simple noise
        return noise3(x, y, z, repeatx, repeaty, repeatz, base)
    elif octaves > 1:
        freq = 1.0
        amp = 1.0
        max = 0.0
        total = 0.0
        for i in range(0, octaves):
            total += (
                noise3(
                    x * freq,
                    y * freq,
                    z * freq,
                    int(repeatx * freq),
                    int(repeaty * freq),
                    int(repeatz * freq),
                    base,
                )
                * amp
            )
            max += amp
            freq *= lacunarity
            amp *= persistence
        return total / max
    else:
        raise ValueError("Expected octaves value > 0")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = py_noise3(np.linspace(0, 1, 250), np.linspace(0, 1, 250), 0, octaves=4)
    plt.imshow(img[..., 0])
    plt.show()

    #
    # xx = np.linspace(0, 10, 1000)
    # yy = py_noise3(xx, [0, 1], 100.0, octaves=4)[..., 0, 0]
    #
    # plt.plot(xx, yy)
    # plt.show()
