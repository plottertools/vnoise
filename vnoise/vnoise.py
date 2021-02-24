import random
from numbers import Number
from typing import Optional, Sequence, Union, overload

import numpy as np

from ._tables import GRAD3, PERM


def _lerp(t, a, b):
    return a + t * (b - a)


def _grad1(
    hash_value: Union[int, np.ndarray], x: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    g = GRAD3[hash_value & 15]
    return x * g[..., 0]


def _grad2(
    hash_value: Union[int, np.ndarray],
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    g = GRAD3[:, 0:2][hash_value & 15]
    return x * g[..., 0] + y * g[..., 1]


def _grad3(
    hash_value: Union[int, np.ndarray],
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    g = GRAD3[hash_value & 15]
    return x * g[..., 0] + y * g[..., 1] + z * g[..., 2]


class Noise:
    def __init__(self, seed: Optional[int] = None):

        if seed is not None:
            self.seed(seed)
        else:
            self._set_perm(PERM)

    def seed(self, s: int) -> None:
        perm = list(PERM)
        random.Random(s).shuffle(perm)
        self._set_perm(perm)

    def _set_perm(self, perm: Sequence[int]) -> None:
        self._perm = np.array(list(perm) * 2, dtype=np.uint8)

    def _noise1_impl(
        self, x: Union[float, np.ndarray], repeat: int, base: int
    ) -> Union[float, np.ndarray]:
        i = np.floor(np.fmod(x, repeat)).astype(int)
        ii = np.fmod(i + 1, repeat)
        i = (i & 255) + base
        ii = (ii & 255) + base
        x = x - np.floor(x)
        fx = x * x * x * (x * (x * 6 - 15) + 10)
        # the triple nested self._perm is required so that noise1(x) == noise2(x, 0.)
        return _lerp(
            fx,
            _grad1(self._perm[self._perm[self._perm[i]]], x),
            _grad1(self._perm[self._perm[self._perm[ii]]], x - 1),
        )

    def _noise2_impl(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        repeat_x: int,
        repeat_y: int,
        base: int,
        grid_mode: bool,
    ) -> Union[float, np.ndarray]:
        i = np.floor(np.fmod(x, repeat_x)).astype(int)
        j = np.floor(np.fmod(y, repeat_y)).astype(int)
        ii = np.fmod(i + 1, repeat_x)
        jj = np.fmod(j + 1, repeat_y)

        i = (i & 255) + base
        j = (j & 255) + base
        ii = (ii & 255) + base
        jj = (jj & 255) + base

        x = x - np.floor(x)
        y = y - np.floor(y)

        x1 = x - 1
        y1 = y - 1

        if grid_mode:
            x, y = np.meshgrid(x, y, indexing="ij", copy=False)
            x1, y1 = np.meshgrid(x1, y1, indexing="ij", copy=False)
            i, j = np.meshgrid(i, j, indexing="ij", copy=False)
            ii, jj = np.meshgrid(ii, jj, indexing="ij", copy=False)

        fx = x * x * x * (x * (x * 6 - 15) + 10)
        fy = y * y * y * (y * (y * 6 - 15) + 10)

        A = self._perm[i]
        AA = self._perm[A + j]
        AB = self._perm[A + jj]
        B = self._perm[ii]
        BA = self._perm[B + j]
        BB = self._perm[B + jj]

        return _lerp(
            fy,
            _lerp(fx, _grad2(self._perm[AA], x, y), _grad2(self._perm[BA], x1, y)),
            _lerp(fx, _grad2(self._perm[AB], x, y1), _grad2(self._perm[BB], x1, y1)),
        )

    def _noise3_impl(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        repeat_x: int,
        repeat_y: int,
        repeat_z: int,
        base: int,
        grid_mode: bool,
    ) -> Union[float, np.ndarray]:
        i = np.floor(np.fmod(x, repeat_x)).astype(int)
        j = np.floor(np.fmod(y, repeat_y)).astype(int)
        k = np.floor(np.fmod(z, repeat_z)).astype(int)
        ii = np.fmod(i + 1, repeat_x)
        jj = np.fmod(j + 1, repeat_y)
        kk = np.fmod(k + 1, repeat_z)

        i = (i & 255) + base
        j = (j & 255) + base
        k = (k & 255) + base
        ii = (ii & 255) + base
        jj = (jj & 255) + base
        kk = (kk & 255) + base

        x = x - np.floor(x)
        y = y - np.floor(y)
        z = z - np.floor(z)

        x1 = x - 1
        y1 = y - 1
        z1 = z - 1

        if grid_mode:
            x, y, z = np.meshgrid(x, y, z, indexing="ij", copy=False)
            x1, y1, z1 = np.meshgrid(x1, y1, z1, indexing="ij", copy=False)
            i, j, k = np.meshgrid(i, j, k, indexing="ij", copy=False)
            ii, jj, kk = np.meshgrid(ii, jj, kk, indexing="ij", copy=False)

        fx = x * x * x * (x * (x * 6 - 15) + 10)
        fy = y * y * y * (y * (y * 6 - 15) + 10)
        fz = z * z * z * (z * (z * 6 - 15) + 10)

        A = self._perm[i]
        AA = self._perm[A + j]
        AB = self._perm[A + jj]
        B = self._perm[ii]
        BA = self._perm[B + j]
        BB = self._perm[B + jj]

        return _lerp(
            fz,
            _lerp(
                fy,
                _lerp(
                    fx,
                    _grad3(self._perm[AA + k], x, y, z),
                    _grad3(self._perm[BA + k], x1, y, z),
                ),
                _lerp(
                    fx,
                    _grad3(self._perm[AB + k], x, y1, z),
                    _grad3(self._perm[BB + k], x1, y1, z),
                ),
            ),
            _lerp(
                fy,
                _lerp(
                    fx,
                    _grad3(self._perm[AA + kk], x, y, z1),
                    _grad3(self._perm[BA + kk], x1, y, z1),
                ),
                _lerp(
                    fx,
                    _grad3(self._perm[AB + kk], x, y1, z1),
                    _grad3(self._perm[BB + kk], x1, y1, z1),
                ),
            ),
        )

    @overload
    def noise1(
        self,
        x: float,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        repeat: int = 1024,
        base: int = 0,
    ) -> float:
        ...

    @overload
    def noise1(
        self,
        x: np.ndarray,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        repeat: int = 1024,
        base: int = 0,
    ) -> np.ndarray:
        ...

    def noise1(self, x, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024, base=0):
        scalar = isinstance(x, Number)

        if not scalar:
            x = np.array(x, dtype=float)

        if octaves == 1:
            res = self._noise1_impl(x, repeat, base)
        elif octaves > 1:
            freq = 1.0
            ampl = 1.0
            max_ampl = 0.0
            total = 0.0
            for i in range(0, octaves):
                total += self._noise1_impl(x * freq, int(repeat * freq), base) * ampl
                max_ampl += ampl
                freq *= lacunarity
                ampl *= persistence
            res = total / max_ampl
        else:
            raise ValueError("Expected octaves value > 0")

        return res

    @overload
    def noise2(
        self,
        x: float,
        y: float,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        repeat_x: int = 1024,
        repeat_y: int = 1024,
        base: int = 0,
        grid_mode: bool = True,
    ) -> float:
        ...

    @overload
    def noise2(
        self,
        x: np.ndarray,
        y: Union[float, np.ndarray],
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        repeat_x: int = 1024,
        repeat_y: int = 1024,
        base: int = 0,
        grid_mode: bool = True,
    ) -> np.ndarray:
        ...

    def noise2(
        self,
        x,
        y,
        octaves=1,
        persistence=0.5,
        lacunarity=2.0,
        repeat_x=1024,
        repeat_y=1024,
        base=0,
        grid_mode=True,
    ):
        """2 dimensional perlin improved noise function (see _noise3_impl for more info)"""
        scalar = isinstance(x, Number) and isinstance(y, Number)

        if not scalar:
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)
        else:
            grid_mode = False

        if octaves == 1:
            res = self._noise2_impl(x, y, repeat_x, repeat_y, base, grid_mode)
        elif octaves > 1:
            freq = 1.0
            ampl = 1.0
            max_ampl = 0.0
            total = 0.0
            for i in range(0, octaves):
                total += (
                    self._noise2_impl(
                        x * freq,
                        y * freq,
                        int(repeat_x * freq),
                        int(repeat_y * freq),
                        base,
                        grid_mode,
                    )
                    * ampl
                )
                max_ampl += ampl
                freq *= lacunarity
                ampl *= persistence
            res = total / max_ampl
        else:
            raise ValueError("Expected octaves value > 0")

        if scalar or not grid_mode:
            return res
        else:
            return res[
                0 if x.shape == () else slice(None),
                0 if y.shape == () else slice(None),
            ]

    @overload
    def noise3(
        self,
        x: float,
        y: float,
        z: float,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        repeat_x: int = 1024,
        repeat_y: int = 1024,
        repeat_z: int = 1024,
        base: int = 0,
        grid_mode: bool = True,
    ) -> float:
        ...

    @overload
    def noise3(
        self,
        x: np.ndarray,
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        repeat_x: int = 1024,
        repeat_y: int = 1024,
        repeat_z: int = 1024,
        base: int = 0,
        grid_mode: bool = True,
    ) -> np.ndarray:
        ...

    def noise3(
        self,
        x,
        y,
        z,
        octaves=1,
        persistence=0.5,
        lacunarity=2.0,
        repeat_x=1024,
        repeat_y=1024,
        repeat_z=1024,
        base=0,
        grid_mode=True,
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

        repeat_x, repeat_y, repeat_z -- specifies the interval along each axis when
        the noise values repeat. This can be used as the tile size for creating
        tileable textures

        base -- specifies a fixed offset for the input coordinates. Useful for
        generating different noise textures with the same repeat interval"
        """

        scalar = isinstance(x, Number) and isinstance(y, Number) and isinstance(z, Number)

        if not scalar:
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)
            z = np.array(z, dtype=float)
        else:
            grid_mode = False

        if octaves == 1:
            res = self._noise3_impl(
                x, y, z, repeat_x, repeat_y, repeat_z, base, grid_mode and not scalar
            )
        elif octaves > 1:
            freq = 1.0
            ampl = 1.0
            max_ampl = 0.0
            total = 0.0
            for i in range(0, octaves):
                total += (
                    self._noise3_impl(
                        x * freq,
                        y * freq,
                        z * freq,
                        int(repeat_x * freq),
                        int(repeat_y * freq),
                        int(repeat_z * freq),
                        base,
                        grid_mode and not scalar,
                    )
                    * ampl
                )
                max_ampl += ampl
                freq *= lacunarity
                ampl *= persistence
            res = total / max_ampl
        else:
            raise ValueError("Expected octaves value > 0")

        if scalar or not grid_mode:
            return res
        else:
            return res[
                0 if x.shape == () else slice(None),
                0 if y.shape == () else slice(None),
                0 if z.shape == () else slice(None),
            ]
