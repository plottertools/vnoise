# *vnoise*

*vnoise* is a pure-Python, Numpy-based, vectorized port of the [*noise*](https://github.com/caseman/noise) library. It
currently implements the Perlin noise functions in 1D, 2D and 3D version (*noise* also implements simplex noise).

## Why?

*vnoise* was started because the original *noise* library is no longer supported and binaries for recent versions of
Python are unavailable, making it hard to install for non-technical users. *vnoise* does not suffer from the same issue
since it is written in pure Python.

## Is *vnoise* slow?

For scalar input (e.g. when computing noise values one at a time), yes (~300-2000x slower depending on the conditions).

*vnoise* deals with this by offering a vectorized API to compute large numbers of noise values in a single call. Since is
uses Numpy internally, its performance can match or exceed the original library (0.3-4x faster depending on the conditions).

## Installing

```
$ pip install vnoise
```

## Using *vnoise*

Basic example:

```python
>>> import vnoise
>>> noise = vnoise.Noise()
>>> noise.noise1(0.5)
0.0
>>> noise.noise1(0.1)
0.09144000000000001
>>> noise.noise2(0.1, 0.3)
0.09046282464000001
>>> noise.noise3(0.1, 0.3, 0.7)
0.27788822071249925
```

The `noiseX()` functions also accept sequences as arguments:

```python
>>> import numpy as np
>>> noise.noise2([0.1, 0.2, 0.3], np.linspace(0.5, 0.8, 10), grid_mode=True)
array([[0.0893    , 0.08919374, 0.08912713, 0.08910291, 0.08912241,
        0.08918551, 0.08929079, 0.08943552, 0.08961588, 0.08982716],
       [0.1276    , 0.126881  , 0.12643032, 0.12626645, 0.12639833,
        0.12682535, 0.12753768, 0.12851697, 0.12973738, 0.13116695],
       [0.09615   , 0.09412557, 0.09285663, 0.09239525, 0.09276657,
        0.09396889, 0.09597453, 0.09873182, 0.10216802, 0.10619312]])
```

With `grid_mode=True` (default value), a noise value is computed for every combination of the input value. In this case, the length of 
the first input is 3, and the length of the second input is 10. The result is an array with shape `(3, 10)`.

If `grid_mode=False`, all the input must have the same length, and the result has the same shape: 

```python
>>> noise.noise2(np.linspace(0.1, 0.3, 30), np.linspace(10, 10.5, 30), grid_mode=False)
array([0.099144  , 0.12303124, 0.14685425, 0.17057388, 0.194133  ,
       0.21745892, 0.24046583, 0.26305716, 0.28512793, 0.30656706,
       0.32725957, 0.34708877, 0.36593838, 0.38369451, 0.40024759,
       0.41549419, 0.42933872, 0.44169499, 0.45248762, 0.46165335,
       0.46914216, 0.4749182 , 0.47896062, 0.48126415, 0.48183955,
       0.4807138 , 0.47793023, 0.47354831, 0.46764335, 0.460306  ])
```

A seed value can be specified when creating the `Noise` class or afterward using the `seed()` function:

```python
>>> noise = vnoise.Noise(4)
>>> noise.seed(5)
```

## License

This code is available under the MIT license, see [LICENSE](LICENSE).


## Acknowledgments

This code is based on Casey Duncan's [noise](https://github.com/caseman/noise) library. The port was done with the help of [@tatarize](https://github.com/tatarize)