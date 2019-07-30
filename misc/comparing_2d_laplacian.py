#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from timeit import timeit
import numpy as np
from scipy import signal, ndimage
from numba import njit, stencil

MAX, MIN = -5, 5
SIZES = [
    #  100,
    #  1_000,
    #  10_000,
    #  100_00,
    #  1_000_000
    512,
    1024,
    2048,
    4096
]
STENCIL = np.array([
    [0., 1., 0.],
    [1.,-4., 1.],
    [0., 1., 0.]
])
ITERATIONS = 10
START = "======\n\nComparing implementations to compute the 2D-Laplacian\n\n\t%s\n\t%s\n\n======\n\n"
HEAD = "grid size: %d\n|name\t\t\t|time per run (sec)\t|max error\n|------"
RES = "|%s\t|%e\t\t|%e"
DELIM = "\n\n=======\n\n"


def f(x, y):
    """f(x) = e^(-x^2 -y^2)"""
    return np.exp(- x**2 - y**2)

def ddf(x, y, fxy):
    """Lap f(x) = 4*f(x, y)*(x^2 + y^2 - 1)"""
    return 4 * fxy * (x**2 + y**2 - 1)



def my_lap2d(a, o=None):
    """custom implementation for a 2d laplacian of an array `a`
    for reference: https://stackoverflow.com/a/4699973"""
    r, c = a.shape
    if o is None:
        L = np.zeros_like(a)
    else:
        L = o

    # x component
    # left and right boundaries
    L[:,0] = a[:,0] - 2*a[:,1] + a[:,2]
    L[:,c-1] = a[:,c-3] - 2*a[:,c-2] + a[:,c-1]
    # interior
    L[:,1:c-1] = a[:,2:c] - 2*a[:,1:c-1] + a[:,0:c-2]

    # y component
    # left and right boundaries
    L[0,:] = L[0,:] + a[0,:] - 2*a[1,:] + a[2,:]
    L[r-1,:] = L[r-1,:] + a[r-3,:] - 2*a[r-2,:] + a[r-1,:]
    # interior
    L[1:r-1,:] = L[1:r-1,:] + a[2:r,:] - 2*a[1:r-1,:] + a[0:r-2,:]

    return L


@stencil
def _numba_lap_stencil(a):
    return a[-1,0] + a[0,-1] - 4*a[0,0] + a[1,0] + a[0,1]

@njit
def numba_lap_stencil(a, o):
    _numba_lap_stencil(a, out=o)
    return o


FUNCS = {
    "my 2d laplacian "  :   my_lap2d,
    #  "signal.convolve2d" :
    #      lambda a, o: signal.convolve2d(a, STENCIL, mode='same'),
    "ndimage.convolve"  :
        lambda a, o: ndimage.convolve(a, STENCIL, mode='constant', output=o),
    #  "ndimage.laplace " :
    #      lambda a, o: ndimage.laplace(a, mode='constant', output=o),
    "numba stencil\t"   :
        #  lambda a, o: _numba_lap_stencil(a, out=o)
        numba_lap_stencil,
    "numpy roll\t"      :
        lambda a, o: np.roll(a, -1, 0) + np.roll(a, 1, 0) \
                   + np.roll(a, -1, 1) + np.roll(a, 1, 1) \
                   - 4*a
}


def main():
    print(START % (f.__doc__, ddf.__doc__))

    for n in SIZES:
        step = (MAX-MIN) / n
        X, Y = np.mgrid[MIN:MAX:step,MIN:MAX:step]
        fxy = f(X, Y)
        lap_ana = ddf(X, Y, fxy)
        lap_num = np.zeros_like(fxy)

        print(HEAD % n)

        for name, func in FUNCS.items():
            t = timeit('f(a, o)', number=ITERATIONS,
                       setup='f(a, o)', # run it once to make sure it is compiled
                       globals=dict(a=fxy, f=func, o=lap_num)) / ITERATIONS
            lap_num = func(fxy, lap_num) / step**2
            err = np.max(np.abs(lap_num-lap_ana))

            print(RES % (name, t, err))

        print(DELIM)


if __name__ == "__main__":
    main()


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
