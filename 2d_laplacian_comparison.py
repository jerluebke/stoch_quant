# -*- coding: utf-8 -*-

from multiprocessing.managers import Namespace
import numpy as np
from scipy import signal, ndimage

MAX, MIN = -5, 5
SIZES = [100, 1_000, 10_000, 100_00, 1_000_000]
ITERATIONS = 1000
HEAD = "grid size: %d\nname\t\t|\ttime per run (sec)\t|max error\n------"
RES = "%s\t|\t%s\t\t|\t%s"
DELIM = "\n\n=======\n\n"


def f(x, y):
    """f(x) = e^(-x^2 -y^2)"""
    return np.exp(- x**2 - y**2)

def ddf(x, y, fxy):
    """Lap f(x) = 4*f(x, y)*(x^2 + y^2 - 1)"""
    return 4 * fxy * (x**2 + y**2 - 1)



def my_lap2d(a):
    """custom implementation for a 2d laplacian of an array `a`
    for reference: https://stackoverflow.com/a/4699973"""
    r, c = a.shape
    dx = np.ones((1, c-1))
    dy = np.ones((r-1, 1))
    L = np.zeros_like(a)

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


FUNCS = {
    "my 2d laplacian"   :   my_lap2d,
    "signal.convolve2d" :   lambda a: signal.convolve2d(a, STENCIL, mode='same'),
    "ndimage.convolve"  :   lambda a: ndimage.convolve(a, STENCIL, mode='constant'),
    "ndimage.laplace"   :   lambda a: ndimage.laplace(a, mode='constant')
}


def main():
    for n in SIZES:
        step = (MAX-MIN) / n
        X, Y = np.meshgrid[MAX:MIN:step]
        fxy = f(X, Y)
        lap_ana = ddf(X, Y, fxy)

        print(HEAD % n)

        for name, func in FUNCS:
            t = timeit('f(a)', number=ITERATIONS,
                       globals=Namespace(a=fxy, f=func)) / ITERATIONS
            lap_num = func(fxy) / step**2
            err = np.max(np.abs(lap_num-lap_ana))

            print(RES % (name, t, err))

        print(DELIM)


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
