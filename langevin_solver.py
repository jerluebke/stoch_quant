# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np
from scipy.ndimage import convolve1d


LAP_1D_STENCIL = np.array([1., -4., 1.])

def Lap(a, o, dx):
    return convolve1d(a, LAP_1D_STENCIL, output=o, mode='constant') / dx**2


np.random.seed(100)     # for reproducibility

#  dtau
#  a
#  dS
#  h


class Simulation:
    def __init__(self, dV, x0, grid, dtau, a, m, h):
        """Initializing the Simulation

        Params
        =====
        dV      :   callable dV(x, x0), derivative of potential
        x0      :   shape parameter for potential
        grid    :   tuple (Nt, Nx_1, ..., Nx_n), number of lattice points for
                     each dimension
        dtau    :   time step in Langevin time
        a       :   lattice spacing
        m       :   mass
        h       :   source strength (Parisi trick)


        The simulation state is encoded in the arrays `x` and `xh` and the
        integer `steps`.

        Additionally the array `L` is provided to store the currently used
        Laplacian; it is used for both `Lap(x)` and `Lap(xh)`. This is done to
        prevent unnecessary allocations of memory.

        The arrays `x_sum`, `xh_sum` are used for computing the average in tau
        after the simulation is done.
        """
        self.grid = grid
        self.params = dict(
            dV  =   dV
            x0  =   x0
            dtau=   dtau
            a   =   a
            m   =   m
            h   =   h
        )
        self.arrays = dict(
            x   =   np.zeros(grid),
            xh  =   np.zeros(grid),
            L   =   np.zeros(grid),
        )
        self.x_sum  = np.zeros(grid)
        self.xh_sum = np.zeros(grid)
        self.steps  = 0


    def step(self):
        """Performing one tau-step """
        # pull parameters and arrays into local scope to prevent littering the
        # code with `self`
        dV, x0, dtau, a, m, h = self.params.values()
        # arrays are passed as references, so `self.x`, etc. are updated
        # automatically
        x, xh, Lx, Lxh = self.arrays.values()

        # create noise
        R = sqrt(2*dtau/a) * np.random.normal(0., 1., self.grid)

        # advancing x
        L = Lap(x, L, a)
        x = x + m*dtau*(L - dV(x, x0)) + R

        # advancing xh
        L = Lap(xh, L, a)
        xh = xh + m*dtau*(L - omega_sq*x) + R

        # fix source in xh
        xh[0] += dtau*h

        # update sums and number of steps
        # TODO: wait for equilibrium?
        self.x_sum += x
        self.xh_sum += xh
        self.step += 1



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
