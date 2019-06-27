# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np
from scipy.ndimage import convolve1d
from collections import OrderedDict


LAP_1D_STENCIL = np.array([1., -2., 1.])

def Lap(a, o, dx):
    """Laplacian

    Params
    ======
    a   :   input array
    o   :   output array
    dx  :   step width

    Compute the Laplacian of a given array by convolving it with a stencil and
    dividing it by the square of the step width.

    The boundaries are filled with zeros.
    """                                         #============#
    convolve1d(a, LAP_1D_STENCIL, output=o,     # boundaries #
                                                #============#
               #  mode='constant',                 # constant = 0
               #  mode='wrap'                      # periodic
               mode='nearest'                   # continued
              )
    return np.divide(o, dx**2, out=o)



class Simulation:

    def __init__(self, dV, grid, dtau, a, m=1., h=1e-6, x0=1.):
        """Initializing the Simulation

        Params
        =====
        dV      :   callable dV(x, x0), derivative of potential
        grid    :   tuple (Nt, Nx_1, ..., Nx_n), number of lattice points for
                     each dimension
        dtau    :   time step in Langevin time
        a       :   lattice spacing
        m       :   mass
        h       :   source strength (Parisi trick)
        x0      :   shape parameter for potential


        The simulation state is encoded in the arrays `x` and `xh` and the
        integer `steps`.

        Additionally the arrays `L` and `Lh` are provided to store the
        Laplacian of `x` and `xh` respectivly.

        `xold` is used to compare `x` after each step to check for stability
        and rescale dtau if necessary.

        The arrays `x_sum`, `xh_sum` are used for computing the average in tau
        after the simulation is done.
        """
        self.grid = grid
        self.params = OrderedDict(
            dV  =   dV,
            dtau=   dtau,
            a   =   a,
            m   =   m,
            h   =   h,
            x0  =   x0,
        )
        self.arrays = OrderedDict(
            x   =   np.zeros(grid),
            xold=   np.zeros(grid),
            xh  =   np.zeros(grid),
            L   =   np.zeros(grid),
            Lh  =   np.zeros(grid),
        )
        self.x_sum  = np.zeros(grid)
        self.xh_sum = np.zeros(grid)
        self.steps  = 0


    def step(self):
        """Performing one tau-step """
        # pull parameters and arrays into local scope to prevent littering the
        # code with `self`
        dV, dtau, a, m, h, x0 = self.params.values()

        # arrays are passed as references, so `self.x`, etc. are updated
        # automatically
        x, xold, xh, L, Lh = self.arrays.values()

        # make copy of current x for stability check
        np.copyto(xold, x)

        while True:
            # create noise
            R = sqrt(2*dtau/a) * np.random.normal(0., 1., self.grid)

            # advancing x
            Lap(xold, L, a)
            x[:] = xold + dtau * m * L - dtau * dV(xold, x0) + R

            # rescaling dtau for stability
            # get max value and its coordinates of new x
            xmax = np.max(np.abs(x))
            ind = np.unravel_index(x.argmax(), x.shape)

            # difference by which x is changed (without noise) must not be
            # larger than the max value of x (as an estimate for stability)
            if (x - xold - R)[ind] > xmax:
                dtau *= 0.95
                print('new dtau = %e' % dtau)
                self.params['dtau'] = dtau
            else:
                break

        # advancing xh
        Lap(xh, Lh, a)
        xh += dtau * m * Lh - dtau * dV(xh, x0) + R

        # fix source in xh
        xh[0] += dtau * h

        # update sums and number of steps
        self.x_sum  += x
        self.xh_sum += xh
        self.steps  += 1


    def multistep(self, n):
        """perform n steps with one call"""
        for _ in range(n):
            self.step()


    def multistep2(self, n):
        """perform n steps and compute the average"""
        x_sum_2 = np.zeros((*self.grid, 2))
        for _ in range(n):
            self.step()
            x_sum_2[...,0] += self.arrays['x']
            x_sum_2[...,1] += self.arrays['xh']
        return x_sum_2 / n


    @staticmethod
    def _compute_slope(x, dx):
        return (x[1:] - x[:-1]) / dx

    @property
    def x_average(self):
        return self.x_sum / self.steps

    @property
    def x0_x_correlation(self):
        return (self.xh_sum - self.x_sum) / self.steps / self.params['h']

    @property
    def log_slope(self):
        return self._compute_slope(np.log(self.x0_x_correlation),
                                   self.params['a'])



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
