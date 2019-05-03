# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np
from scipy.ndimage import convolve1d
from matplotlib import pyplot as plt
from matplotlib import animation
from collections import OrderedDict

np.random.seed(100)     # for reproducibility


POTENTIAL = 'parabolic'
#  POTENTIAL = 'cosh'
#  POTENTIAL = 'quartic'
#  POTENTIAL = 'double_well'
Nt = 128


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
    """
    convolve1d(a, LAP_1D_STENCIL, output=o, mode='wrap')
    return np.divide(o, dx**2, out=o)



class Simulation:
    def __init__(self, dV, grid, dtau, a, m=1., h=1e-6, x0=0.):
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
        # TODO: wait for equilibrium?
        self.x_sum += x
        self.xh_sum += xh
        self.steps += 1


    def multistep(self, n):
        """perform n steps with one call"""
        for _ in range(n):
            self.step()


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



def dV_parabolic(x, *unused):
    return x

def dV_cosh(x, b):
    V0 = 1.
    return 2*b*V0*np.tanh(b*x) / np.cosh(b*x)**2

def dV_double_well(x, x0):
    #  return (x**3/x0**2 - x) / 2.
    return 4*(x**3 - x*x0**2)

def dV_quartic(x, mu):
    return 4*mu*x**3



def init():
    avg_line.set_data([], [])
    cor_line.set_data([], [])
    slope_line.set_data([], [])
    return avg_line, cor_line, slope_line


def animate(i):
    sim.multistep(1000)

    avg_line.set_data(grid, sim.x_average)
    cor_line.set_data(grid, sim.x0_x_correlation)
    log_slope = sim.log_slope
    slope_line.set_data(grid[1:], log_slope)

    # TODO: give theoretical result and error, if available
    print('%f\t|%d' % (-log_slope[1], sim.steps))

    return avg_line, cor_line, slope_line



sim_kwds = dict(
    default = dict(
        grid= (Nt,),
        dtau= 0.1,
        a   = 0.1,
    ),
    parabolic = dict(
        dV  = dV_parabolic,
    ),
    cosh = dict(
        dV  = dV_cosh,
        x0  = 0.1,
    ),
    double_well = dict(
        dV  = dV_double_well,
        x0  = 4.,
    ),
    quartic = dict(
        dV  = dV_quartic,
        a   = 1.,   # overwriting default
        x0  = 1.,
    ),
)

plt_kwds = dict(
    ylim = dict(
        parabolic   = [(-1.5, 1.5), (0., 0.1), (-2., 2.)],
        cosh        = [(-1.5, 0.5), (0., 1.), (-0.5, 0.5)],
        double_well = [(-5., 5.), (0., 0.02), (-5., 5.)],
        quartic     = [(-2., 2.), (0., 0.1), (-2., 2.)],
    )
)


config_dict = {**sim_kwds['default'], **sim_kwds[POTENTIAL]}
dt          = config_dict['a']
grid        = np.arange(0, Nt*dt, dt)
sim         = Simulation(**config_dict)


fig = plt.figure()
a, c, s = fig.subplots(1, 3)

for ax, ti, ylim in zip([a, c, s],
                        ["avg", "cor", "slope"],
                        plt_kwds['ylim'][POTENTIAL]):
    ax.set_title(ti)
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(ylim)

avg_line,       = a.plot([], [])
cor_line,       = c.plot([], [])
slope_line,     = s.plot([], [])


#  FFWriter = animation.FFMpegWriter(fps=10)
print('Energy\t\t|Steps\n======\t\t======')
anim_obj = animation.FuncAnimation(fig, animate, init_func=init, blit=True)


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
