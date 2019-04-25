# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np
from scipy.ndimage import convolve1d
from matplotlib import pyplot as plt
from matplotlib import animation

np.random.seed(100)     # for reproducibility



LAP_1D_STENCIL = np.array([1., -4., 1.])

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
    return convolve1d(a, LAP_1D_STENCIL, output=o, mode='constant') / dx**2



class Simulation:
    def __init__(self, dV, grid, dtau, a, m=1., h=1e-6, x0=0., eq=1000):
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

        Additionally the array `L` is provided to store the currently used
        Laplacian; it is used for both `Lap(x)` and `Lap(xh)`. This is done to
        prevent unnecessary allocations of memory.

        The arrays `x_sum`, `xh_sum` are used for computing the average in tau
        after the simulation is done.
        """
        self.grid = grid
        self.h = h
        self.params = dict(
            dV  =   dV,
            dtau=   dtau,
            a   =   a,
            m   =   m,
            x0  =   x0,
        )
        self.arrays = dict(
            x   =   np.zeros(grid),
            xh  =   np.zeros(grid),
            L   =   np.zeros(grid),
        )
        self.x_sum  = np.zeros(grid)
        self.xh_sum = np.zeros(grid)
        self.steps  = 0
        # TODO: remove later
        self.eq = eq
        self.x_sum_eq  = np.zeros(grid)
        self.xh_sum_eq = np.zeros(grid)
        self.steps_eq = 0


    def step(self):
        """Performing one tau-step """
        # pull parameters and arrays into local scope to prevent littering the
        # code with `self`
        dV, x0, dtau, a, m = self.params.values()
        # arrays are passed as references, so `self.x`, etc. are updated
        # automatically
        x, xh, L = self.arrays.values()

        # create noise
        R = sqrt(2*dtau/a) * np.random.normal(0., 1., self.grid)

        # advancing x
        L = Lap(x, L, a)
        x = x + m*dtau*(L - dV(x, x0)) + R

        # advancing xh
        L = Lap(xh, L, a)
        xh = xh + m*dtau*(L - dV(xh, x0)) + R

        # fix source in xh
        xh[0] += dtau*self.h

        # update sums and number of steps
        # TODO: wait for equilibrium?
        # TODO: adjust tau?
        self.x_sum += x
        self.xh_sum += xh

        # TODO: remove later
        if self.steps > self.eq:
            self.x_sum_eq += x
            self.xh_sum_eq += xh
            self.steps_eq += 1

        self.steps += 1


    def multistep(self, n):
        """perform n steps with one call"""
        for _ in range(n):
            self.step()



def compute_slope(x, dx):
    return (x[1:] - x[:-1]) / dx


def dV_parabolic(x, *unused):
    return x


def init():
    avg_line.set_data([], [])
    cor_line.set_data([], [])
    slope_line.set_data([], [])
    avg_line_eq.set_data([], [])
    cor_line_eq.set_data([], [])
    slope_line_eq.set_data([], [])
    return avg_line, cor_line, slope_line, avg_line_eq, cor_line_eq, slope_line_eq

def animate(i):
    sim.multistep(2000)

    avg = sim.x_sum / sim.steps
    cor = (sim.xh_sum - sim.x_sum) / sim.steps / sim.h
    log_slope = compute_slope(np.log(cor), sim.params['a'])

    avg_eq = sim.x_sum_eq / sim.steps_eq
    cor_eq = (sim.xh_sum_eq - sim.x_sum_eq) / sim.steps_eq / sim.h
    log_slope_eq = compute_slope(np.log(cor_eq), sim.params['a'])

    avg_line.set_data(grid, avg)
    cor_line.set_data(grid, cor)
    slope_line.set_data(grid[1:], log_slope)
    avg_line_eq.set_data(grid, avg_eq)
    cor_line_eq.set_data(grid, cor_eq)
    slope_line_eq.set_data(grid[1:], log_slope_eq)

    print(sim.steps)

    return avg_line, cor_line, slope_line, avg_line_eq, cor_line_eq, slope_line_eq



dt = 0.1
grid = np.arange(0, 128*dt, dt)
parabolic_kwds = dict(
    dV  = dV_parabolic,
    grid= grid.shape,
    dtau= 0.1,
    a   = dt,
)
sim = Simulation(**parabolic_kwds)


fig = plt.figure()
(a, c, s), (ae, ce, se) = fig.subplots(2, 3, sharex=True)

for ax, ti, y in zip([a, c, s, ae, ce, se],
                     ["avg", "cor", "slope", "avg-eq", "cor-eq", "slope-eq"],
                     [0.01, 0.1, 2, 0.01, 0.1, 2]
                    ):
    ax.set_title(ti)
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(-y, y)

avg_line,       = a.plot([], [])
cor_line,       = c.plot([], [])
slope_line,     = s.plot([], [])
avg_line_eq,    = ae.plot([], [])
cor_line_eq,    = ce.plot([], [])
slope_line_eq,  = se.plot([], [])


#  FFWriter = animation.FFMpegWriter(fps=10)
anim_obj = animation.FuncAnimation(fig, animate, init_func=init, blit=True)


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
