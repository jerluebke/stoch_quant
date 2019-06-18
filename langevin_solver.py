# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np
from scipy.ndimage import convolve, convolve1d
from matplotlib import pyplot as plt
from matplotlib import animation
from collections import OrderedDict

np.random.seed(100)     # for reproducibility


#  POTENTIAL = 'parabolic'
#  POTENTIAL = 'cosh'
#  POTENTIAL = 'quartic'
POTENTIAL = 'double_well'
#  POTENTIAL = 'parabolic_with_peak'
#  POTENTIAL = 'quartic_with_peak'
#  POTENTIAL = 'false_vacuum'
#  POTENTIAL = 'cosine'
Nt = 128

# fixed start and end
X0, X1 = .04, -.04
#  X0, X1 = 2*np.pi*0.01, -2*np.pi*0.01


LAP_1D_STENCIL = np.array([1., -2., 1.])
LAP_3D_STENCIL = np.array((
    [[0., 0., 0.],
     [0., 1., 0.],
     [0., 0., 0.]],
    [[0., 1., 0.],
     [1.,-6., 1.],
     [0., 1., 0.]],
    [[0., 0., 0.],
     [0., 1., 0.],
     [0., 0., 0.]]
))


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
    #  convolve1d(a, LAP_1D_STENCIL, output=o,     # boundaries #
                                                #============#
    convolve(a, LAP_3D_STENCIL, output=o,
               #  mode='constant',                 # constant = 0
               mode='wrap'                      # periodic
               #  mode='nearest'                   # continued
              )
    #  o[0] = X0                                   # fixed by X0, X1
    #  o[-1] = X1
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
            # 1d quantum mechanics
            #  x[:] = xold + dtau * m * L - dtau * dV(xold, x0) + R
            # free field (Klein-Gordon)
            #  x[:] = xold + dtau * (L - m * xold) + R
            # quartic interaction
            x[:] = xold + dtau * (L - m * xold - x0 * xold**3 / 6.) + R

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
        #  xh += dtau * m * Lh - dtau * dV(xh, x0) + R
        #  xh += dtau * (L - m * xh) + R
        xh += dtau * (L - m * xh - x0 * xh**3 / 6.) + R

        # fix source in xh
        #  xh[0] += dtau * h
        xh[...,0][0, 0] += dtau * h

        # update sums and number of steps
        self.x_sum  += x
        self.xh_sum += xh
        self.steps  += 1


    def multistep(self, n):
        """perform n steps with one call"""
        for _ in range(n):
            self.step()


    def multistep2(self, n):
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



def dV_parabolic(x, a=1):
    """ a/2 * x**2 """
    return a*x

def dV_cosh(x, b=0.1):
    """ -V0 / cosh(b*x)**2 """
    V0 = 1.
    return -2*b*V0*np.tanh(b*x) / np.cosh(b*x)**2

def dV_double_well(x, a=4.):
    """ (x**2 - a**2) / 4 """
    return x**3 - x*a**2

def dV_quartic(x, mu=4.):
    """ V = mu/4 * x**4 """
    return mu*x**3

def dV_parabolic_with_peak(x, *unused):
    """ m/2 * x**2 + h * np.exp(-(x/sigma)**2) """
    m = 1.
    h = 10.
    sigma = 0.5
    return m*x - 2*x*h/sigma**2 * np.exp(-(x/sigma)**2)

def dV_quartic_with_peak(x, *unused):
    """ m/4 * x**4 + h * np.exp(-(x/sigma)**2) """
    m = 1.
    h = 10.
    sigma = 0.5
    return m*x**3 - 2*x*h/sigma**2 * np.exp(-(x/sigma)**2)

def dV_false_vacuum(x, *unused):
    """ 1/4 * (a**2 * x**2 - b**2)**2 - c*x + d """
    a = b = 3.
    c = 3.
    return a**4 * x**3 - a**2 * b**2 * x - c

def dV_cosine(x, *unused):
    """ V = 1 - a*cos(x) """
    a = 1.
    return a*np.sin(x)


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


def animate2(i):
    avg2 = sim.multistep2(1000)

    avg1_line.set_data(grid, sim.x_average)
    avg2_line.set_data(grid, avg2[...,0])

    print('steps = %d' % sim.steps)

    return avg1_line, avg2_line


def animate3(i):
    #  fsim.multistep(100)
    #  favg.set_data(fsim.x_average[0])
    favg.set_data(i)
    #  print('steps = %d' % fsim.steps)
    return favg,



sim_kwds = dict(
    default = dict(
        grid= (Nt,),
        dtau= 0.1,
        a   = 0.1,
    ),
    parabolic = dict(
        dV  = dV_parabolic,
        #  x0  = 0.5,
    ),
    quartic = dict(
        dV  = dV_quartic,
        x0  = 4e-3,
    ),
    cosh = dict(
        dV  = dV_cosh,
        x0  = 4.,
    ),
    double_well = dict(
        dV = dV_double_well,
        #  a   = 0.05,
        #  dtau= 1e-3,
        #  x0 = 1.6,
        x0 = 1.2,
    ),
    parabolic_with_peak = dict(
        dV = dV_parabolic_with_peak,
        a = 0.05,
    ),
    quartic_with_peak = dict(
        dV = dV_quartic_with_peak,
        a = 0.05,
    ),
    false_vacuum = dict(
        dV = dV_false_vacuum,
        #  a = 0.05,
    ),
    cosine = dict(
        dV = dV_cosine,
    ),
)

plt_kwds = dict(
    ylim = dict(
        parabolic   = [(-1.5, 1.5), (0., 0.1), (-2., 2.)],
        quartic     = [(-2., 2.), (0., 0.1), (-2., 2.)],
        cosh        = [(-1.0, 5.0), (0., 2.), (-0.5, 0.5)],
        double_well = [(-5., 5.), (0., 0.02), (-5., 5.)],
        parabolic_with_peak = [(-5., 5.), (0., 0.02), (-5., 5.)],
        quartic_with_peak   = [(-5., 5.), (0., 0.02), (-5., 5.)],
        false_vacuum= [(-5., 5.), (0., 0.02), (-5., 5.)],
        cosine      = [(-10., 4.), (0., 1.), (-5., 5.)],
    )
)


config_dict = {**sim_kwds['default'], **sim_kwds[POTENTIAL]}
dt          = config_dict['a']
grid        = np.arange(0, Nt*dt, dt)
sim         = Simulation(**config_dict)


fsim = Simulation(None, (64, 64, 64), dtau=1e-3, a=.1, m=-1., x0=1.)


# adjust initial conditions
# false vacuum potential: start in false vacuum, then tunnel to true vacuum
#  sim.arrays['x'] = -.5*np.ones(Nt)
#  sim.arrays['xh'] = -.5*np.ones(Nt)
# start at oposite locations (kink solution as initial condition), use with
# mode='nearest' in `convolve1d`
# line over whole grid from X0 to X1
#  sim.arrays['x'] = -4/grid[-1]*grid+2
#  sim.arrays['xh'] = -4/grid[-1]*grid+2
# only boundaries
#  sim.arrays['x'][0] = 4.
#  sim.arrays['x'][1] = -4.
#  sim.arrays['xh'][0] = 4.
#  sim.arrays['xh'][1] = -4.
#  sim.arrays['x'] = np.pi*np.ones(Nt)
#  sim.arrays['xh'] = np.pi*np.ones(Nt)


fig = plt.figure()
a, c, s = fig.subplots(1, 3)

for ax, ti, ylim in zip([a, c, s],
                        ["avg", "cor", "slope"],
                        plt_kwds['ylim'][POTENTIAL]):
    ax.set_title(ti)
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(ylim)

avg_line,   = a.plot([], [])
cor_line,   = c.plot([], [])
slope_line, = s.plot([], [])


FFWriter = animation.FFMpegWriter(fps=10)
#  print('Energy\t\t|Steps\n======\t\t======')
#  anim_obj = animation.FuncAnimation(fig, animate, init_func=init, blit=True)



fig2, ax2 = plt.subplots()
ax2.set_xlim(0, grid[-1])
ax2.set_ylim(plt_kwds['ylim'][POTENTIAL][0])

avg1_line,= ax2.plot([], [])
avg2_line,= ax2.plot([], [])
#  anim_obj  = animation.FuncAnimation(fig2, animate2, blit=True, repeat=False)



fig3, ax3 = plt.subplots()
ax3.grid(False)
favg = ax3.imshow(fsim.arrays['x'][0], animated=True)

frames = np.zeros((100, 64, 64))
for i in range(10):
    fsim.multistep(100)
    print('steps = %d' % fsim.steps)
    frames[i] = fsim.x_average[0]

anim = \
    animation.FuncAnimation(fig3, animate3, frames, blit=True,
                            repeat=False)
                            #.save('free-field.mp4', writer=FFWriter, dpi=100)


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
