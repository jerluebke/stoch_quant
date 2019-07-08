#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from langevin_solver import Simulation
from config import Nt, SIM_KWDS, PLT_KWDS


# for reproducibility
np.random.seed(100)


#####################
# CHOOSE POTENTIAL  #
#####################

#  POTENTIAL = 'parabolic'
#  POTENTIAL = 'cosh'
#  POTENTIAL = 'quartic'
POTENTIAL = 'double_well'
#  POTENTIAL = 'parabolic_with_peak'
#  POTENTIAL = 'quartic_with_peak'
#  POTENTIAL = 'false_vacuum'
#  POTENTIAL = 'cosine'


#####################
# SET UP SIMULATION #
#####################

config_dict = {**SIM_KWDS['default'], **SIM_KWDS[POTENTIAL]}
sim         = Simulation(**config_dict)

# grid array for plotting, use given lattice spacing
dt          = config_dict['a']
grid        = np.arange(0, Nt*dt, dt)


# adjust initial conditions
# false vacuum potential: start in false vacuum, then tunnel to true vacuum
#  sim.arrays['x'] = -.5*np.ones(Nt)
#  sim.arrays['xh'] = -.5*np.ones(Nt)


#########################
# ANIMATION FUNCTIONS   #
#########################

def animate1():
    """animate three quantities in seperate axes:
        average
        correlation function
        log slope of correlation function

    returns created figure and animation object
    """
    fig = plt.figure()
    a, c, s = fig.subplots(1, 3)

    for ax, ti, ylim in zip([a, c, s],
                            ["avg", "cor", "slope"],
                            PLT_KWDS['ylim'][POTENTIAL]):
        ax.set_title(ti)
        ax.set_xlim(0, grid[-1])
        ax.set_ylim(ylim)

    avg_line,   = a.plot([], [])
    cor_line,   = c.plot([], [])
    slope_line, = s.plot([], [])


    def anim_step(i):
        sim.multistep(1000)

        avg_line.set_data(grid, sim.x_average)
        cor_line.set_data(grid, sim.x0_x_correlation)
        log_slope = sim.log_slope
        slope_line.set_data(grid[1:], log_slope)

        # TODO: give theoretical result and error, if available
        print('%f\t|%d' % (-log_slope[1], sim.steps))

        return avg_line, cor_line, slope_line


    print('Energy\t\t|Steps\n======\t\t======')
    anim_obj = animation.FuncAnimation(fig, anim_step,
                                       blit=True, repeat=False)

    return fig, anim_obj



def animate2():
    """animate long term average over total time and short term average after
    every multistep2 in one common ax

    returns created figure and animation object
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(PLT_KWDS['ylim'][POTENTIAL][0])
    ax.set(title='average', xlabel='time', ylabel='x')

    avg1_line, = ax.plot([], [])
    avg2_line, = ax.plot([], [])

    def anim_step(i):
        avg2 = sim.multistep2(1000)
        avg1_line.set_data(grid, sim.x_average)
        avg2_line.set_data(grid, avg2[...,0])
        print('steps = %d' % sim.steps)
        return avg1_line, avg2_line

    anim_obj  = animation.FuncAnimation(fig, anim_step,
                                        blit=True, repeat=False)

    return fig, anim_obj



def animate3():
    """animate long term average over total time and short term average after
    every multistep2 in one common ax

    returns created figure and animation object
    """
    fig, (ax_avg, ax_hist) = plt.subplots(1,2)
    ax_avg.set_xlim(0, grid[-1])
    ax_avg.set_ylim(PLT_KWDS['ylim'][POTENTIAL][0])
    ax_avg.set(title='average', xlabel='time', ylabel='x')
    ax_hist.set_ylim(-3, 3)
    ax_hist.set(title='history', xlabel='tau', ylabel='x')

    avg1_line, = ax_avg.plot([], [])
    avg2_line, = ax_avg.plot([], [])
    line, = ax_hist.plot([], [])

    steps = 1000
    t = np.arange(steps)
    x = np.zeros(steps)

    def anim_step(i):
        avg2 = sim.multistep2(1000)

        avg1_line.set_data(grid, sim.x_average)
        avg2_line.set_data(grid, avg2[...,0])

        x[i] = avg2[0,0]
        line.set_data(t[:i], x[:i])
        ax_hist.set_xlim(-1,i+1)

        return avg1_line, avg2_line, line

    #  https://stackoverflow.com/a/14421998
    anim_obj  = animation.FuncAnimation(fig, anim_step, frames=steps,
                                        blit=False, repeat=False)

    return fig, anim_obj



def count_transitions_for_varying_heights():
    heights = np.arange(0.1, 8., 0.5)
    jumps   = np.zeros_like(heights)

    for h in range(heights.size):
        print('counting for height = %f' % heights[h])
        config_dict['x0'][0] = heights[h]
        sim = Simulation(**config_dict)

        old = sim.multistep2(100)[64,0]
        for i in range(1000):
            current = sim.multistep2(1000)[64,0]
            if old * current < 0:
                jumps[h] += 1
            old = current

    plt.plot(heights,jumps)



if __name__ == '__main__':
    #  f, a = animate1()
    #  f, a = animate2()
    #  f, a = animate3()
    count_transitions_for_varying_heights()

    plt.show()


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
