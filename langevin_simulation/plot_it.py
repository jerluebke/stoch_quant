#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, time
from multiprocessing import Pool, current_process

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
        avg2 = sim.multistep2(100)
        avg1_line.set_data(grid, sim.x_average)
        avg2_line.set_data(grid, avg2)
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



def count_transitions_for_varying_heights(heights,
                                          steps=10_000,
                                          ms_steps=100,
                                          width=1.,
                                          sample=64,
                                          proc_name='none'):
    jumps = np.zeros_like(heights)
    config_dict['x0']['a'] = width

    for h in range(heights.size):
        config_dict['x0']['h'] = heights[h]
        sim = Simulation(**config_dict)
        print('[%(p)s] - counting for height = %(h).1f, width = %(a).1f' \
              % {'p' : proc_name, **sim.params['x0']})

        old = sim.multistep2(ms_steps)[sample]
        for i in range(steps):
            current = sim.multistep2(ms_steps)[sample]
            if old * current < 0:
                jumps[h] += 1
            old = current

    return jumps



def worker(idx, file_name, heights, **kwargs):
    proc_name = current_process().name
    print('[%s] - starting worker %d' % (proc_name, idx))
    np.random.seed(None)    # seed from device entropy
    jumps = count_transitions_for_varying_heights(heights, proc_name=proc_name,
                                                 **kwargs)
    np.save('%s-%d' % (file_name, idx), jumps)
    print('[%s] - closing worker %d' % (proc_name, idx))



if __name__ == '__main__':
    #  f, a = animate1()
    #  f, a = animate2()
    #  f, a = animate3()
    #  plt.show()
    #  sys.exit(0)

    PROCESSES   = 4
    NO_OF_RUNS  = 8
    HEIGHTS     = np.array([.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 7.,
                            8., 9., 10., 12., 14., 16.])
    MEASUREMENTS_CONFIG = dict(steps=10_000, ms_steps=100, width=1.6, sample=64)


    TEST = False

    if TEST:
        jumps = count_transitions_for_varying_heights(
            HEIGHTS, **MEASUREMENTS_CONFIG)
        plt.plot(HEIGHTS, jumps)
        plt.show()


    else:
        file_name = input('save data in: ')

        start = time.time()

        with Pool(PROCESSES) as pool:
            res = [pool.apply_async(worker,
                                    (idx, file_name, HEIGHTS),
                                    MEASUREMENTS_CONFIG)
                   for idx in range(NO_OF_RUNS)]
            for r in res:
                r.wait()

        exec_time = time.time() - start
        print('\nexecution time: %f seconds' % exec_time)



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
