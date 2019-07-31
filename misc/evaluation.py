#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as so


# DATA
heights = np.array([
    .5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5,
    5., 6., 7., 8., 9., 10., 12., 14., 16.
])
h_arr   = np.linspace(0, 16.5)
jumps   = np.load('../langevin_simulation/results/all-results-a1.npy')
mean    = jumps.mean(axis=1)
std     = jumps.std(axis=1)


# MODEL
def lin_func(B, x):
    return B[0] * x + B[1]

model   = so.Model(lin_func)
data    = so.RealData(heights, np.log(mean), sy=std/mean)
odr     = so.ODR(data, model, beta0=[1., 1.])
out     = odr.run()


# PLOTTING
errorbar_config = dict(ecolor='tab:gray', capsize=2.5, ls='none', lw=1.2,
                       marker='o', ms=4, mec='tab:red', mfc='tab:red')

plt.figure(figsize=(5, 3.75), dpi=300)
plt.subplot(111)

plt.errorbar(heights, mean, yerr=std, label='data', **errorbar_config)
plt.plot(h_arr, np.exp(out.beta[1] + h_arr*out.beta[0]), c='tab:blue',
         label='linear fit, with\n$m=(%.3f\\pm%.3f$)\n$b=(%.3f\\pm%.3f)$'
                % (out.beta[0], out.sd_beta[0], out.beta[1], out.sd_beta[1]))

plt.yscale('log')
plt.grid(False)
plt.tick_params(which='both', direction='out', top=False, right=False)
plt.xlim(0, 16.5)
plt.ylim(10, 2000)
plt.xlabel('$h$ / a.u.')
plt.ylabel('$N(h)$')
plt.legend(loc='upper right')
plt.savefig('transitions.pdf', dpi=300, bbox_inches='tight')

#  plt.show()


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
