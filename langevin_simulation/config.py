# -*- coding: utf-8 -*-
"""Simulation and plotting presets"""

from potentials import *


# number of points on time grid
Nt = 128


SIM_KWDS = dict(
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
        #  x0  = .1,
    ),
    double_well = dict(
        #  dV = dV_double_well,
        dV = dV_double_well_2,
        #  dV = dV_double_well_3,
        #  a   = 0.05,
        dtau= 1e-3,
        #  x0 = 1.6,
        #  x0 = .2
        x0 = {'h' : 1., 'a' : 1.}
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
        x0 = 1.
    ),
)


PLT_KWDS = dict(
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


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai : 
