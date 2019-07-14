# -*- coding: utf-8 -*-
"""Some predefined potentials given as their derivatives to be passed to
the Langevin Simulation class

The generic interface:
    dV(x, x0)
where x is the array which gives the current state of the system, and x0 is
usually a shape parameter (which has different names in the function
definitions below, according to the parameters name in literatur)
"""

import numpy as np


def dV_parabolic(x, a):
    """ a/2 * x**2 """
    return a*x

def dV_cosh(x, b):
    """ -V0 / cosh(b*x)**2 """
    V0 = 1.
    return -2*b*V0*np.tanh(b*x) / np.cosh(b*x)**2

def dV_double_well(x, a):
    """ (x**2 - a**2)**2 / 4 """
    return x**3 - x*a**2

def dV_double_well_2(x, kwargs):
    """ h * (x**2 - a**2)**2 """
    h = kwargs['h']
    a = kwargs['a']
    return 4. * h * (x**3 - x*a**2)

def dV_double_well_3(x, E_0):
    """ E_0 * (C / 4 * x**4 - x**2) """
    C = 0.18
    return E_0 * (C*x**3 - 2.*x)

def dV_quartic(x, mu):
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

def dV_cosine(x, a):
    """ V = 1 - a*cos(x) """
    return a*np.sin(x)


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai : 
