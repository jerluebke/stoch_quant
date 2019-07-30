#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import diags, spdiags, linalg


Delta = '\N{GREEK CAPITAL LETTER DELTA}'
_2 = '\N{SUBSCRIPT TWO}'
DeltaE_2 = Delta + 'E' + _2

h, a = 1., 1.

x = np.linspace(-32, 32, 1024)
dx = x[1] - x[0]
Lap = diags([1, -2, 1], [-1, 0, 1], shape=(1024, 1024)) / dx**2
V = spdiags(h*(x**2 - a**2)**2, 0, 1024, 1024)
H = -Lap/2. + V

evals, evecs = linalg.eigsh(H, which='SM')

print(DeltaE_2 + ' = %.3f' % (evals[1] - evals[0]))


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
