# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

#  potential = "parabolic"
potential = "cosh"
#  potential = "double_well"
#  potential = "quartic"

#  plot_type = "average_x"
#  plot_type = "x0_xl_correlation"
plot_type = "log_slope"


class Simulation(object):
    def __init__(self, dV, x0, Nt, dt, dtau, h, m):
        # simulation parameters
        self.dV = dV
        self.x0 = x0
        self.Nt = Nt
        self.dt = dt
        self.dtau = dtau
        self.h = h
        self.m = m

        # simulation state
        self.x = np.zeros(Nt)
        self.x_sum = np.zeros(Nt)
        self.xh = np.zeros(Nt)
        self.xh_sum = np.zeros(Nt)
        self.steps = 0


    def step(self):
        # pull all of these variables into local scope to avoid
        # littering everything with 'self'
        x = self.x
        xh = self.xh
        dV = self.dV
        x0 = self.x0
        dt = self.dt
        dtau = self.dtau
        m = self.m
        h = self.h

        while True:
            dw = np.sqrt(2*dtau/dt) * np.random.randn(self.Nt)
            new_x = (x
                    + dtau * m*(np.roll(x, 1) - 2*x + np.roll(x, -1))/dt**2
                    - dtau * dV(x, x0)
                    + dw)
            new_xh = (xh
                    + dtau * m*(np.roll(xh, 1) - 2*xh + np.roll(xh, -1))/dt**2
                    - dtau * dV(xh, x0)
                    + dw)
            new_xh[0] += dtau * h
            #check if dtau needs to be rescaled for stability
            qmax = np.max(abs(new_x))
            ind = np.unravel_index(new_x.argmax(), new_x.shape)
            if (new_x-x-dw)[ind] > qmax:
                dtau *= 0.95
            else:
                break

        self.dtau = dtau # write back the dtau, since it may have been changed
        self.x = new_x
        self.xh = new_xh
        self.x_sum += new_x
        self.xh_sum += new_xh
        self.steps += 1


    def multistep(self, n):
        for i in range(n):
            self.step()



def dV_parabolic(x, x0):
    return x

def dV_cosh(x, b):
    b=0.1
    V0=1.
    return 2*b*V0/np.cosh(b*x)**2* np.tanh(b* x)

def dV_double_well(x, x0):
    return 1/2.* x* (-1. + x**2./x0**2.)

def dV_quartic(x,x0):
    mu=1.
    return 4*mu*x**3

def compute_slope(x, dt):
    return (np.roll(x, -1) - x) / dt

def init():
    line.set_data([], [])
    text.set_text("$E_1$")
    return line, text

def animate(i):
    simulation.multistep(1000)
    x0_xl_correlation = (simulation.xh_sum-simulation.x_sum) / simulation.steps / simulation.h
    log_slope = compute_slope(np.log(x0_xl_correlation), simulation.dt)
    E_1 = -log_slope[2]
    text.set_text("$(\\Delta E)_1^{0} = {1:.5f}$".format("{\mathrm{simu}}",E_1))
    if plot_type == "average_x":
        line.set_data(t_values, simulation.x_sum/simulation.steps)
    elif plot_type == "x0_xl_correlation":
        line.set_data(t_values, x0_xl_correlation)
    elif plot_type == "log_slope":
        line.set_data(t_values, log_slope)
    print(simulation.steps)
    return line, text

presets = {
"parabolic": {
"V": dV_parabolic,
"x0": 4.,
"dtau": 0.3,
"Nt": 100,
"dt": 0.1
},
"cosh": {
"V": dV_cosh,
"x0": 4.,
"dtau": 0.1,
"Nt": 100,
"dt": 1.
},
"double_well": {
"V": dV_double_well,
"x0": 4.,
"dtau": 0.1,
"Nt": 100,
"dt": 1.
},
"quartic": {
"V": dV_quartic,
"x0": 4.,
"dtau": 0.1,
"Nt": 100,
"dt": 1.
}
}

preset = presets[potential]
simulation = Simulation(preset["V"], preset["x0"], preset["Nt"], preset["dt"], preset["dtau"], h=1e-6, m=1.)

plot_presets = {
"average_x": {
"ylabel": "$\\langle x(t) \\rangle$",
"ylim": {
"parabolic": (-1.5, 1.5),
"cosh": (-1, 1),
"double_well": (-simulation.x0 * 1.2, simulation.x0 * 1.2),
"quartic": (-2.,2.)
}
},
"x0_xl_correlation": {
"ylabel": "$\\langle x(0) x(t) \\rangle$",
"ylim": {
"parabolic": (0, 0.01),
"cosh": (0, 0.01),
"double_well": (0, 0.01),
"quartic": (0, 0.01),
}
},
"log_slope": {
"ylabel": "$\\partial_t \\> \\log \\> \\langle x(0) x(t) \\rangle$",
"ylim": {
"parabolic": (-2., 2.),
"cosh": (-0.5, 0.5),
"double_well": (-2., 2.),
"quartic": (-2., 2.)
}
}
}

t_values = np.linspace(0, (simulation.Nt - 1) * simulation.dt, simulation.Nt)
plot_xmax = simulation.Nt * simulation.dt
plot_ymin, plot_ymax = plot_presets[plot_type]["ylim"][potential]
plot_height = plot_ymax - plot_ymin
fig = plt.figure()
plt.axes(xlim=(0, plot_xmax), ylim=(plot_ymin, plot_ymax))
#these are the globals used for animation:
line = plt.plot([], [])[0]
text = plt.text(0.2*plot_xmax, plot_ymin + 0.9*plot_height, \
                "$(\\Delta E)_1^{\mathrm{simu}}$")
if potential == "cosh":
    plt.text(0.2*plot_xmax, plot_ymin + 0.8*plot_height, "$(\\Delta E)_1^{\mathrm{theo}}=0.13151$", fontsize=20)
if potential == "quartic":
    plt.text(0.2*plot_xmax, plot_ymin + 0.8*plot_height, "$(\\Delta E)_1^{\mathrm{theo}}=1.72362$", fontsize=20)
plt.xlabel('$t$', fontsize=16)
plt.ylabel(plot_presets[plot_type]["ylabel"], fontsize=16)

# we need to keep a reference to this so it doesn't get garbage collected
animationObject = anim.FuncAnimation(fig, animate, init_func=init, blit=True)
plt.show()

#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
