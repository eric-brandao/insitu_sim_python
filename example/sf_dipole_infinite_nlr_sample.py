# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 2025

@author: Eric Brandão

Simulate the sound field caused by a dipole above an infinite non-locally reacting sample
"""

#%% Importing modules
from controlsair import AlgControls, AirProperties
from sources import Source
from receivers import Receiver
from material import PorousAbsorber
from field_inf_nlr import NLRInfSphDipole
import matplotlib.pyplot as plt
import numpy as np

#%% Define air properties and algorithm controls
air = AirProperties(c0 = 343, rho0 = 1.21)
print(r"Sound speed: {} [m/s]; Air density: {} [kg/m^3]".format(air.c0, air.rho0))
controls = AlgControls(c0 = air.c0, freq_vec = [1500])
print(r"Frequency vector: {} [Hz]".format(controls.freq))
print(r"Wave-number mag. vector: {} [rad/m]".format(controls.k0))

#%% Define your reference material - Miki model and layer over rigid backing
material = PorousAbsorber(air = air, controls = controls)
material.delany_bazley(resistivity = 100000) # this will give you the Characteristic Impedance and complex wave-num
_ = material.layer_over_rigid(thickness = 0.5, theta = 0); # this will give you the surface impedance and absorption (plane-wave inc)
#material.plot_absorption()

#%% Define source
source = Source(coord = [0, 0, 0.1])
print("source coordinates {} [m]".format(source.coord))

#%% Define receivers
receivers = Receiver()
receivers.line_array(line_len = 0.6, step = 0.01, axis = 'x', start_at = 0, zr = 0.01)
print("Receivers coordinates {} [m]".format(receivers.coord))

#%% Define the sound field and run calculations
field = NLRInfSphDipole(air = air, controls = controls, material = material, 
                  sources = source, receivers = receivers)
#field.plot_scene()

#%% Compute sound field
field.p_nlr_semiinf(upper_int_limit = 10)
#field.p_nlr_layer(upper_int_limit = 10)
#%%
plt.figure(figsize = (6,4))
pres = np.real(field.pres_s[0][:,0])/np.amax(np.abs(np.real(field.pres_s[0][:,0])))
plt.plot(receivers.coord[:,0], pres)
pres = np.imag(field.pres_s[0][:,0])/np.amax(np.abs(np.imag(field.pres_s[0][:,0])))
plt.plot(receivers.coord[:,0], pres, '--')
plt.xlabel("x [m]")
plt.ylabel(r"$|P(f)|$")
plt.ylim((-1.2,1.2))
# plt.xticks(ticks = [125, 250, 500, 1000, 2000, 4000], labels = ['125', '250', '500', '1000', '2000', '4000'])
plt.grid(linestyle = '--')


# ax.plot(100*r_vec, np.real(pres[:len(r_vec)])/np.amax(np.abs(np.real(pres[:len(r_vec)]))), label = "Re")
#     ax.plot(100*r_vec, np.imag(pres[:len(r_vec)])/np.amax(np.abs(np.imag(pres[:len(r_vec)]))), '--', label = "Im")
#     ax.grid()
#     ax.legend()
#     ax.set_xlabel("Radial distance (cm)")
#     ax.set_ylabel("Norm. Amp.")
#     ax.set_title(title + " at {} Hz (z = {} m)".format(freq, z))
#     ax.set_ylim((-1.2,1.2))
#     ax.set_xlim((0, 100*r_vec[-1]));
    

