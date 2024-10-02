# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:48:40 2024

@author: Eric Brandão

Simulate the sound field above an infinite non-locally reacting sample
"""

#%% Importing modules
from controlsair import AlgControls, AirProperties
from sources import Source
from receivers import Receiver
from material import PorousAbsorber
from field_inf_nlr import NLRInfSph
import matplotlib.pyplot as plt
import numpy as np

#%% Define air properties and algorithm controls
air = AirProperties(c0 = 343, rho0 = 1.21)
print(r"Sound speed: {} [m/s]; Air density: {} [kg/m^3]".format(air.c0, air.rho0))
controls = AlgControls(c0 = air.c0, freq_init = 100, freq_end = 2500, freq_step = 100)
print(r"Frequency vector: {} [Hz]".format(controls.freq))
print(r"Wave-number mag. vector: {} [rad/m]".format(controls.k0))

#%% Define your reference material - Miki model and layer over rigid backing
material = PorousAbsorber(air = air, controls = controls)
material.miki(resistivity = 10000) # this will give you the Characteristic Impedance and complex wave-num
material.layer_over_rigid(thickness = 0.04, theta = 0) # this will give you the surface impedance and absorption (plane-wave inc)
print("surface impedance vector {} [N s/m]".format(material.Zs))
material.plot_absorption()

#%% Define source
source = Source(coord = [0, 0, 1.0])
print("source coordinates {} [m]".format(source.coord))

#%% Define receivers
receivers = Receiver(coord = [0, 0, 0.02])
receivers.double_rec(z_dist = 0.02)

# receivers = Receiver()
# receivers.random_3d_array2(x_len = 0.3, y_len = 0.3, z_len = 0.15, zr = 0.02,
#                       nx = 5, ny = 5, nz = 2, delta_xyz = None, seed = 0)
# receivers.round_array(num_of_dec_cases = 3)
print("Receivers coordinates {} [m]".format(receivers.coord))

#%% Define the sound field and run calculations
field = NLRInfSph(air = air, controls = controls, material = material, 
                  sources = source, receivers = receivers)
field.plot_scene()

#%% Compute sound field
field.p_nlr(upper_int_limit = 10)
#field.uz_nlr(upper_int_limit = 10) # If you want particle velocity uncomment here
#print("Computed sound pressure vector {}".format(field.pres_s[0]))

#%%
plt.figure(figsize = (6,4))
plt.semilogx(field.controls.freq, 20*np.log10(np.abs(field.pres_s[0].T)))
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$|P(f)|$")
plt.xticks(ticks = [125, 250, 500, 1000, 2000, 4000], labels = ['125', '250', '500', '1000', '2000', '4000'])
plt.grid()


