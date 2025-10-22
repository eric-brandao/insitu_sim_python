# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:21:11 2025

@author: Eric Brandão

"""
#%%
import matplotlib.pyplot as plt
import numpy as np
from directDCISM import dDCISM
from controlsair import AlgControls, AirProperties
from material import PorousAbsorber
from sources import Source
from receivers import Receiver
from field_inf_nlr import NLRInfSph  # Field Inf NLR
from lcurve_functions import nmse
#%%
air = AirProperties(c0 = 343.3777064773073, rho0 = 1.2040975259119113)
controls = AlgControls(c0 = air.c0, freq_vec = np.arange(100, 4000, 20))

#%%
source = Source(coord = [0, 0, 0.3])
receivers = Receiver(coord = [0.85, 0.0, 0.04])

#%%
material = PorousAbsorber(air = air, controls = controls)
material.jcal(resistivity = 9207, porosity = 0.99, tortuosity = 1.0, 
              lam = 0.00012643721037645686, lam_l = 0.00013655218720657342)
material.layer_over_rigid(thickness = 0.04, theta = 0.0);
#%%
sf = dDCISM(air = air, controls = controls, source = source, 
            receivers = receivers, material = material,
            T0 = 7.5, dt = 0.1, tol = 1e-6, gamma=1.0)
pres_mtx = sf.predict_p_spk()

#%% Numerical integration
field_nlr = NLRInfSph(air = air, controls = controls, material = material, sources = source, 
                      receivers = receivers, sing_step = 0.01)
upper_lim = 10
field_nlr.p_nlr(upper_int_limit = upper_lim)

#%%


plt.figure(figsize=(6,3))
plt.semilogx(controls.freq, 20*np.log10(np.abs(field_nlr.pres_s[0].flatten())), '--k', label = 'Int')
plt.semilogx(controls.freq, 20*np.log10(np.abs(pres_mtx[0,:])), label = 'MydDCIMS')
plt.grid(linestyle = '--')
plt.title("NMSE = {:.6f}".format(nmse(field_nlr.pres_s[0].flatten(), pres_mtx[0,:])), loc = 'right')
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$|p|$ [dB]")

plt.figure(figsize=(6,3))
plt.semilogx(controls.freq, np.angle(field_nlr.pres_s[0].flatten()), 
             '--k', label = 'Int')
plt.semilogx(controls.freq, np.angle(pres_mtx[0,:]), label = 'MydDCIMS')
plt.grid(linestyle = '--')
plt.title("NMSE = {:.6f}".format(nmse(field_nlr.pres_s[0].flatten(), pres_mtx[0,:])), loc = 'right')
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$|p|$ [dB]")