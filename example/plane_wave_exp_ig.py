# -*- coding: utf-8 -*-
"""
Created on Wed Jan 02 2025
@author: Eric Brandão

Interact with DecompositionEv2 class
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from controlsair import AlgControls, AirProperties, sph2cart
#from sources import Source
from receivers import Receiver
from decomposition_ev_ig import DecompositionEv2, ZsArrayEvIg, filter_evan
import utils_insitu as ut_is
#%% Load file and build controls, receivers and data needed for decomposition
data = np.load('D:/Work/dev/insitu_sim_python/example/todecompdata_1sph.npz')

air = AirProperties(c0 = 343.0, rho0 = 1.21)
controls = AlgControls(c0 = air.c0, freq_vec = data['freq']) 
sf_array = Receiver()
sf_array.coord = data['rec_coords']
sf_array.plot_array(x_lim = [-0.7, 0.7],y_lim = [-0.7, 0.7], z_lim = [0, 0.2])
z_top = np.amax(sf_array.coord[:,2]) + np.amin(sf_array.coord[:,2])
# sf_array.ax = 0.02
# sf_array.ay = 0.04

#%% Decomposition class
ded = DecompositionEv2(p_mtx = data['pres'], controls = controls, receivers = sf_array, 
                       delta_x = 0.045, delta_y = 0.045, regu_par = 'gcv')
ded.prop_dir(n_waves = 642, plot = False)
ded.pk_tikhonov_ev_ig(f_ref=1, f_inc=1, factor=1.5, zs_inc = z_top, zs_ref = 0.0, 
                      plot_l=False, method = 'Tikhonov', 
                      num_of_grid_pts = 3*int(ded.prop_waves_dir.n_prop**0.5)) #3*int(ded.prop_waves_dir.n_prop**0.5)

# ded.pk_tikhonov_ev_ig_balarea(f_ref=1, f_inc=1, factor=1.5, zs_inc = z_top, zs_ref = 0.0, 
#                       plot_l=False, method = 'Tikhonov')

# ded.pk_tikhonov_ev_ig_balarea_vs2(f_ref=1, f_inc=1, factor=1.5, zs_inc = z_top, zs_ref = 0.0, 
#                       plot_l=False, method = 'Tikhonov')

#%% Directivity plot
freq2plot = 700
dinrange = 45
# set plotting object
x, y, z = sph2cart(2, np.deg2rad(15), np.deg2rad(-105))
eye = dict(x=x, y=y, z=z)

fig2, trace2 = ded.plot_directivity(freq = freq2plot, color_method = 'dB', radius_method = 'dB', 
                                    dinrange = dinrange, color_code = 'jet', view = 'iso_z', eye=eye,  
                                    renderer = "browser", true_directivity = False)
fig2.show()

#%% Plot of the kxy grid as the optmized version
# ded.compute_radiation_circle_areas() # mean areas inside radiation circle
# print("2* mean areas inside rad circle: {}".format(2*ded.radiation_circle_mean_areas))
# print("2* max areas inside rad circle: {}".format(2*ded.radiation_circle_max_areas))

idk = [0, 1]
klim = 40
total_area_kxy_map = (2*np.pi/ded.delta_x)*(2*np.pi/ded.delta_y)

# Instantiate figure
fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 8), squeeze = False,
                        sharex = True, sharey = True)

jk = 0
for col in range(2):
    # The optmized way    
    k0 = ded.controls.k0[idk[jk]]
    #max_kxy = 3 * k0
    #total_area_kxy_map = max_kxy**2
    # ded.delta_x = 0.04#np.pi/max_kxy
    # ded.delta_y = 0.04#np.pi/max_kxy #0.04#
    # max_area_in_rad_circle_x2 = 2 * ded.radiation_circle_mean_areas[idk]
    mean_area_in_rad_circle = ded.prop_waves_dir.triangle_areas_mean * k0**2
    #num_of_grid_pts = (total_area_kxy_map/mean_area_in_rad_circle)**0.5
    num_of_grid_pts = ((total_area_kxy_map-2*np.pi*k0**2)/mean_area_in_rad_circle)**0.5
    # area_to_cover = total_area_kxy_map-np.pi*ded.prop_waves_dir.triangle_areas_mean * k0**2
    # num_of_grid_pts = (area_to_cover/mean_area_in_rad_circle)**0.5
    #print("Estimated number of grid pts: {}".format(num_of_grid_pts))
    # Regular grid
    kx, ky, kx_grid_f, ky_grid_f = ded.kxy_init_reg_grid(num_of_grid_pts = int(num_of_grid_pts))
    #print("Estimated evan area: {}".format((kx[-1]-kx[-2])**2))
    # Filter
    kx_eig, ky_eig, n_e = ded.filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
    # compute evanescent kz
    kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
    # Stack evanescent kz with propagating kz for incident and reflected grids
    k_vec_inc, k_vec_ref = ded.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
    print(k_vec_inc.shape)
    # plot axis
    axs[0,col].scatter(np.real(k_vec_ref[:,0]), np.real(k_vec_ref[:,1]), s = 0.5)
    axs[0,col].plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
        k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'k')
    axs[1,col].set_xlabel(r'$k_x$ [rad/m]')
    # axs[0,col].set_ylabel(r'$k_y$ [rad/m]');
    axs[0,col].set_xlim((-klim,klim))
    axs[0,col].set_ylim((-klim,klim))
    
    # The old way
    num_of_grid_pts = 3*int(ded.prop_waves_dir.n_prop**0.5)
    kx, ky, kx_grid_f, ky_grid_f = ded.kxy_init_reg_grid(num_of_grid_pts = num_of_grid_pts)
    # Regular grid
    kx, ky, kx_grid_f, ky_grid_f = ded.kxy_init_reg_grid(num_of_grid_pts = int(num_of_grid_pts))
    #print("Estimated evan area: {}".format((kx[-1]-kx[-2])**2))
    # Filter
    kx_eig, ky_eig, n_e = ded.filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
    # compute evanescent kz
    kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
    # Stack evanescent kz with propagating kz for incident and reflected grids
    k_vec_inc, k_vec_ref = ded.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
    # plot axis
    axs[1,col].scatter(np.real(k_vec_ref[:,0]), np.real(k_vec_ref[:,1]), s = 0.5)
    axs[1,col].plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
        k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'k')
    # axs[1,col].set_xlabel(r'$k_x$ [rad/m]')
    axs[col,0].set_ylabel(r'$k_y$ [rad/m]');
    axs[1,col].set_xlim((-klim,klim))
    axs[1,col].set_ylim((-klim,klim))
    jk += 1

plt.tight_layout()

#%% Sound field reconstruction test
plane_xz = Receiver()
plane_xz.planar_array_xz(x_len = 1.2, n_x = 120, z_len = 0.2, n_z = 20, zr = 0, yr = 0.0)

#ded.reconstruct_pu(plane_xz, compute_uxy = True)
ded.reconstruct_p(plane_xz, compute_inc_ref = True)

#%%
ded.reconstruct_particle_velocity(compute_inc_ref = True, full_field = True)
#%%
ded.get_total_intensity()
ded.get_incident_intensity()
ded.get_reflected_intensity()
#%% Plot pressure color map
desired_freq = 1100
id_desfreq = ut_is.find_freq_index(controls.freq, freq_target = desired_freq)

dinrange = 10

fig, ax = plt.subplots(1, 3, figsize = (9,5), sharey = True)
color_par = ut_is.get_color_par_dBnorm(ded.pt_recon[:, id_desfreq], dinrange)
p = ut_is.plot_map_ax(ax[0], plane_xz.coord[:,0], plane_xz.coord[:,2], 
                      color_par, dinrange)

color_par = ut_is.get_color_par_dBnorm(ded.pi_recon[:, id_desfreq], dinrange)
p = ut_is.plot_map_ax(ax[1], plane_xz.coord[:,0], plane_xz.coord[:,2], 
                      color_par, dinrange)

color_par = ut_is.get_color_par_dBnorm(ded.pr_recon[:, id_desfreq], dinrange)
p = ut_is.plot_map_ax(ax[2], plane_xz.coord[:,0], plane_xz.coord[:,2], 
                      color_par, dinrange)

for i in range(3):
    ax[i].set_xlim((-0.6, 0.6))
    ax[i].set_xlabel('x (m)')
    ax[i].set_ylabel('z (m)')
cbar_ax_start = 0.2
cbar_ax = fig.add_axes([0.91, cbar_ax_start, 0.010, 1-2*cbar_ax_start])
fig.colorbar(p, cax=cbar_ax, shrink=1, ticks=np.arange(-dinrange, 5, 5), label = r'$|\tilde{p}_T(x, z)|$ (dB)');
#plt.tight_layout()

#%% Plot intensity field
desired_freq = 1100
id_desfreq = ut_is.find_freq_index(controls.freq, freq_target = desired_freq)
Ix = -ded.Ix[:,id_desfreq]
Iz = -ded.Iz[:,id_desfreq]
It = ded.It[:,id_desfreq]
fig, ax = plt.subplots(3,1, figsize = (9,7), sharex = True)
q = ax[0].quiver(plane_xz.coord[:,0], plane_xz.coord[:,2], 
                 Ix/It, Iz/It, It, cmap = 'bwr')
fig.colorbar(q, ax=ax[0], shrink=1,  label = r'$I_T(x, z)$');


Ix = -ded.Ix_i[:,id_desfreq]
Iz = -ded.Iz_i[:,id_desfreq]
It = ded.It_i[:,id_desfreq]
q = ax[1].quiver(plane_xz.coord[:,0], plane_xz.coord[:,2], 
                 Ix/It, Iz/It, It, cmap = 'bwr')
fig.colorbar(q, ax=ax[1], shrink=1,  label = r'$I_T(x, z)$');

Ix = -ded.Ix_r[:,id_desfreq]
Iz = -ded.Iz_r[:,id_desfreq]
It = ded.It_r[:,id_desfreq]
q = ax[2].quiver(plane_xz.coord[:,0], plane_xz.coord[:,2], 
                 Ix/It, Iz/It, It, cmap = 'bwr')
fig.colorbar(q, ax=ax[2], shrink=1,  label = r'$I_T(x, z)$');

for i in range(3):
    ax[i].set_xlim((-0.6, 0.6))      
#%% Kxy map
dinrange = 18
freqs2plot = [700,  900, 1100, 1300, 1500]
#fig, axs = plt.subplots(2, 5, figsize = (5*3, 3), sharey = True)

# ded.plot_inc_pkmap_severalfreq(freqs2plot = freqs2plot, db = True, dinrange = dinrange,
#     color_code = 'inferno', figsize = (5*3, 3), fine_tune_subplt = [0.08, 0.2, 0.9, 0.95])

# ded.plot_ref_pkmap_severalfreq(freqs2plot = freqs2plot, db = True, dinrange = dinrange,
#     color_code = 'inferno', figsize = (5*3, 3), fine_tune_subplt = [0.08, 0.2, 0.9, 0.95])

fig, axs = ded.plot_inc_ref_pkmap_severalfreq(freqs2plot = freqs2plot, db = True, dinrange = dinrange,
    color_code = 'inferno', figsize = (5*3, 6), fine_tune_subplt = [0.08, 0.1, 0.9, 0.9])

# for jf, f2p in enumerate(freqs2plot):
#     ax, cbar = ded.plot_inc_pkmap(axs[0, jf], freq = f2p, db = True, dinrange = dinrange, 
#                                      color_code='inferno')
#     ax, cbar = ded.plot_ref_pkmap(axs[1, jf], freq = f2p, db = True, dinrange = dinrange, 
#                                      color_code='inferno')



#ded.plot_inc_pkmap(freq = 1100, db = True, dinrange = 24, color_code='inferno', figsize = (5,4))
# ax, cbar_ax = ded.plot_ref_pkmap(freq = 1100, db = True, dinrange = 24, color_code='inferno')
# ded.plot_inc_ref_pkmap(freq = 700, db = True, dinrange = 24, color_code='inferno', 
#                        figsize = (10,4.5), fine_tune_subplt = [0.08, 0.1, 0.9, 0.95])