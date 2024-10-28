# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:38:08 2024
@author: Eric Brandão

Interact with receiver class
"""
#%%
from receivers import Receiver

#%% Simplest thing - two microphone method
rec = Receiver(coord = [0, 0, 0.01])
rec.double_rec(z_dist = 0.01)
rec.plot_array()

#%% translate array
rec.translate_array(axis = 'z', delta=0.1)
rec.plot_array()

#%% Line array with rounding
rec.line_array(line_len = 1.0, step = 0.1, axis = 'x', start_at = 0, zr = 0.1)
rec.round_array(num_of_dec_cases = 3)
rec.plot_array()
#%% Double line array with rotation
rec.double_line_array(line_len = 1.0, step = 0.1, axis = 'x', start_at = 0, zr = 0.1, dz = 0.02)
rec.rotate_array(axis = 'z', theta_deg = -45)
rec.plot_array()

#%% Planar array
rec.planar_array(x_len = 0.65, n_x = 11, y_len = 0.57, n_y = 10, zr = 0.1)
rec.round_array(num_of_dec_cases = 3)
rec.plot_array()

#%% Planar xz array (useful for validation)
rec.planar_array_xz(x_len = 0.5, n_x = 20, z_len = 0.5, n_z = 15, yr = -0.1, zr = 0.1)
rec.plot_array()
#%% Random planar array
rec.random_planar_array(x_len = 0.8, y_len = 0.6, zr = 0.1, nx = 10, ny = 10,
                     delta_xy = None, seed = 0, plot = False)
rec.round_array(num_of_dec_cases = 3)
rec.plot_array()

#%% Double layer array
rec.double_planar_array(x_len = 0.65, n_x = 11, y_len=0.57,n_y=10, zr=0.015, dz=0.03)
rec.plot_array()
#%% Brick 3D array
rec.brick_array(x_len = 1.0, n_x = 8, y_len = 1.0, n_y = 8, z_len = 0.3, n_z = 6, zr = 0.1)
rec.plot_array()

#%% Random 3D array by num of microphones
rec.random_3d_array(x_len = 0.3, y_len = 0.3, z_len = 0.25, zr = 0.1, n_total = 200, seed = 0)
rec.round_array(num_of_dec_cases = 3)
rec.plot_array()

#%% Random 3D array by num of microphones
rec.random_3d_array2(x_len = 0.3, y_len = 0.3, z_len = 0.25, zr = 0.1,
                      nx = 10, ny = 11, nz = 2, delta_xyz = None, seed = 0, plot = False)
rec.round_array(num_of_dec_cases = 3)
rec.plot_array()

#%% arc
rec.arc(radius = 20.0, n_recs = 36)
rec.plot_array()
#%% Hemisphere
rec.hemispherical_array(radius = 20, n_rec_target = 642)
rec.plot_array(x_lim=[-25, 25], y_lim=[-25, 25], z_lim=[0, 25])