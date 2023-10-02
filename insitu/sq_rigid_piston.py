#%% import general python modules
import numpy as np
import matplotlib.pyplot as plt
# import impedance modules
from controlsair import AlgControls, AirProperties, compare_alpha
from receivers import Receiver
from field_pistonbaffle import PistonOnBaffle
from decompositionclass import Decomposition
from decomposition_ev import DecompositionEv
from decomposition_ev_ig import DecompositionEv2
from utils_insitu import plot_3d_polar, plot_3d_polar2, pre_balloon
#%% Set up the simulation
L = 1.5
air = AirProperties(temperature = 20)
controls = AlgControls(c0 = air.c0, freq_vec = [1000])
array = Receiver()
#array.double_planar_array(x_len=0.6, y_len=0.6, n_x = 10, n_y = 10, dz = 0.029, zr = 0.015)
array.planar_array(x_len=0.6, y_len=0.6, n_x = 4, n_y = 4, zr = 0.02)
# receivers.brick_array(x_len=0.6, y_len=0.8, z_len=0.25, n_x = 8, n_y = 8, n_z=3)
# receivers.random_3d_array(x_len=0.6, y_len=0.8, z_len=0.25, n_total = 290, zr = 0.1)
arc = Receiver()
arc.arc(radii=20, n_recs=36)

#%%

sph = Receiver()
sph.hemispherical_array(radius = 20, n_rec_target = 162)

fig = plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter(sph.coord[:,0], sph.coord[:,1], sph.coord[:,2])
#%% Run the field
arc_field = PistonOnBaffle(air=air, controls=controls, receivers=arc)
arc_field.p_rigid_squared(Lx = L, Ly = L)
arc_field.plot_scene()
#%% Polar plots
thetas = np.linspace(0, np.pi, len(arc.coord))
fig = plt.figure()
#fig.canvas.set_window_title("poloar plots")
for jf in np.arange(0, len(arc_field.controls.freq)):
    plt.subplot(2, 3, jf+1, projection='polar')
    plt.plot(thetas, np.abs(arc_field.pres_s[:,jf]))
fig.tight_layout()
plt.show()
#%% Run a field with array 
field = PistonOnBaffle(air=air, controls=controls, receivers=array)
field.p_rigid_squared(Lx = L, Ly = L)
#%%
field.plot_scene(vsam_size = 2)
#%% Run a decomposition
pk = Decomposition(p_mtx=field.pres_s, controls=controls, receivers=array)
pk.wavenum_dir(n_waves=2000)
#pk.wavenum_direv(n_waves=50)
#pk.pk_tikhonov_ev(method = 'Tikhonov')
pk.pk_tikhonov(method='Tikhonov')
pk.pk_interpolate()

#%% maps
freq=2000
dinrange = 20

pk.plot_pk_map(freq=freq, db=True, dinrange=dinrange)
pk.plot_pk_sphere(freq=freq, db=True, dinrange=dinrange)
#pk.plot_pk_evmap(freq=freq, db=True, dinrange=dinrange)
plt.show()
#%%
pk = DecompositionEv(p_mtx=field.pres_s, controls=controls, receivers=array)
pk.create_kx_ky(n_kx = 50, n_ky = 50, plot=True, freq = 1000)
#pk.pk_tikhonov_ev(method = 'Tikhonov')
pk.pk_tikhonov_ev(method='Tikhonov', f_ref = 1.0, f_inc = 0, factor = 1.5, z0 = 0)
#pk.pk_interpolate()


#%% maps
freq=1000
dinrange = 20

pk.plot_pkmap(freq = freq, db = True, dinrange = dinrange)

#%%
pk = DecompositionEv2(p_mtx = field.pres_s, controls = controls, 
                     receivers = array, regu_par = 'l-curve')
pk.prop_dir(n_waves = 642, plot = False)
pk.pk_tikhonov_ev_ig(f_ref=1, f_inc=0, factor=1, z0 = 13/1000+29/1000, zref = 0.0, plot_l=False, method = 'Tikhonov')

#%%
pk.plot_pkmap_v2(freq = 1000, db = True, dinrange = 40, color_code='inferno', fileformat = 'pdf', 
    fig_title='', plot_incident = False, save = False, path = '', fname = 'ref_wns', dpi=600, figsize = (7,5))

#%%

# Loading pre-computed
field_sph = PistonOnBaffle()
field_sph.load('D:/Work/UFSM/Disciplinas/Problemas Inversos/6_Plane_Wave_Expansion_Evanescent_Waves/pob_sphericalcase')
#field_sph.plot_scene(vsam_size = 40)

fig, trace = plot_3d_polar2(field_sph.receivers.coord, field_sph.receivers.conectivities, field_sph.pres_s[:,0], dinrange = 50,
                  color_method = 'dB', radius_method = 'dB',
                  color_code = 'jet', view = 'iso_z', eye = None,
                  renderer = 'notebook', remove_axis = False)
fig.show()