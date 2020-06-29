#%% import general python modules
import numpy as np
import matplotlib.pyplot as plt
# import impedance modules
from controlsair import AlgControls, AirProperties, compare_alpha
from receivers import Receiver
from field_pistonbaffle import PistonOnBaffle
from decompositionclass import Decomposition
#%% Set up the simulation
air = AirProperties(temperature = 20)
controls = AlgControls(c0 = air.c0, freq_vec = [125, 250, 500, 1000, 2000])
array = Receiver()
array.double_planar_array(x_len=0.6, y_len=0.6, n_x = 10, n_y = 10, dz = 0.029, zr = 0.015)
# receivers.brick_array(x_len=0.6, y_len=0.8, z_len=0.25, n_x = 8, n_y = 8, n_z=3)
# receivers.random_3d_array(x_len=0.6, y_len=0.8, z_len=0.25, n_total = 290, zr = 0.1)
arc = Receiver()
arc.arc(radii=20, n_recs=36)
#%% Run the field
arc_field = PistonOnBaffle(air=air, controls=controls, receivers=arc)
arc_field.p_rigid_squared(Lx = 0.2, Ly = 0.2)
# arc_field.plot_scene()
#%% Polar plots
thetas = np.linspace(0, np.pi, len(arc.coord))
fig = plt.figure()
fig.canvas.set_window_title("poloar plots")
for jf in np.arange(0, len(arc_field.controls.freq)):
    plt.subplot(2, 3, jf+1, projection='polar')
    plt.plot(thetas, np.abs(arc_field.pres_s[:,jf]))
fig.tight_layout()
plt.show()
#%% Run a field with array 
field = PistonOnBaffle(air=air, controls=controls, receivers=array)
field.p_rigid_squared(Lx = 0.2, Ly = 0.2)
#%% Run a decomposition
pk = Decomposition(p_mtx=field.pres_s, controls=controls, receivers=array)
pk.wavenum_dir(n_waves=2000)
pk.wavenum_direv(n_waves=50)
pk.pk_tikhonov_ev(method = 'cvx')
# pk.pk_tikhonov(method='cvx')
pk.pk_interpolate()
#%% maps
freq=2000
dinrange = 20

pk.plot_pk_map(freq=freq, db=True, dinrange=dinrange)
pk.plot_pk_sphere(freq=freq, db=True, dinrange=dinrange)
pk.plot_pk_evmap(freq=freq, db=True, dinrange=dinrange)
plt.show()