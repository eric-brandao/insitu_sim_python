from .receivers import Receiver
from .controlsair import compare_alpha, sph2cart
import numpy as np
from progress.bar import ChargingBar
from .field_bemflush import BEMFlush

theta = [0] #[0, 15, 30, 45, 60, 75, 85]

#%% Set the receivers
receivers = Receiver()#[0, 0, 0.013]
# receivers.double_planar_array(x_len=.6, y_len=0.8, n_x = 11, n_y = 13, dz = 0.029, zr = 0.013)
# receivers.double_planar_array(x_len=0.6, y_len=0.6, n_x = 11, n_y = 11, dz = 0.029, zr = 0.013)
# receivers.double_planar_array(x_len=.175, y_len=0.175, n_x = 8, n_y = 8, dz = 0.029, zr = 0.013)
# receivers.random_3d_array(x_len = 0.6, y_len = 0.8, z_len = 0.25, n_total= 290, zr = 0.013)
# receivers.planar_xz(x_len = 0.5, n_x = 40, z0 = 0.005, z_len = 0.1, n_z = 20)
#%% Open file with gij
# field = BEMFlush()
path = '/home/eric/research/insitu_arrays/results/field/bemflush/L30x30/25mm/'
# filename = 'bemf_Lx100cm_Ly100cm_r10900_d100mm_60deg'
# field.load(path = path, filename=filename)
for jel, el in enumerate(theta):
    print('processing angle {} deg'.format(int(el)))
    filename = 'bemf_Lx30cm_Ly30cm_r10900_d25mm_' + str(int(el)) + 'deg_octs'
    field = BEMFlush()
    field.load(path = path, filename=filename)
    field.receivers = receivers
    #run and save
    # field.p_fps()
    field.uz_fps(compute_ux = True)
    new_filename = filename + '_vfp'
    # field.save(path = path, filename=new_filename)