import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from insitu.controlsair import load_cfg
import scipy as spy
import scipy.io as sio
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
import pickle
import time
from controlsair import AirProperties, AlgControls, cart2sph
from sources import Source
from receivers import Receiver
from material import PorousAbsorber
from controlsair import plot_spk

class BEMmat():
    '''
    A class to parse simulation data from matlab. The simulation is a coupled BEM-BEM that emulates the
    measurement on finite and non-locally reactive samples
    The inputs are the objects: air, controls, material, sources, receivers
    '''
    def __init__(self, path='/home/eric/research/insitu_arrays/bem_bem/free sides - single sw/DATA_60cm_x_60cm_h25mm_fmax_6000Hz/s0/', fname = 'DATA_60cm_x_60cm_1st-rec_array_3drandom_2nd-rec_array_2layer_3rd-rec___s0_h25mm_fmax_6kHz', array_type = '3D random'):
        # Load the .mat file
        mat_contents = sio.loadmat(path+fname, squeeze_me = False)
        # Get the air data
        air = mat_contents['amb'][0,0]
        self.air = AirProperties(c0 = air['co'].item(), rho0 = air['rho'].item(),
            temperature=air['Tem'].item(), humid=air['HR'].item(), p_atm=air['Po'].item())
        # # Get the controls data
        freq_vec = mat_contents['datafreq_ar'][0,0]
        self.controls = AlgControls(c0 = self.air.c0, freq_vec=freq_vec['frequencia'][0,:])
        # Get the source data
        source_struct = mat_contents['fonte'][0,0]
        source = source_struct['coord'][0,0]
        self.sources = Source()
        self.sources.coord = np.array([source['xs'].item(), source['ys'].item(), source['zs'].item()])
        self.sources.coord = np.reshape(self.sources.coord, (1,3))
        r, theta, phi = cart2sph(self.sources.coord[0,0], self.sources.coord[0,1], self.sources.coord[0,2])
        # Get the material data
        material = mat_contents['material'][0,0]
        self.material = PorousAbsorber(self.air, self.controls)
        self.material.miki(resistivity=material['resist'].item())
        # self.material.resistivity = material['resist'].item()
        # self.material.Zp = material['imp_caract'][0,:]
        # self.material.kp = material['num_onda'][0,:]
        self.material.layer_over_rigid(thickness=material['espessura'].item(),
            theta = np.pi/2-theta)
        # test with my material
        # mymat = PorousAbsorber(self.air, self.controls)
        # mymat.miki(resistivity=material['resist'].item())
        # mymat.layer_over_rigid(thickness=material['espessura'].item(),
        #     theta = np.pi/2-theta)
        # fig = plt.figure()
        # fig.canvas.set_window_title("alpha test")
        # plt.semilogx(self.controls.freq, self.material.alpha, '--k', linewidth = 3)
        # plt.semilogx(self.controls.freq, mymat.alpha, 'b')
        # plt.grid(linestyle = '--', which='both')
        # plt.show()
        self.material.model = "Paulo's Miki"
        # self.material.plot_absorption()
        # Get receiver and sound pressure data - they depend on the type of array
        receivers_struct = mat_contents['recept'][0,0]
        receivers = receivers_struct['coord'][0,0]
        pres = mat_contents['pressoes_eric'][0,0]
        # pres = pres_struct['p_total_rec'][0,,]
        self.receivers = Receiver()
        if array_type == '3D random':
            self.receivers.coord = (np.array([receivers['xr'][0:290,0],
                receivers['yr'][0:290,0], receivers['zr'][0:290,0]])).T
            self.pres_s = [np.array(pres['p_total_rec'][0:290,:])]
        elif array_type == '2 layer':
            self.receivers.coord = (np.array([receivers['xr'][290:418,0],
                receivers['yr'][290:418,0], receivers['zr'][290:418,0]])).T
            self.pres_s = [np.array(pres['p_total_rec'][290:418,:])]
        else:
            self.receivers.coord = (np.array([receivers['xr'][418:,0],
                receivers['yr'][418:,0], receivers['zr'][418:,0]])).T
            self.pres_s = [np.array(pres['p_total_rec'][418:,:])]
        # Get sample size
        size = mat_contents['dim_max_painel'][0,0]
        self.Lx = size['x'].item()
        self.Ly = size['y'].item()
        # Get the pressure data

        # self.controls = controls
        # self.material = material
        # self.sources = sources
        # self.receivers = receivers
        # try:
        #     self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance
        # except:
        #     self.beta = []
        # self.pres_s = []

    def plot_pres(self):
        '''
        Method to plot the spectrum of the sound pressure
        '''
        plot_spk(self.controls.freq, self.pres_s, ref = 20e-6)

    def plot_scene(self, baffle_size = 1.2):
        '''
        a simple plot of the scene using matplotlib - not redered
        '''
        fig = plt.figure()
        fig.canvas.set_window_title("Measurement scene")
        ax = fig.gca(projection='3d')
        # plot side 1
        vertices = np.array([[-self.Lx/2, -self.Ly/2, 0],
            [-self.Lx/2, -self.Ly/2, self.material.thickness],
            [-self.Lx/2, self.Ly/2, self.material.thickness],
            [-self.Lx/2, self.Ly/2, 0]])
        verts = [list(zip(vertices[:,0],
            vertices[:,1], vertices[:,2]))]
        collection = Poly3DCollection(verts,
            linewidths=1, alpha=1, edgecolor = 'gold')
        collection.set_facecolor('gold')
        ax.add_collection3d(collection)
        # plot side 2
        vertices = np.array([[self.Lx/2, -self.Ly/2, 0],
            [self.Lx/2, -self.Ly/2, self.material.thickness],
            [self.Lx/2, self.Ly/2, self.material.thickness],
            [self.Lx/2, self.Ly/2, 0]])
        verts = [list(zip(vertices[:,0],
            vertices[:,1], vertices[:,2]))]
        collection = Poly3DCollection(verts,
            linewidths=1, alpha=1, edgecolor = 'gold')
        collection.set_facecolor('gold')
        ax.add_collection3d(collection)
        # plot side 3
        vertices = np.array([[-self.Lx/2, -self.Ly/2, 0],
            [self.Lx/2, -self.Ly/2, 0],
            [self.Lx/2, -self.Ly/2, self.material.thickness],
            [-self.Lx/2, -self.Ly/2, self.material.thickness]])
        verts = [list(zip(vertices[:,0],
            vertices[:,1], vertices[:,2]))]
        collection = Poly3DCollection(verts,
            linewidths=1, alpha=1, edgecolor = 'gold')
        collection.set_facecolor('gold')
        ax.add_collection3d(collection)
        # plot side 4
        vertices = np.array([[-self.Lx/2, self.Ly/2, 0],
            [self.Lx/2, self.Ly/2, 0],
            [self.Lx/2, self.Ly/2, self.material.thickness],
            [-self.Lx/2, self.Ly/2, self.material.thickness]])
        verts = [list(zip(vertices[:,0],
            vertices[:,1], vertices[:,2]))]
        collection = Poly3DCollection(verts,
            linewidths=1, alpha=1, edgecolor = 'gold')
        collection.set_facecolor('gold')
        ax.add_collection3d(collection)
        # plot top of sample
        vertices = np.array([[-self.Lx/2, -self.Ly/2, self.material.thickness],
            [self.Lx/2, -self.Ly/2, self.material.thickness],
            [self.Lx/2, self.Ly/2, self.material.thickness],
            [-self.Lx/2, self.Ly/2, self.material.thickness]])
        verts = [list(zip(vertices[:,0],
            vertices[:,1], vertices[:,2]))]
        collection = Poly3DCollection(verts,
            linewidths=1, alpha=1, edgecolor = 'gold')
        collection.set_facecolor('gold')
        ax.add_collection3d(collection)
        # plot baffle
        vertices = np.array([[-baffle_size/2, -baffle_size/2, 0.0],
            [baffle_size/2, -baffle_size/2, 0.0],
            [baffle_size/2, baffle_size/2, 0.0],
            [-baffle_size/2, baffle_size/2, 0.0]])
        verts = [list(zip(vertices[:,0],
            vertices[:,1], vertices[:,2]))]
        collection = Poly3DCollection(verts,
            linewidths=1, alpha=0.2, edgecolor = 'black')
        collection.set_facecolor('black')
        ax.add_collection3d(collection)
        # plot source
        for s_coord in [self.sources.coord]:
            ax.scatter(s_coord[0], s_coord[1], s_coord[2],
                color='red',  marker = "o", s=200)
        # plot receiver
        for r_coord in self.receivers.coord:
            ax.scatter(r_coord[0], r_coord[1], r_coord[2],
                color='blue',  marker = "o")
        ax.set_xlabel('X axis')
        # plt.xticks([], [])
        ax.set_ylabel('Y axis')
        # plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='both')
        ax.set_xlim((-baffle_size/2, baffle_size/2))
        ax.set_ylim((-baffle_size/2, baffle_size/2))
        ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.view_init(elev=15, azim=-50)
        # ax.invert_zaxis()
        plt.show() # show plot

    def save(self, filename = 'my_bembem', path = '/home/eric/research/insitu_arrays/results/field/bembem/'):
        '''
        This method is used to save the simulation object
        '''
        filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'my_bembem', path = '/home/eric/research/insitu_arrays/results/field/bembem/'):
        '''
        This method is used to load a simulation object. You build a empty object
        of the class and load a saved one. It will overwrite the empty one.
        '''
        lpath_filename = path + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
