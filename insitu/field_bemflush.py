import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
import pickle

# import impedance-py/C++ module
import insitu_cpp

class BEMFlush(object):
    '''
    A class to calculate the sound pressure and particle velocity
    using the BEM for a flush sample on a baffle (exact for spherical waves on locally reactive and
    finite samples)
    The inputs are the objects: air, controls, material, sources, receivers
    '''
    def __init__(self, air, controls, material, sources, receivers):
        self.air = air
        self.controls = controls
        self.material = material
        self.sources = sources
        self.receivers = receivers
        self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance
        self.pres_s = []
        self.uz_s = []

    def generate_mesh(self, Lx = 1.0, Ly = 1.0, Nel_per_wavelenth = 10):
        '''
        This method is used to generate the mesh for simulation
        '''
        # Get the maximum frequency to estimate element size required
        freq_max = self.controls.freq[-1]
        el_size = 0.25 #self.air.c0 / (Nel_per_wavelenth * lambda_min)
        # Number of elementes spaning x and y directions
        Nel_x = np.int(np.ceil(Lx / el_size))
        Nel_y = np.int(np.ceil(Ly / el_size))
        # x and y coordinates of element's center
        xjc = np.linspace(-Lx/2 + el_size/2, Lx/2 - el_size/2, Nel_x)
        yjc = np.linspace(-Ly/2 + el_size/2, Ly/2 - el_size/2, Nel_y)
        # A Nel_x * Nel_y by 2 matrix containing the x and y coords of element centers
        self.el_center = np.zeros((Nel_x*Nel_y, 2), dtype = np.float32)
        # x and y coordinates of elements edges
        xje = np.linspace(-Lx/2, Lx/2, Nel_x+1)
        yje = np.linspace(-Ly/2, Ly/2, Nel_y+1)
        # A Nel_x * Nel_y by 4 matrix containing the x and y coords of element x and y corners
        self.node_x = np.zeros((Nel_x*Nel_y, 4), dtype = np.float32)
        self.node_y = np.zeros((Nel_x*Nel_y, 4), dtype = np.float32)
        # form a matrix of coordinates of centers and corners
        d = 0
        for m in np.arange(len(yjc)):
            for n in np.arange(len(xjc)):
                self.el_center[d,:]=[xjc[n], yjc[m]] # each line of matrix nodes is a node xj and yj coordinate
                self.node_x[d,:]=[xje[n], xje[n]+el_size, xje[n]+el_size, xje[n]]
                self.node_y[d,:]=[yje[m], yje[m], yje[m]+el_size, yje[m]+el_size]
                d += 1
        # print(xje)
        # print(self.el_center)
        # print(self.node_x)

    def psurf(self,):
        '''
        This method is used to calculate the acoustic field (pressure/velocity)
        as a function of frequency. It will assemble the BEM matrix and solve for
        the surface pressure based on the incident sound pressure. Each column is
        a complex surface pressure for each element in the mesh. Each row represents
        the evolution of frequency for a single element in the mesh.
        Therefore, there are Nel_x*Nel_y vs N_freq element in this matrix.
        '''
        # self.sources.coord[2]**2
        # Calculate the distance from source to each element center
        # el_3Dcoord = np.zeros((len(self.el_center), 3), dtype=np.float32)
        # el_3Dcoord[:,0:2] = self.el_center
        # # print(el_3Dcoord)
        # r_unpt = self.sources.coord[0,:] - self.el_center
        # r_unpt = (self.sources.coord[0,2]**2 + self.el_center[:,0]**2 + self.el_center[:,1]**2)**0.5
        # print(r_unpt)
        # Load Gauss points and weights
        with open('/home/eric/dev/insitu/data/' + 'gauss_data' + '.pkl', 'rb') as input:
            Nzeta = pickle.load(input)
            Nweights = pickle.load(input)
        # bar = ChargingBar('Calculating the surface pressure for each frequency step',
        #     max=len(self.controls.k0), suffix='%(percent)d%%')
        for jf, k0 in enumerate(self.controls.k0):
            fakebeta = np.array(0.02+1j*0.2)
            gij = insitu_cpp._bemflush_mtx(self.el_center, self.node_x, self.node_y,
            Nzeta, Nweights.T, k0, fakebeta)
            # print('Assembling the matrix for frequency {} Hz'.format(self.controls.freq[jf]))
            # bar.next()
        # bar.finish()
        # print(gij)




    # def plot_scene(self, vsam_size = 50):
    #     '''
    #     a simple plot of the scene using matplotlib - not redered
    #     '''
    #     fig = plt.figure()
    #     fig.canvas.set_window_title("Measurement scene")
    #     ax = fig.gca(projection='3d')
    #     # vertexes plot
    #     vertices = np.array([[-vsam_size/2, -vsam_size/2, 0.0],
    #         [vsam_size/2, -vsam_size/2, 0.0],
    #         [vsam_size/2, vsam_size/2, 0.0],
    #         [-vsam_size/2, vsam_size/2, 0.0]])
    #     # ax.scatter(vertices[:,0], vertices[:,1],
    #     #     vertices[:,2], color='blue')
    #     verts = [list(zip(vertices[:,0],
    #             vertices[:,1], vertices[:,2]))]
    #     # patch plot
    #     collection = Poly3DCollection(verts,
    #         linewidths=1, alpha=0.9, edgecolor = 'gray')
    #     collection.set_facecolor('silver')
    #     ax.add_collection3d(collection)
    #     for s_coord in self.sources.coord:
    #         ax.scatter(s_coord[0], s_coord[1], s_coord[2],
    #             color='black',  marker = "*", s=500)
    #     for r_coord in self.receivers.coord:
    #         ax.scatter(r_coord[0], r_coord[1], r_coord[2],
    #             color='blue',  marker = "o")
    #     ax.set_xlabel('X axis')
    #     plt.xticks([], [])
    #     ax.set_ylabel('Y axis')
    #     plt.yticks([], [])
    #     ax.set_zlabel('Z axis')
    #     # ax.grid(linestyle = ' ', which='none')
    #     ax.set_xlim((-vsam_size/2, vsam_size/2))
    #     ax.set_ylim((-vsam_size/2, vsam_size/2))
    #     ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
    #     ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
    #     ax.view_init(elev=5, azim=-55)
    #     # ax.invert_zaxis()
    #     plt.show() # show plot