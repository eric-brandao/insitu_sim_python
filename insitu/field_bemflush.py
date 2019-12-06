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
import time

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
        # Load Gauss points and weights
        with open('/home/eric/dev/insitu/data/' + 'gauss_data' + '.pkl', 'rb') as input:
            Nzeta = pickle.load(input)
            Nweights = pickle.load(input)
        self.Nzeta = Nzeta
        self.Nweights = Nweights

    def generate_mesh(self, Lx = 1.0, Ly = 1.0, Nel_per_wavelenth = 6):
        '''
        This method is used to generate the mesh for simulation
        '''
        # Get the maximum frequency to estimate element size required
        self.Lx = Lx
        self.Ly = Ly
        freq_max = self.controls.freq[-1]
        el_size = self.air.c0 / (Nel_per_wavelenth * freq_max)
        print('The el_size is: {}'.format(el_size))
        # Number of elementes spaning x and y directions
        Nel_x = np.int(np.ceil(Lx / el_size))
        Nel_y = np.int(np.ceil(Ly / el_size))
        # x and y coordinates of element's center
        xjc = np.linspace(-Lx/2 + el_size/2, Lx/2 - el_size/2, Nel_x)
        yjc = np.linspace(-Ly/2 + el_size/2, Ly/2 - el_size/2, Nel_y)
        # A Nel_x * Nel_y by 2 matrix containing the x and y coords of element centers
        self.el_center = np.zeros((Nel_x*Nel_y, 2), dtype = np.float32)
        self.jacobian = np.float32((el_size**2)/4.0)
        print(self.jacobian)
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
        # Allocate memory for the surface pressure data (# each column a frequency, each line an element)
        Nel = len(self.el_center)
        self.p_surface = np.zeros((Nel, len(self.controls.k0)), dtype=np.csingle)
        # Generate the C matrix
        c_mtx = 0.5 * np.identity(len(self.el_center), dtype = np.float32)
        # Calculate the distance from source to each element center
        el_3Dcoord = np.zeros((len(self.el_center), 3), dtype=np.float32)
        el_3Dcoord[:,0:2] = self.el_center
        rsel = np.repeat(np.reshape(self.sources.coord[0,:],(1,3)),len(self.el_center),axis=0)-\
            el_3Dcoord
        r_unpt = np.linalg.norm(rsel, axis = 1)
        # Assemble the r-matrix (this runs once and stays in memory for freq loop)
        print("I am assembling a matrix of gauss distances once...")
        r_mtx = insitu_cpp._bemflush_rmtx(self.el_center,
            self.node_x, self.node_y, self.Nzeta)
        # Set a time count for performance check
        tinit = time.time()
        bar = ChargingBar('Calculating the surface pressure for each frequency step',
            max=len(self.controls.k0), suffix='%(percent)d%%')
        for jf, k0 in enumerate(self.controls.k0):
            fakebeta = np.array(0.02+1j*0.2)
            # Assemble the bem matrix (c++)
            # print("Assembling matrix for freq: {} Hz.".format(self.controls.freq[jf]))
            # Version 1 (distances in loop)
            # gij = insitu_cpp._bemflush_mtx(self.el_center, self.node_x, self.node_y,
            # self.Nzeta, self.Nweights.T, k0, self.beta[jf])

            # Version 2 (distances in memory)
            gij = insitu_cpp._bemflush_mtx2(self.Nweights.T,
                r_mtx, self.jacobian, k0, self.beta[jf])
            # Calculate the unperturbed pressure
            p_unpt = 2.0 * np.exp(-1j * k0 * r_unpt) / r_unpt
            # Solve system of equations
            # print("Solving system of eqs for freq: {} Hz.".format(self.controls.freq[jf]))
            self.p_surface[:, jf] = np.linalg.solve(c_mtx + gij, p_unpt)
            # print('Assembling the matrix for frequency {} Hz'.format(self.controls.freq[jf]))
            bar.next()
        bar.finish()
        tend = time.time()
        print("elapsed time: {}".format(tend-tinit))

    def p_fps(self,):
        '''
        This method calculates the sound pressure spectrum for all sources and receivers
        Inputs:

        Outputs:
            pres_s - this is an array of objects. Inside each object there is a
            (N_rec x N_freq) matrix. Each line of the matrix is a spectrum of a sound
            pressure for a receiver. Each column is a set of sound pressures measured
            by the receivers for a given frequency
        '''
        # Loop the receivers
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1]**2))**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                # print('Calculate p_scat and p_fp for rec: {}'.format(r_coord))
                print('Calculate sound pressure for source {} and receiver {}'.format(js+1, jrec+1))
                bar = ChargingBar('Processing sound pressure at field point', max=len(self.controls.k0), suffix='%(percent)d%%')
                for jf, k0 in enumerate(self.controls.k0):
                    # print('the ps passed is: {}'.format(self.p_surface[:,jf]))
                    fakebeta = np.array(0.02+1j*0.2)
                    # r_coord = np.reshape(np.array([0, 0, 0.01], dtype = np.float32), (1,3))
                    p_scat = insitu_cpp._bemflush_pscat(r_coord, self.node_x, self.node_y,
                        self.Nzeta, self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
                    # print('p_scat for freq {} Hz is: {}'.format(self.controls.freq[jf], p_scat))
                    pres_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1) +\
                        (np.exp(-1j * k0 * r2) / r2) + p_scat
                    bar.next()
                bar.finish()
                    # print('p_fp for freq {} Hz is: {}'.format(self.controls.freq[jf], pres_rec[jrec, jf]))
            self.pres_s.append(pres_rec)

    def plot_scene(self, vsam_size = 2):
        '''
        a simple plot of the scene using matplotlib - not redered
        '''
        fig = plt.figure()
        fig.canvas.set_window_title("Measurement scene")
        ax = fig.gca(projection='3d')
        # vertexes plot
        for jel in np.arange(len(self.el_center)):
            nodex_el = self.node_x[jel]
            nodey_el = self.node_y[jel]
            nodex = np.reshape(nodex_el.flatten(), (4, 1))
            nodey = np.reshape(nodey_el.flatten(), (4, 1))
            nodez = np.reshape(np.zeros(4), (4, 1))
            vertices = np.concatenate((nodex, nodey, nodez), axis=1)
            verts = [list(zip(vertices[:,0],
                    vertices[:,1], vertices[:,2]))]
            # mesh points
            for v in verts[0]:
                ax.scatter(v[0], v[1], v[2],
                color='black',  marker = "o", s=1)
            # patch plot
            collection = Poly3DCollection(verts,
                linewidths=1, alpha=0.9, edgecolor = 'gray', zorder=1)
            collection.set_facecolor('silver')
            ax.add_collection3d(collection)
        # baffle
        # vertices = np.array([[-vsam_size/2, -vsam_size/2, 0.0],
        #     [vsam_size/2, -vsam_size/2, 0.0],
        #     [vsam_size/2, vsam_size/2, 0.0],
        #     [-vsam_size/2, vsam_size/2, 0.0]])
        # verts = [list(zip(vertices[:,0],
        #         vertices[:,1], vertices[:,2]))]
        # # patch plot
        # collection = Poly3DCollection(verts,
        #     linewidths=2, alpha=0.1, edgecolor = 'black', zorder=2)
        # ax.add_collection3d(collection)
        # collection.set_facecolor('white')
        # plot source
        for s_coord in self.sources.coord:
            ax.scatter(s_coord[0], s_coord[1], s_coord[2],
                color='black',  marker = "*", s=500)
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
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.view_init(elev=30, azim=-50)
        # ax.invert_zaxis()
        plt.show() # show plot

    def plot_pres(self):
        # plt.figure(1)
        figp, axs = plt.subplots(2,1)
        for js, p_s_mtx in enumerate(self.pres_s):
            for jrec, p_spk in enumerate(p_s_mtx):
                leg = 'source ' + str(js+1) + ' receiver ' + str(jrec+1)
                axs[0].semilogx(self.controls.freq, 20 * np.log10(np.abs(p_spk) / 20e-6), label = leg)
                # axs[0].semilogx(self.controls.freq, np.abs(p_spk), label = leg)
        axs[0].grid(linestyle = '--', which='both')
        axs[0].legend(loc = 'best')
        # axs[0].set(xlabel = 'Frequency [Hz]')
        axs[0].set(ylabel = '|p(f)| [dB]')
        for p_s_mtx in self.pres_s:
            for p_ph in p_s_mtx:
                axs[1].semilogx(self.controls.freq, np.angle(p_ph), label=leg)
        axs[1].grid(linestyle = '--', which='both')
        axs[1].set(xlabel = 'Frequency [Hz]')
        axs[1].set(ylabel = 'phase [-]')
        plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
        xticklabels=['50', '100', '500', '1000', '5000', '10000'])
        plt.setp(axs, xlim=(0.8 * self.controls.freq[0], 1.2*self.controls.freq[-1]))
        plt.show()

    def plot_colormap(self, n_pts = 10):
        pass
        # x = np.linspace(-2*self.Lx, 2*self.Lx, n_pts)
        # y = np.zeros(len(x))
        # z = np.linspace(0.01, 1.0, n_pts)
        # xv, zv = np.meshgrid(x, z)
        # r_fpts = np.zeros((xv.shape[0]**2, 3))
        # print(xv.shape)
        # # print(yv.shape)
        # print(zv.shape)
        # for jrec in np.arange(len(r_fpts)):
        #     r_fpts[jrec, :] = np.array([xv[jrec,jrec],
        #         0.0, zv[jrec,jrec]])
        # print(r_fpts)
        


