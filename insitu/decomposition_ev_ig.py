import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
# from matplotlib import cm
# from insitu.controlsair import load_cfg
# import scipy.integrate as integrate
# import scipy as spy
from scipy.interpolate import griddata
from sklearn.linear_model import Ridge
import time
from tqdm import tqdm
#from tqdm.autonotebook import tqdm
import sys
# from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
#from tqdm._tqdm_notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# import cvxpy as cp
# from scipy import linalg # for svd
# from scipy import signal
# from scipy.sparse.linalg import lsqr, lsmr
import lcurve_functions as lc
import pickle
from receivers import Receiver
from material import PorousAbsorber
from controlsair import cart2sph
from rayinidir import RayInitialDirections
#from parray_estimation import octave_freq, octave_avg #, get_hemispheres, get_inc_ref_dirs
from decompositionclass import filter_evan
import utils_insitu


import gmsh
import meshio

import plotly.io as pio

SMALL_SIZE = 11
BIGGER_SIZE = 18
#plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('legend', fontsize=BIGGER_SIZE)
#plt.rc('title', fontsize=BIGGER_SIZE)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('figure', titlesize=BIGGER_SIZE)


class DecompositionEv2(object):
    """ Decomposition of the sound field using propagating and evanescent waves.

    The class has several methods to perform sound field decomposition into a set of
    incident and reflected plane waves. These sets of plane waves are composed of
    propagating and evanescent waves. We create a regular grid on the kx and ky plane,
    wich will contain the evanescent waves. The grid for the propagating plane waves is
    created from the uniform angular subdivision of the surface of a sphere of
    radius k [rad/m]. In the end, we combine the grid of evanescent and propagating
    waves onto two grids - one for the incident and another for the reflected sound field.

    Attributes
    ----------
    p_mtx : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of all the receivers
        Each column is a set of sound pressure at all receivers for a frequency.
    controls : object (AlgControls)
        Controls of the decomposition (frequency spam)
    material : object (PorousAbsorber)
        Contains the material properties (surface impedance). This can be used as reference
        when simulations is what you want to do.
    receivers : object (Receiver)
        The receivers in the field - this contains the information of the coordinates of
        the microphones in your array
    pdir : (n_prop x 3) numpy array
        contains the directions for the reflected propagating waves.
    n_prop : int
        The number of reflected propagating plane waves
    decomp_type : str
        Decomposition description
    f_ref : float
        The amplitude of the reflected waves (or radiating waves)
    f_inc : float
        The amplitude of the incident waves.
    zp : float
        Virtual source plane location (reflected waves).
    zm : float
        Virtual source plane location (incident waves).
    kx : numpy 1darray
        kx wave numbers (for later plots)
    ky : numpy 1darray
        ky wave numbers (for later plots)
    cond_num : (1 x N_freq) numpy 1darray
        condition number of sensing matrix
    pk : list
        List of estimated amplitudes of all plane waves.
        Each element in the list is relative to a frequency of the measurement spectrum.
    fpts : object (Receiver)
        The field points in the field where pressure and velocity are reconstructed
    p_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed sound pressure
        at all the field points
    uz_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed particle vel (z)
        at all the field points
    p_scat : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed reflected sound pressure
        at all the field points
    p_inc : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed incident sound pressure
        at all the field points

    Methods
    ----------
    prop_dir(n_waves = 642, plot = False)
        Create the propagating wave number directions (reflected part)

    pk_tikhonov_ev_ig(f_ref = 1.0, f_inc = 1.0, factor = 2.5, z0 = 1.5, plot_l = False)
        Wave number spectrum estimation using Tikhonov inversion

    reconstruct_pu(receivers)
        Reconstruct the sound pressure and particle velocity at a receiver object

    def reconstruct_pref(receivers, compute_uxy = True)
        Reconstruct the reflected sound pressure at a receiver object

    plot_colormap(self, freq = 1000, total_pres = True)
        Plots a color map of the pressure field.

    plot_pkmap_v2(freq = 1000, db = False, dinrange = 20,
    save = False, name='name', color_code = 'viridis')
        Plot wave number spectrum as a 2D maps (vs. kx and ky)

    plot_pkmap_prop(freq = 1000, db = False, dinrange = 20,
        save = False, name='name', color_code = 'viridis'):
        Plot wave number spectrum  - propagating only (vs. phi and theta)

    save(filename = 'my_bemflush', path = '/home/eric/dev/insitu/data/bem_simulations/')
        To save the simulation object as pickle

    load(filename = 'my_qterm', path = '/home/eric/dev/insitu/data/bem_simulations/')
        Load a simulation object.
    """

    def __init__(self, p_mtx = None, controls = None, material = None, receivers = None,
                 regu_par = 'L-curve'):
        """

        Parameters
        ----------
        p_mtx : (N_rec x N_freq) numpy array
            A matrix containing the complex amplitudes of all the receivers
            Each column is a set of sound pressure at all receivers for a frequency.
        controls : object (AlgControls)
            Controls of the decomposition (frequency spam)
        material : object (PorousAbsorber)
            Contains the material properties (surface impedance).
        receivers : object (Receiver)
            The receivers in the field

        The objects are stored as attributes in the class (easier to retrieve).
        """
        self.controls = controls
        self.material = material
        self.receivers = receivers
        self.pres_s = p_mtx
        # self.flag_oct_interp = False
        if regu_par == 'L-curve' or regu_par == 'l-curve':
            self.regu_par_fun = lc.l_curve
            print("You choose L-curve to find optimal regularization parameter")
        elif regu_par == 'gcv' or regu_par == 'GCV':
            self.regu_par_fun = lc.gcv_lambda
            print("You choose GCV to find optimal regularization parameter")
        else:
            self.regu_par_fun = lc.l_curve
            print("Returning to default L-curve to find optimal regularization parameter")

    def prop_dir(self, n_waves = 642, plot = False):
        """ Create the propagating wave number directions (reflected part)

        The creation of the evanescent waves grid is independent of the propagating waves
        grid. The reflected propagating wave number directions are uniformily distributed
        over the surface of an hemisphere (which will have radius k [rad/m] during the
        decomposition). The directions of propagating waves are calculated with the
        triangulation of an icosahedron used previously (originally implemented in a
        ray tracing algorithm).

        Parameters
        ----------
            n_waves : int
                The number of intended wave-directions to generate (Default is 642).
                Usually the subdivision of the sphere will return an equal or higher
                number of directions. Then, we take the reflected part only (half of it).
            plot : bool
                whether you plot or not the directions in space (bool)
        """
        directions = RayInitialDirections()
        directions, n_sph, elements = directions.isotropic_rays(Nrays = int(n_waves))
        # elements = directions.indices
        id_dir = np.where(directions[:,2]>=0)
        self.id_dir = id_dir
        self.pdir = directions[id_dir[0],:]
        self.pdir_all = directions
        self.n_prop = len(self.pdir[:,0])
        self.conectivities_all = elements
        self.conectivity_correction()
        
        if plot:
            fig = plt.figure()
            ax = plt.axes(projection ="3d")
            ax.scatter(self.pdir[:,0], self.pdir[:,1], self.pdir[:,2])
    
    def conectivity_correction(self,):
        self.sign_vec = np.array([self.pdir_all[self.conectivities_all[:,0], 2],
                             self.pdir_all[self.conectivities_all[:,1], 2],
                             self.pdir_all[self.conectivities_all[:,2], 2]])
        self.sign_z = np.sign(self.sign_vec)
        n_rows = np.sum(self.sign_z.T < 0 , 1)
        self.p_rows = np.where(n_rows == 0)[0]
        self.conectivities = self.conectivities_all[self.p_rows,:]
        self.delta = self.id_dir[0]-np.arange(self.pdir.shape[0])
        
        
        # conectivities2 = np.zeros(self.conectivities.shape, dtype = int)
        for jrow in np.arange(self.conectivities.shape[0]):
            for jcol in np.arange(self.conectivities.shape[1]):
                id_jc = np.where(self.id_dir[0] == self.conectivities[jrow, jcol])[0]
                delta = self.delta[id_jc]
                self.conectivities[jrow, jcol] = self.conectivities[jrow, jcol] - delta
        
        # self.conectivities = conectivities2
        
        # flattened_conectivities = self.conectivities.flatten()
        # sorted_fc = np.sort(flattened_conectivities)
        # fc_idx = np.argsort(flattened_conectivities) #  indices that would sort an array
        
        # sorted_fc_idx = np.argsort(sorted_fc) #  indices that would sort an array
        # count_repetitions = np.bincount(sorted_fc)
        
        # # loop through id_dir
        # delta_array = np.zeros(len(sorted_fc), dtype = int)
        # start = 0
        # for jid, iddir in enumerate(self.id_dir[0]):
        #     n_reps = count_repetitions[iddir]
        #     # delta_array.append(np.repeat(self.delta[jid], n_reps))
        #     stop = start + n_reps - 1
        #     delta_array[start: stop+1] = self.delta[jid]
        #     start = stop + 1
            
            
        # conectivities_corrected_flattend = sorted_fc - delta_array
        # self.conectivities2 = np.reshape(flattened_conectivities[sorted_fc_idx],
        #                                  (self.conectivities.shape[0],3), order='F')
        # # print(flattened_conectivities)
        # print(flattened_conectivities[sorted_fc_idx]-flattened_conectivities)
        # print(sorted_fc)
        # print(sorted_fc_idx)
        # print(np.array(delta_array, dtype=object))
        # print(flattened_conectivities[sorted_fc_idx])
        
        # for jc in np.arange(self.conectivities.shape[1]-1):
        #     for jv in np.arange(self.pdir_all.shape[0]):
        #         id_val = np.where(self.conectivities[:,jc] == jv)[0]
        #         print(id_val)
                # if id_val.size != 0:
                #     self.conectivities[:,id_val] = self.conectivities[:,id_val]-100
            
        # sort a column of conectivities
        # sorted_column_idx = np.argsort(self.conectivities[:,0])
        # ndx = sorted_column_idx[np.searchsorted(self.conectivities[sorted_column_idx,0], 
        #                       self.id_dir[0])]
        # print(self.conectivities[sorted_column_idx,0])
        # print(sorted_column_idx)
        # print(ndx)
    def prop_dir_gmsh(self, n_waves = 642, radius = 50, plot = False):
        """ Create the propagating wave number directions (hemisphere)
        
        The creation of the evanescent waves grid is independent of the propagating waves
        grid. The reflected propagating wave number directions are uniformily distributed
        over the surface of an hemisphere (which will have radius k [rad/m] during the
        decomposition). The directions of propagating waves are computed from gmsh.
        
        Parameters
        ----------
            n_waves : int
                The number of intended wave-directions to generate (Default is 642).
                Usually the subdivision of the sphere will return an equal or higher
                number of directions. Then, we take the reflected part only (half of it).
            plot : bool
                whether you plot or not the directions in space (bool)
        """
        radius = radius
        axis='z'
        #center_support = False
        h = np.sqrt(8 * np.pi * radius ** 2 / (np.sqrt(3) * n_waves)) * 20.2 * np.pi / 50
        coord, elem, all_coords, hemi_index = utils_insitu.gmsh_hemisphere(radius, 
           [h - 1, h], axis)
        axis_dict = {"x": 0, "y": 1, "z": 2}
        
        self.prop_all_coords_gmsh = all_coords
        #self.prop_all_coords_gmsh /= np.linalg.norm(self.prop_all_coords_gmsh, axis = 1)[:,None]
        self.pdir = coord #all_coords #
        #self.pdir /= np.linalg.norm(self.pdir, axis = 1)[:,None]
        self.n_prop = len(self.pdir[:,0])
        #self.prop_coords_gmsh = coord
        
        self.conectivities = elem
        self.receiver_indexes = hemi_index
        #self.len_center_support = len(arc_coords) if center_support else None
        if plot:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(self.pdir[:,0], self.pdir[:,1], self.pdir[:,2])
            #ax.scatter(self.prop_all_coords_gmsh[:,0], self.prop_all_coords_gmsh[:,1], self.prop_all_coords_gmsh[:,2])

    def pk_tikhonov_ev_ig(self, f_ref = 1.0, f_inc = 1.0, factor = 2.5, z0 = 1.5,
        plot_l = False, method = 'Tikhonov', zref = 0.0):
        """ Wave number spectrum estimation using Tikhonov inversion

        Estimate the wave number spectrum using regularized Tikhonov inversion.
        The choice of the regularization parameter is baded on the L-curve criterion.
        This sound field is modelled by a set of propagating and evanescent waves. We
        use a grid for the incident and another for the reflected sound field. This
        method is an adaptation of SONAH, implemented in:
            Hald, J. Basic theory and properties of statistically optimized near-field acoustical
            holography, J Acoust Soc Am. 2009 Apr;125(4):2105-20. doi: 10.1121/1.3079773

        The inversion steps are: (i) - Generate the evanescent wave regular grid;
        (ii) - filter out the propagating waves from this regular grid;
        (iii) - concatenate the filterd evanescent waves grid with the propagating grid
        created earlier, and; (iv) - form the incident and reflected wave number grids;
        (v) - form the sensing matrix; (vi) - compute SVD of the sensing matix;
        (vi) - compute the regularization parameter (L-curve); (vii) - matrix inversion.

        Parameters
        ----------
        f_ref : float
            The amplitude of the reflected waves (or radiating waves). Default value is 1.0
        f_inc : float
            The amplitude of the incident waves . Default value is 1.0.
            If you only have a radiatig structure you can set this to 0.0.
            The algorithm will use the appropiate basis for the decomposition.
        factor : float
            This parameter controls the position of your source-plane. It is combined
            with the microphone spacing on the array. This is an hyper-parameter important
            for regularization of the problem. Typically, the location of the source plane
            will be a multiple of the array spacing - this is what factor represents.
            Default value is 2.5 - found optimal on a large set of simulations.
        z0 : float
            The location of the source plane for the incident sound field.
            It should be a vaule somewhat close to the array. The recomendations is to
            use some sensible value that promotes some simetry relative to the array
            distance from the sample and the array thickness.
        plot_l : bool
            Whether to plot the L-curve or not. Default is false.
        method : string
            Method used on inversion: default is Tikhonov, which uses
            the SVD and is similar to the Matlab implementation. The
            other is Ridge, which uses de sklearn function
        zref : float
            Location of the reference z plane. Default is 0.0
        """
        self.decomp_type = 'Tikhonov (transparent array) w/ evanescent waves - uses irregular grid'
        # Incident and reflected amplitudes
        self.f_ref = f_ref
        self.f_inc = f_inc
        # reflected and incident virtual source plane distances
        self.zref = zref
        self.zp = self.zref - factor * np.amax([self.receivers.ax, self.receivers.ay])
        self.zm = z0 + factor * np.amax([self.receivers.ax, self.receivers.ay])
        # Generate kx and ky for evanescent wave grid
        self.kx = np.linspace(start = -np.pi/self.receivers.ax,
            stop = np.pi/self.receivers.ax, num = 3*int(self.n_prop**0.5))
        self.ky = np.linspace(start = -np.pi/self.receivers.ay,
            stop = np.pi/self.receivers.ay, num = 3*int(self.n_prop**0.5))
        kx_grid, ky_grid = np.meshgrid(self.kx,self.ky)
        kx_e = kx_grid.flatten()
        ky_e = ky_grid.flatten()
        # Initialize variables
        self.cond_num = np.zeros(len(self.controls.k0))
        self.lambd_value_vec = np.zeros(len(self.controls.k0))
        self.pk = []        
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0),
            desc = 'Calculating Tikhonov inversion (with evanescent waves and irregular grid)...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # Filter propagating from evanescent wave grid
            kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
            # compute evanescent kz
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            # Stack evanescent kz with propagating kz for incident and reflected grids
            k_vec_inc = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T,
                np.array([kx_eig, ky_eig, -kz_eig]).T))
            k_vec_ref = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
                np.array([kx_eig, ky_eig, kz_eig]).T))
            # The receivers relative to the virtual source planes 
            recs_inc = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
                self.receivers.coord[:,2]-self.zm]).T
            recs_ref = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
                self.receivers.coord[:,2]-self.zp]).T
            # Forming the sensing matrix
            # fz_ref = np.sqrt(k0/np.abs(k_vec_inc[:,2]))
            psi_inc = np.exp(-1j * recs_inc @ k_vec_inc.T)
            psi_ref = np.exp(-1j * recs_ref @ k_vec_ref.T)
            h_mtx = np.hstack((psi_inc, psi_ref))
            self.cond_num[jf] = np.linalg.cond(h_mtx)
            # Measured data
            pm = self.pres_s[:,jf].astype(complex)
            # Compute SVD
            u, sig, v = lc.csvd(h_mtx)
            # u, sig, v = np.linalg.svd(h_mtx, full_matrices=False)
            # Find the optimal regularization parameter.
            lambd_value = self.regu_par_fun(u, sig, pm, plot_l)
            self.lambd_value_vec[jf] = lambd_value
            # Matrix inversion
            if method == 'Ridge':
                x = lc.ridge_solver(h_mtx,pm,lambd_value)
            elif method == 'Tikhonov':
                x = lc.tikhonov(u,sig,v,pm,lambd_value)
            else:
                x = lc.direct_solver(h_mtx,pm,lambd_value)         
            self.pk.append(x)
            bar.update(1)
        bar.close()

    def filter_wns(self, kc_factor = 1.0, tapper = 2.0, plot_filter = False):
        """ 2D filter in k dommain
        """
        # recover original regular grid
        kx_grid, ky_grid = np.meshgrid(self.kx,self.ky)
        kx_e = kx_grid.flatten()
        ky_e = ky_grid.flatten()
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0), desc = 'Filtering WNS...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # construct the flattened wave numbers
            kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            k_vec = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
                np.array([kx_eig, ky_eig, kz_eig]).T))
            # kr and initialization of the filter
            kx = np.real(k_vec[:,0])
            ky = np.real(k_vec[:,1])
            kr = (kx**2 + ky**2)**0.5
            kfilter = np.zeros(len(kr))
            # Transition band
            idx = np.where(np.logical_and(kr >= -tapper*kc_factor*k0, kr < kc_factor*k0))
            kfilter[idx[0]] = 0.5*(1 - np.cos(2*np.pi*kr[idx[0]]/(tapper*kc_factor*k0)))
            idx = np.where(np.logical_and(kr <= tapper*kc_factor*k0, kr > kc_factor*k0))
            kfilter[idx[0]] = 0.5*(1 - np.cos(2*np.pi*kr[idx[0]]/(tapper*kc_factor*k0)))
            # Pass band
            idx = np.where(np.logical_and(kr >= -kc_factor*k0, kr <= kc_factor*k0))
            kfilter[idx[0]] = 1.0
            # Filter
            pk = np.squeeze(np.asarray(self.pk[jf]))
            pk_i = pk[:int(len(pk)/2)] # incident
            pk_r = pk[int(len(pk)/2):] # reflected
            pk_i = pk_i * kfilter
            pk_r = pk_r * kfilter
            self.pk[jf] = np.reshape(np.concatenate((pk_i, pk_r)), (1,len(pk)))
            # self.pk[jf] = self.pk[jf] * kfilter
            # Plot the filter tapper*k0
            if plot_filter:
                fig = plt.figure(figsize=(7, 5))
                fig.canvas.set_window_title('Filter for freq {:.2f} Hz'.format(self.controls.freq[jf]))
                ax = fig.gca(projection='3d')
                p=ax.scatter(k_vec[:,0], k_vec[:,1], kfilter,
                    c = kfilter, vmin=0, vmax=1)
                fig.colorbar(p)
                # plt.xlim((self.kx[0],self.kx[-1]))
                # plt.ylim((self.ky[0],self.ky[-1]))
                plt.tight_layout()
                plt.show()
            bar.update(1)
        bar.close()



    # def filter_wns(self, tapper = 1.5, plot_filter = False):
    #     """ Construct a circular window to filter the evanescent waves of the WNS
    #     """
    #     import matplotlib.tri as tri
    #     from skimage.filters import window
    #     # recover original regular grid
    #     kx_grid, ky_grid = np.meshgrid(self.kx,self.ky)
    #     kx_e = kx_grid.flatten()
    #     ky_e = ky_grid.flatten()
    #     # Initialize filter
    #     wns_sqfilter = np.zeros((len(self.kx), len(self.ky)))
    #     # Initializa bar
    #     bar = tqdm(total = len(self.controls.k0), desc = 'Filtering WNS...')
    #     # Freq loop
    #     for jf, k0 in enumerate(self.controls.k0):
    #         # get the number of samples in the window
    #         idkx = np.where(np.logical_and(self.kx >= -tapper*k0, self.kx <= tapper*k0))
    #         idky = np.where(np.logical_and(self.ky >= -tapper*k0, self.ky <= tapper*k0))
    #         # Create a base grid for windowing
    #         kxw = np.linspace(-tapper*2*k0, tapper*2*k0, len(idkx[0]))
    #         kyw = np.linspace(-tapper*2*k0, tapper*2*k0, len(idky[0]))
    #         kxwg, kywg = np.meshgrid(kxw, kyw)
    #         # Create a circular tukey window - radialy it goes from 0 to tapper*k0
    #         w = window(('tukey', 0.5*tapper), (len(kxw), len(kyw)))
    #         # Insert the window in the middle of wns_sqfilter (same as zero tappering)
    #         lower = (wns_sqfilter.shape[0]) // 2 - (w.shape[0] // 2)
    #         upper = (wns_sqfilter.shape[0] // 2) + (w.shape[0] // 2)
    #         wns_sqfilter[lower:upper, lower:upper] = w
    #         # substitute propagating part and assemble
    #         idev = np.where(kx_e**2+ky_e**2 > k0**2)
    #         wns_evfilter = wns_sqfilter.flatten()[idev[0]]
    #         wns_pfilter = np.ones(len(self.pdir[:,0]))
    #         wns_filter = np.concatenate((wns_pfilter, wns_evfilter))

    #         # w1 = np.ones(len(self.pdir[:,0]))
    #         # kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
    #         # wns_filter = np.ones(int(self.pk[jf].shape[1]/2))
    #         # idp = np.where(kx_e**2+ky_e**2 > k0**2)
    #         # wns_filter[idp[0]] = wns_sqfilter.flatten()[idp[0]]
    #         print('freq {:.2f}, filter: {}, pk: {}'.format(self.controls.freq[jf],
    #             len(wns_filter), int(self.pk[jf].shape[1]/2)))

    #         # Plot the filter tapper*k0
    #         if plot_filter:
    #             # kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
    #             # kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
    #             # k_vec_ref = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
    #             #     np.array([kx_eig, ky_eig, kz_eig]).T))
    #             # triang = tri.Triangulation(k_vec_ref[:,0], k_vec_ref[:,1])
    #             # fig = plt.figure()
    #             # fig.canvas.set_window_title('Filtered evanescent waves')
    #             # plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
    #             #     k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
    #             # p = plt.tricontourf(wns_filter, levels=50)
    #             kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
    #             kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
    #             k_vec_ref = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
    #             np.array([kx_eig, ky_eig, kz_eig]).T))

    #             fig = plt.figure(figsize=(7, 5))
    #             fig.canvas.set_window_title('Filter for freq {:.2f} Hz'.format(self.controls.freq[jf]))
    #             ax = fig.gca(projection='3d')
    #             # plt.plot(self.controls.k0[jf]*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
    #             #     self.controls.k0[jf]*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
    #             # p = plt.contourf(kx_grid, kx_grid, wns_sqfilter, np.linspace(0,1,101))
    #             # p=ax.scatter(kx_grid.flatten(), ky_grid.flatten(), wns_sqfilter.flatten(),
    #             #     c = wns_sqfilter.flatten(), vmin=0, vmax=1)
    #             p=ax.scatter(k_vec_ref[:,0], k_vec_ref[:,1], wns_filter,
    #                 c = wns_filter, vmin=0, vmax=1)
    #             fig.colorbar(p)
    #             # plt.xlim((self.kx[0],self.kx[-1]))
    #             # plt.ylim((self.ky[0],self.ky[-1]))
    #             plt.tight_layout()
    #             plt.show()
    #         bar.update(1)
    #     bar.close()



    def reconstruct_pu(self, receivers, compute_uxy = True):
        """ Reconstruct the sound pressure and particle velocity at a receiver object

        Reconstruct the pressure and particle velocity at a set of desired field points.
        This can be used on impedance estimation or to plot spatial maps of pressure,
        velocity, intensity.

        The steps are: (i) - Generate the evanescent wave regular grid;
        (ii) - filter out the propagating waves from this regular grid;
        (iii) - concatenate the filterd evanescent waves grid with the propagating grid
        created earlier, and; (iv) - form the incident and reflected wave number grids;
        (v) - form the new sensing matrix; (vi) - compute p and u.

        Parameters
        ----------
        receivers : object (Receiver)
            contains a set of field points at which to reconstruct
        compute_uxy : bool
            Whether to compute x and y components of particle velocity or not (Default is False)
        """
        self.fpts = receivers
        self.p_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        self.uz_recon = np.zeros((self.fpts.coord.shape[0], len(self.controls.k0)), dtype=complex)
        if compute_uxy:
            self.ux_recon = np.zeros((self.fpts.coord.shape[0], len(self.controls.k0)), dtype=complex)
            self.uy_recon = np.zeros((self.fpts.coord.shape[0], len(self.controls.k0)), dtype=complex)
        # Generate kx and ky for evanescent wave grid
        kx_grid, ky_grid = np.meshgrid(self.kx, self.ky)
        kx_e = kx_grid.flatten()
        ky_e = ky_grid.flatten()
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing sound field...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # Filter propagating from evanescent wave grid
            kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
            # compute evanescent kz
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            # Stack evanescent kz with propagating kz for incident and reflected grids
            k_vec_inc = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T,
                np.array([kx_eig, ky_eig, -kz_eig]).T))
            k_vec_ref = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
                np.array([kx_eig, ky_eig, kz_eig]).T))
            # The receivers relative to the virtual source planes 
            recs_inc = np.array([self.fpts.coord[:,0], self.fpts.coord[:,1],
                self.fpts.coord[:,2]-self.zm]).T
            recs_ref = np.array([self.fpts.coord[:,0], self.fpts.coord[:,1],
                self.fpts.coord[:,2]-self.zp]).T
            # Forming the sensing matrix
            psi_inc = np.exp(-1j * recs_inc @ k_vec_inc.T)
            psi_ref = np.exp(-1j * recs_ref @ k_vec_ref.T)
            h_mtx = np.hstack((psi_inc, psi_ref))
            # Compute p and uz
            self.p_recon[:,jf] = np.squeeze(np.asarray(h_mtx @ self.pk[jf].T))
            self.uz_recon[:,jf] = np.squeeze(np.asarray(-((np.divide(np.concatenate((k_vec_inc[:,2], k_vec_ref[:,2])), k0)) *\
                h_mtx) @ self.pk[jf].T))
            if compute_uxy:
                self.ux_recon[:,jf] = np.squeeze(np.asarray(((np.divide(np.concatenate(
                    (k_vec_inc[:,0], k_vec_ref[:,0])), k0)) * h_mtx) @ self.pk[jf].T))
                self.uy_recon[:,jf] = np.squeeze(np.asarray(((np.divide(np.concatenate(
                    (k_vec_inc[:,1], k_vec_ref[:,1])), k0)) * h_mtx) @ self.pk[jf].T))
            bar.update(1)
        bar.close()

    def reconstruct_pref(self, receivers, compute_pinc = False):
        """ Reconstruct the reflected sound pressure at a receiver object

        Reconstruct the the reflected sound pressure at a set of desired field points.
        This can be used on spatial maps of scattered pressure,
        velocity, intensity.

        The steps are: (i) - Generate the evanescent wave regular grid;
        (ii) - filter out the propagating waves from this regular grid;
        (iii) - concatenate the filterd evanescent waves grid with the propagating grid
        created earlier, and; (iv) - form the reflected wave number grids;
        (v) - form the new sensing matrix; (vi) - compute p and u.

        Parameters
        ----------
        receivers : object (Receiver)
            contains a set of field points at which to reconstruct
        compute_pinc : bool
            Whether to compute incident sound pressure or not (Default is False)
        """
        self.fpts = receivers
        self.p_scat = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        if compute_pinc:
            self.p_inc = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        # Generate kx and ky for evanescent wave grid
        kx_grid, ky_grid = np.meshgrid(self.kx, self.ky)
        kx_e = kx_grid.flatten()
        ky_e = ky_grid.flatten()
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing scattered sound field...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # Filter propagating from evanescent wave grid
            kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
            # compute evanescent kz
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            # Stack evanescent kz with propagating kz for incident and reflected grids
            k_vec_ref = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
                np.array([kx_eig, ky_eig, kz_eig]).T))
            # The receivers relative to the virtual source planes 
            recs_ref = np.array([self.fpts.coord[:,0], self.fpts.coord[:,1],
                self.fpts.coord[:,2]-self.zp]).T
            # Forming the sensing matrix
            h_mtx = np.exp(-1j * recs_ref @ k_vec_ref.T)
            # Compute pref
            pk = np.squeeze(np.asarray(self.pk[jf]))
            pk_r = pk[int(len(pk)/2):] # reflected
            self.p_scat[:,jf] = np.squeeze(np.asarray(h_mtx @ pk_r.T))
            if compute_pinc:
                # Stack evanescent kz with propagating kz for incident and reflected grids
                k_vec_inc = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T,
                    np.array([kx_eig, ky_eig, -kz_eig]).T))
                # The receivers relative to the virtual source planes 
                recs_inc = np.array([self.fpts.coord[:,0], self.fpts.coord[:,1],
                    self.fpts.coord[:,2]-self.zm]).T
                # Forming the sensing matrix
                h_mtx_i = np.exp(-1j * recs_inc @ k_vec_inc.T)
                # Compute pref
                pk_i = pk[:int(len(pk)/2)] # incident
                self.p_inc[:,jf] = np.squeeze(np.asarray(h_mtx_i @ pk_i.T))
            bar.update(1)
        bar.close()

    def plot_colormap(self, freq = 1000, total_pres = True, dinrange = 20):
        """Plots a color map of the pressure field.

        Parameters
        ----------
        freq : float
            desired frequency of the color map. If the frequency does not exist
            on the simulation, then it will choose the frequency just before the target.
        total_pres : bool
            Whether to plot the total sound pressure (Default = True) or the reflected only.
            In the later case, we use the reflectd grid only
        dinrange : float
            Dinamic range of the color map

        Returns
        ---------
        plt : Figure object
        """
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        # color parameter
        # color_par = 20*np.log10(np.abs(self.p_recon[:, id_f])/np.amax(np.abs(self.p_recon[:, id_f])))
        if total_pres:
            color_par = 20*np.log10(np.abs(self.p_recon[:, id_f])/np.amax(np.abs(self.p_recon[:, id_f])))
        else:
            color_par = 20*np.log10(np.abs(self.p_scat[:, id_f])/np.amax(np.abs(self.p_scat[:, id_f])))
            # color_par = np.real(self.p_scat[:, id_f])
        # Create triangulazition
        triang = tri.Triangulation(self.fpts.coord[:,0], self.fpts.coord[:,2])
        # Figure
        fig = plt.figure() #figsize=(8, 8)
        # fig = plt.figure()
        fig.canvas.set_window_title('pressure color map')
        plt.title('|P(f)| - reconstructed')
        # p = plt.tricontourf(triang, color_par, cmap = 'seismic')
        p = plt.tricontourf(triang, color_par, np.linspace(-dinrange, 0, int(dinrange)), cmap = 'seismic')

        fig.colorbar(p)
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$z$ [m]')
        return plt

    def plot_intensity(self, freq = 1000):
        """Plots a vector map of the intensity field.

        Parameters
        ----------
        freq : float
            desired frequency of the color map. If the frequency does not exist
            on the simulation, then it will choose the frequency just before the target.

        Returns
        ---------
        plt : Figure object
        """
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        c0 = self.controls.c0
        rho0 = self.material.rho0
        # Intensities
        Ix = 0.5*np.real(self.p_recon[:,id_f] *\
            np.conjugate(self.ux_recon[:,id_f]))
        Iy = 0.5*np.real(self.p_recon[:,id_f] *\
            np.conjugate(self.uy_recon[:,id_f]))
        Iz = 0.5*np.real(self.p_recon[:,id_f] *\
            np.conjugate(self.uz_recon[:,id_f]))
        I = np.sqrt(Ix**2+Iy**2+Iz**2)
        # # Figure
        fig = plt.figure() #figsize=(8, 8)
        fig.canvas.set_window_title('Recon. Intensity distribution map')
        cmap = 'viridis'
        plt.title('Reconstructed |I|')
        # if streamlines:
        #     q = plt.streamplot(self.receivers.coord[:,0], self.receivers.coord[:,2],
        #         Ix/I, Iz/I, color=I, linewidth=2, cmap=cmap)
        #     fig.colorbar(q.lines)
        # else:
        q = plt.quiver(self.fpts.coord[:,0], self.fpts.coord[:,2],
            Ix/I, Iz/I, I, cmap = cmap, width = 0.010)
        #fig.colorbar(q)
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$z$ [m]')
        return plt

    def plot_pkmap_v2(self, freq = 1000, db = False, dinrange = 20,
        save = False, fig_title = '', path = '', fname='', color_code = 'viridis',
        plot_incident = True, dpi = 600, figsize=(8, 8), fileformat='png'):
        """ Plot wave number spectrum as a 2D maps (vs. kx and ky)

        Plot the magnitude of the wave number spectrum (WNS) as two 2D maps of
        evanescent and propagating waves. The map is first interpolated into
        a regular grid. It is a normalized version of the magnitude, either between
        0 and 1 or between -dinrange and 0. The maps are ploted as color as function
        of kx and ky. The radiation circle is also ploted.

        Parameters
        ----------
            freq : float
                Which frequency you want to see. If the calculated spectrum does not contain it
                we plot the closest frequency before the target.
            db : bool
                Whether to plot in linear scale (default) or decibel scale.
            dinrange : float
                You can specify a dinamic range for the decibel scale. It will not affect the
                linear scale.
            save : bool
                Whether to save or not the figure. PDF file with simple standard name
            fig_title : str
                Title of the figure file #FixMe
            path : str
                Path to save the figure file
            fname : str
                File name to save the figure file
            color_code : str
                Can be anything that matplotlib supports. Some recomendations given below:
                'viridis' (default) - Perceptually Uniform Sequential
                'Greys' - White (cold) to black (hot)
                'seismic' - Blue (cold) to red (hot) with a white in the middle
            plot_incident : bool
                Whether to plot incident WNS or not
            dpi : float
                dpi of figure - to save
            figsize : tuple
                size of the figure
        """
        id_f = np.where(self.controls.freq <= freq)
        # id_f = np.where(self.freq_oct <= freq)
        id_f = id_f[0][-1]
        # Quatinties from simulation
        k0 = self.controls.k0[id_f]
        pk = np.squeeze(np.asarray(self.pk[id_f]))
        pk_i = pk[:int(len(pk)/2)] # incident
        pk_r = pk[int(len(pk)/2):] # reflected
        # Original kx and ky
        kx_grid, ky_grid = np.meshgrid(self.kx,self.ky)
        kx_e = kx_grid.flatten()
        ky_e = ky_grid.flatten()
        kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
        kxy = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1]]).T,
            np.array([kx_eig, ky_eig]).T))
        # Interpolate
        pk_i_grid = griddata(kxy, np.abs(pk_i), (kx_grid, ky_grid),
            method='cubic', fill_value=np.finfo(float).eps, rescale=False)
        pk_r_grid = griddata(kxy, np.abs(pk_r), (kx_grid, ky_grid),
            method='cubic', fill_value=np.finfo(float).eps, rescale=False)
        # Calculate colors
        if db:
            np.seterr(divide='ignore')
            color_par_i = 20*np.log10(np.abs(pk_i_grid)/np.amax(np.abs(pk_i_grid)))
            color_par_i[color_par_i<-dinrange] = -dinrange
            color_par_r = 20*np.log10(np.abs(pk_r_grid)/np.amax(np.abs(pk_r_grid)))
            color_par_r[color_par_r<-dinrange] = -dinrange
            color_range = np.arange(-dinrange, 0.1, 0.1)#np.linspace(-dinrange, 0, 10*(dinrange+0.1))
        else:
            color_par_i = np.abs(pk_i_grid)/np.amax(np.abs(pk))
            # color_par_i = np.nan_to_num(color_par_i)
            color_par_r = np.abs(pk_r_grid)/np.amax(np.abs(pk))
            # color_par_r = np.nan_to_num(color_par_r)
            color_range = np.linspace(0, 1, 21)
        # Figure
        if plot_incident:
            fig = plt.figure(figsize=figsize)
            #fig.canvas.set_window_title('Incident WNS - PEIG')
            # Incident
            # plt.subplot(2, 1, 1)
            #plt.title('Incident: ' + fig_title)
            plt.plot(self.controls.k0[id_f]*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
                self.controls.k0[id_f]*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'grey')
            p = plt.contourf(kx_grid, ky_grid, color_par_i,
                color_range, vmin=-dinrange, vmax=0, extend='both', cmap = color_code)
            fig.colorbar(p, shrink=1.0, ticks=np.arange(-dinrange, 3, 3), label = r'$|\bar{P}_i(x, y)|$ dB')
            for c in p.collections:
                c.set_edgecolor("face")
            plt.xlabel(r'$k_x$ [rad/m]')
            plt.ylabel(r'$k_y$ [rad/m]')
            plt.tight_layout()
            if save:
                filename = path + fname + '_' + str(int(freq)) + 'Hz_i'
                plt.savefig(fname = filename, format=fileformat, dpi = dpi)
            # Reflected
            # plt.subplot(2, 1, 2)
            fig = plt.figure(figsize=figsize)
            #fig.canvas.set_window_title('Reflected WNS - PEIG')
            #plt.title('Reflected: ' + fig_title)
            plt.plot(self.controls.k0[id_f]*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
                    self.controls.k0[id_f]*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'grey')
            p = plt.contourf(kx_grid, ky_grid, color_par_r,
                color_range, vmin=-dinrange, vmax=0, extend='both', cmap = color_code)
            fig.colorbar(p, shrink=1.0, ticks=np.arange(-dinrange, 3, 3), label = r'$|\bar{P}_r(x, y)|$ dB')
            for c in p.collections:
                c.set_edgecolor("face")
            plt.xlabel(r'$k_x$ [rad/m]')
            plt.ylabel(r'$k_y$ [rad/m]')
            plt.tight_layout()
            if save:
                filename = path + fname + '_' + str(int(freq)) + 'Hz_r'
                plt.savefig(fname = filename, format=fileformat, dpi = dpi)
        else:
            fig = plt.figure(figsize=figsize)
            #fig.canvas.set_window_title('Reflected WNS - PEIG')
            # Reflected
            #plt.title('Reflected: ' + fig_title)
            plt.plot(self.controls.k0[id_f]*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
                    self.controls.k0[id_f]*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'grey')
            p = plt.contourf(kx_grid, ky_grid, color_par_r,
                color_range, extend='both', cmap = color_code)
            fig.colorbar(p, shrink=1.0, ticks=np.arange(-dinrange, 3, 3), label = r'$|\bar{P}_r(x, y)|$ dB')
            for c in p.collections:
                c.set_edgecolor("face")
            plt.xlabel(r'$k_x$ [rad/m]')
            plt.ylabel(r'$k_y$ [rad/m]')
            plt.tight_layout()
            if save:
                filename = path + fname + '_' + str(int(freq)) + 'Hz_r.pdf'
                plt.savefig(fname = filename, format=fileformat, dpi = dpi, bbox_inches='tight')

    def plot_pkmap_prop(self, freq = 1000, db = False, dinrange = 20,
        save = False, name='name', path = '', fname='', color_code = 'viridis',
        dpi = 600):
        """ Plot wave number spectrum  - propagating only (vs. phi and theta)

        Plot the magnitude of the wave number spectrum as a map of
        propagating waves. The map is first interpolated into
        a regular grid of azimuth (phi) and elevation (theta) angle.
        It is a normalized version of the magnitude, either between
        0 and 1 or between -dinrange and 0. The maps are ploted as color as function
        of phi and theta.

        Parameters
        ----------
            freq : float
                Which frequency you want to see. If the calculated spectrum does not contain it
                we plot the closest frequency before the target.
            db : bool
                Whether to plot in linear scale (default) or decibel scale.
            dinrange : float
                You can specify a dinamic range for the decibel scale. It will not affect the
                linear scale.
            save : bool
                Whether to save or not the figure. PDF file with simple standard name
            name : str
                Name of the figure file #FixMe
            path : str
                Path to save the figure file
            fname : str
                File name to save the figure file
            color_code : str
                Can be anything that matplotlib supports. Some recomendations given below:
                'viridis' (default) - Perceptually Uniform Sequential
                'Greys' - White (cold) to black (hot)
                'seismic' - Blue (cold) to red (hot) with a white in the middle
            dpi : float
                dpi of figure - to save
        """

        id_f = np.where(self.controls.freq <= freq)
        # id_f = np.where(self.freq_oct <= freq)
        id_f = id_f[0][-1]
        # Quatinties from simulation
        k0 = self.controls.k0[id_f]
        pk = np.squeeze(np.asarray(self.pk[id_f]))
        # Propagating incident and reflected
        pk_i = pk[:self.n_prop] # incident
        pk_r = pk[int(len(pk)/2):int(len(pk)/2)+self.n_prop] # reflected
        pk_p = np.hstack((pk_i, pk_r))
        # The directions
        directions = np.vstack((-k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
            -k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T))
        # Transform uninterpolated data to spherical coords
        r, theta, phi = cart2sph(directions[:,0], directions[:,1], directions[:,2])
        # phi = phi[phi > -150]
        # theta = theta[phi > -150]
        # pk_p = pk_p[phi > -150]
        thetaphi = np.transpose(np.array([phi, theta]))
        # Create the new grid to iterpolate
        new_phi = np.linspace(np.deg2rad(-175), np.deg2rad(175), 2*int(np.sqrt(2*self.n_prop))+1)
        new_theta = np.linspace(-np.pi/2, np.pi/2,  int(np.sqrt(2*self.n_prop)))#(0, np.pi, nn)
        grid_phi, grid_theta = np.meshgrid(new_phi, new_theta)
        # Interpolate
        pk_grid = griddata(thetaphi, np.abs(pk_p), (grid_phi, grid_theta),
            method='cubic', fill_value=np.finfo(float).eps, rescale=False)
        # Calculate colors
        if db:
            color_par = 20*np.log10(np.abs(pk_grid)/np.amax(np.abs(pk_grid)))
            color_range = np.linspace(-dinrange, 0, dinrange+1)
        else:
            color_par = np.abs(pk_grid)/np.amax(np.abs(pk_grid))
            color_range = np.linspace(0, 1, 21)
        # Figure
        fig = plt.figure()
        fig.canvas.set_window_title('Interpolated map of |P(k)| for freq {} Hz'.format(self.controls.freq[id_f]))
        plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - PEIG decomp. '+ name)
        p = plt.contourf(np.rad2deg(grid_phi), np.rad2deg(grid_theta)+90, color_par,
            color_range, extend='both', cmap = color_code)
        fig.colorbar(p)
        plt.xlabel(r'$\phi$ (azimuth) [deg]')
        plt.ylabel(r'$\theta$ (elevation) [deg]')
        plt.tight_layout()
        if save:
            filename = path + fname + '_' + str(int(freq)) + 'Hz'
            plt.savefig(fname = filename, format='png', dpi = dpi)

    def plot_directivity(self, freq = 1000, dinrange = 20,
        save = False, fig_title = '', path = '', fname='', color_code = 'viridis',
        true_directivity = True, dpi = 600, figsize=(8, 8), fileformat='png',
        color_method = 'dB', radius_method = 'dB',
        view = 'iso_z', eye = None, renderer = 'notebook',
        remove_axis = False):
        """ Plot directivity as a 3D maps (vs. kx and ky)

        Plot the magnitude of the propagating wave number spectrum (WNS) as 
        3D maps of propagating waves. The map is first interpolated into
        a regular grid. It is a normalized version of the magnitude, either between
        0 and 1 or between -dinrange and 0. The maps are ploted as color as function
        of kx and ky. The radiation circle is also ploted.

        Parameters
        ----------
            freq : float
                Which frequency you want to see. If the calculated spectrum does not contain it
                we plot the closest frequency before the target.
            db : bool
                Whether to plot in linear scale (default) or decibel scale.
            dinrange : float
                You can specify a dinamic range for the decibel scale. It will not affect the
                linear scale.
            save : bool
                Whether to save or not the figure. PDF file with simple standard name
            fig_title : str
                Title of the figure file #FixMe
            path : str
                Path to save the figure file
            fname : str
                File name to save the figure file
            color_code : str
                Can be anything that matplotlib supports. Some recomendations given below:
                'viridis' (default) - Perceptually Uniform Sequential
                'Greys' - White (cold) to black (hot)
                'seismic' - Blue (cold) to red (hot) with a white in the middle
            plot_incident : bool
                Whether to plot incident WNS or not
            dpi : float
                dpi of figure - to save
            figsize : tuple
                size of the figure
        """
        # pio.renderers.default = renderer
        # get freq index
        id_f = utils_insitu.find_freq_index(self.controls.freq, freq)
        
        # wavenumber in air
        k0 = self.controls.k0[id_f]
        # propagating plane wave directions
        directions = k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T
        
        # k-space (total)
        pk = np.squeeze(np.asarray(self.pk[id_f]))
        
        # Directivity k-space (reflected)
        pk_r = pk[int(len(pk)/2):int(len(pk)/2)+self.n_prop] # reflected
        if true_directivity:
            pk_r = (directions[:,2]) * pk_r
        
        #return balloon_data #x, y, z, conectivities, r, plot_pressure
        fig, trace = utils_insitu.plot_3d_polar(self.pdir, self.conectivities,
             pk_r, dinrange = dinrange, 
             color_method = color_method, radius_method = radius_method,
             color_code = color_code, view = view, eye = eye, renderer = renderer,
             remove_axis = remove_axis)
        return fig, trace
# =============================================================================
#         # theta phi representation of original spherical points
#         _, theta, phi = cart2sph(directions[:,0], directions[:,1], directions[:,2])
#         
#         # Calculate colors
#         if db:
#             color_par = 20*np.log10(np.abs(pk_r)/np.amax(np.abs(pk_r)))
#             color_range = np.linspace(-dinrange, 0, dinrange+1)
#         else:
#             color_par = np.abs(pk_r)/np.amax(np.abs(pk_r))
#             color_range = np.linspace(0, 1, 21)
#         
#         utils_insitu.plot3D_directivity_tri(phi, theta, color_par)
# =============================================================================
# =============================================================================
#         # create the new grid to iterpolate
#         grid_phi, grid_theta = utils_insitu.create_angle_grid(self.n_prop, 
#               grid_factor = 2, limit_phi = (-180,180), limit_theta = (0, 90))
#         
#         # interpolate
#         pk_grid = utils_insitu.interpolate2regulargrid(directions, np.abs(pk_r), 
#            grid_phi, grid_theta)
# 
#         # Calculate colors
#         if db:
#             color_par = 20*np.log10(np.abs(pk_grid)/np.amax(np.abs(pk_grid)))
#             color_range = np.linspace(-dinrange, 0, dinrange+1)
#         else:
#             color_par = np.abs(pk_grid)/np.amax(np.abs(pk_grid))
#             color_range = np.linspace(0, 1, 21)
#         
#         utils_insitu.plot3D_directivity(grid_phi, grid_theta, color_par)
# =============================================================================
        
# =============================================================================
#         # Figure
#         fig = plt.figure(figsize = figsize)
#         ax = fig.gca(projection='3d')
#         p = ax.plot_surface(np.rad2deg(grid_phi), np.rad2deg(grid_theta)+90, color_par,
#             color_range, cmap = color_code)
#         fig.colorbar(p)
#         ax.set_xlabel(r'$\phi$ (azimuth) [deg]')
#         ax.set_ylabel(r'$\theta$ (elevation) [deg]')
#         ax.set_zlabel(r'$|P(\theta,\phi)|$ [-]')
#         plt.tight_layout()
#         if save:
#             filename = path + fname + '_' + str(int(freq)) + 'Hz'
#             plt.savefig(fname = filename, format='png', dpi = dpi)
# =============================================================================

        #return fig

    def save(self, filename = 'array_zest', path = '/home/eric/dev/insitu/data/zs_recovery/'):
        """ To save the decomposition object as pickle

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'array_zest', path = '/home/eric/dev/insitu/data/zs_recovery/'):
        """ To load the decomposition object as pickle

        You can instantiate an empty object of the class and load a saved one.
        It will overwrite the empty object.

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        lpath_filename = path + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

class ZsArrayEvIg(DecompositionEv2):
    """ Decomposition and impedance estimation using propagating and evanescent waves.

    The class inherit the attributes and methods of the decomposition class
    DecompositionEv2, which has several methods to perform sound field decomposition
    into a set of incident and reflected plane waves. These sets of plane waves are composed of
    propagating and evanescent waves. We create a regular grid on the kx and ky plane,
    wich will contain the evanescent waves. The grid for the propagating plane waves is
    created from the uniform angular subdivision of the surface of a sphere of
    radius k [rad/m]. In the end, we combine the grid of evanescent and propagating
    waves onto two grids - one for the incident and another for the reflected sound field.

    ZsArrayEvIg then adds methods for the estimation of the surface impedance and absorption
    coefficient. 

    Attributes
    ----------
    p_mtx : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of all the receivers
        Each column is a set of sound pressure at all receivers for a frequency.
    controls : object (AlgControls)
        Controls of the decomposition (frequency spam)
    material : object (PorousAbsorber)
        Contains the material properties (surface impedance). This can be used as reference
        when simulations is what you want to do.
    receivers : object (Receiver)
        The receivers in the field - this contains the information of the coordinates of
        the microphones in your array
    pdir : (n_prop x 3) numpy array
        contains the directions for the reflected propagating waves.
    n_prop : int
        The number of reflected propagating plane waves
    decomp_type : str
        Decomposition description
    f_ref : float
        The amplitude of the reflected waves (or radiating waves)
    f_inc : float
        The amplitude of the incident waves.
    zp : float
        Virtual source plane location (reflected waves).
    zm : float
        Virtual source plane location (incident waves).
    kx : numpy 1darray
        kx wave numbers (for later plots)
    ky : numpy 1darray
        ky wave numbers (for later plots)
    cond_num : (1 x N_freq) numpy 1darray
        condition number of sensing matrix
    pk : list
        List of estimated amplitudes of all plane waves.
        Each element in the list is relative to a frequency of the measurement spectrum.
    fpts : object (Receiver)
        The field points in the field where pressure and velocity are reconstructed
    p_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed sound pressure
        at all the field points
    uz_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed particle vel (z)
        at all the field points
    Zs : (1 x N_freq) numpy 1darray
        Estimated surface impedance (impedance reconstrucion at surface)
    alpha : (1 x N_freq) numpy 1darray
        Estimated absorption coefficient (from impedance reconstrucion at surface)
    self.alpha_pk : (1 x N_freq) numpy 1darray
        Estimated absorption coefficient (from propagating waves on WNS)

    Methods
    ----------
    prop_dir(n_waves = 642, plot = False)
        Create the propagating wave number directions (reflected part)

    pk_tikhonov_ev_ig(f_ref = 1.0, f_inc = 1.0, factor = 2.5, z0 = 1.5, plot_l = False)
        Wave number spectrum estimation using Tikhonov inversion

    reconstruct_pu(receivers)
        Reconstruct the sound pressure and particle velocity at a receiver object

    zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [0], avgZs = True)
        Reconstruct the surface impedance and estimate the absorption

    vp_surf(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, avgvp = True)
        Reconstruct the surface reflection coefficient and estimate the absorption

    plot_colormap(self, freq = 1000, total_pres = True)
        Plots a color map of the pressure field.

    plot_pkmap_v2(freq = 1000, db = False, dinrange = 20,
    save = False, name='name', color_code = 'viridis')
        Plot wave number spectrum as a 2D maps (vs. kx and ky)

    plot_pkmap_prop(freq = 1000, db = False, dinrange = 20,
        save = False, name='name', color_code = 'viridis'):
        Plot wave number spectrum  - propagating only (vs. phi and theta)

    save(filename = 'my_bemflush', path = '/home/eric/dev/insitu/data/bem_simulations/')
        To save the simulation object as pickle

    load(filename = 'my_qterm', path = '/home/eric/dev/insitu/data/bem_simulations/')
        Load a simulation object.
    """

    def __init__(self, p_mtx = None, controls = None, material = None, receivers = None,
                 regu_par = 'L-curve'):
        """

        Parameters
        ----------
        p_mtx : (N_rec x N_freq) numpy array
            A matrix containing the complex amplitudes of all the receivers
            Each column is a set of sound pressure at all receivers for a frequency.
        controls : object (AlgControls)
            Controls of the decomposition (frequency spam)
        material : object (PorousAbsorber)
            Contains the material properties (surface impedance).
        receivers : object (Receiver)
            The receivers in the field

        The objects are stored as attributes in the class (easier to retrieve).
        """
        DecompositionEv2.__init__(self, p_mtx, controls, material, receivers, regu_par)
        super().__init__(p_mtx, controls, material, receivers, regu_par)

    def zs(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [0], avgZs = True):
        """ Reconstruct the surface impedance and estimate the absorption

        Reconstruct pressure and particle velocity at a grid of points
        on ther surface of the absorber (z = 0.0). The absorption coefficient
        is also calculated.

        Parameters
        ----------
        Lx : float
            The length of calculation aperture
        Ly : float
            The width of calculation aperture
        n_x : int
            The number of calculation points in x
        n_y : int
            The number of calculation points in y dir
        theta : list
            Target angles to calculate the absorption from reconstructed impedance
        avgZs : bool
            Whether to average over <Zs> (default - True) or over <p>/<uz> (if False)

        Returns
        -------
        alpha : (N_theta x Nfreq) numpy ndarray
            The absorption coefficients for each target incident angle.
        """
        # Set the grid used to reconstruct the surface impedance
        grid = Receiver()
        grid.planar_array(x_len=Lx, y_len=Ly, zr=0.0, n_x = n_x, n_y = n_x)
        if n_x > 1 or n_y > 1:
            self.grid = grid.coord
        else:
            self.grid = np.array([0,0,0])
        # Reconstruct
        self.reconstruct_pu(receivers=grid)
        Zs_pt = np.divide(self.p_recon, self.uz_recon)
        self.Zs = np.mean(Zs_pt, axis=0)#np.zeros(len(self.controls.k0), dtype=complex)
        self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        for jtheta, dtheta in enumerate(theta):
            self.alpha[jtheta,:] = 1 - (np.abs(np.divide((self.Zs  * np.cos(dtheta) - 1),\
                (self.Zs * np.cos(dtheta) + 1))))**2
        return self.alpha

    def vp_surf(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, avgvp = True):
        """ Reconstruct the surface reflection coefficient and estimate the absorption

        Reconstruct incident and reflected pressure at a grid of points
        on ther surface of the absorber (z = 0.0). The absorption coefficient
        is also calculated.

        Parameters
        ----------
        Lx : float
            The length of calculation aperture
        Ly : float
            The width of calculation aperture
        n_x : int
            The number of calculation points in x
        n_y : int
            The number of calculation points in y dir
        avgvp : bool
            Whether to average over <Zs> (default - True) or over <p>/<uz> (if False)

        Returns
        -------
        alpha_vp : (N_theta x Nfreq) numpy ndarray
            The absorption coefficients for each target incident angle.
        """
        # Set the grid used to reconstruct the surface impedance
        grid = Receiver()
        grid.planar_array(x_len=Lx, y_len=Ly, zr=0.0, n_x = n_x, n_y = n_x)
        if n_x > 1 or n_y > 1:
            self.grid = grid.coord
        else:
            self.grid = np.array([0,0,0])
        # Reconstruct
        self.reconstruct_pref(grid, compute_pinc = True)
        vp_pt = np.divide(self.p_scat, self.p_inc)
        vp_recon = np.mean(np.abs(vp_pt)**2, axis=0)
        # vp_recon = np.linalg.norm(vp_pt, ord=2, axis=0)
        # self.alpha_vp = 1 - (np.abs(vp_recon))**2
        # self.alpha_vp = np.zeros(len(self.controls.k0))
        self.alpha_vp = 1 - vp_recon
        return self.alpha_vp

    def alpha_from_pk(self, ):
        """ Calculate the absorption coefficient from wave-number spectra.

        There is no target angle in this method. Simply, the total reflected energy is
        divided by the total incident energy

        Returns
        -------
        alpha_pk : (1 x N_freq) numpy 1darray
            Estimated absorption coefficient (from propagating waves on WNS)
        """
        # Initialize
        self.alpha_pk = np.zeros(len(self.controls.k0))
        bar = tqdm(total = len(self.controls.k0), desc = 'Calculating absorption from P(k)')
        # loop over frequencies
        for jf, k0 in enumerate(self.controls.k0):
            # Whole P(k) - propagating and evanescet
            pk = np.squeeze(np.asarray(self.pk[jf]))
            # Propagating incident and reflected
            pk_i = pk[:int(len(pk)/2)] # incident
            pk_i2 = np.sum((np.abs(pk_i))**2)
            pk_r = pk[int(len(pk)/2):] # reflected
            pk_r2 = np.sum((np.abs(pk_r))**2)
            self.alpha_pk[jf] = 1 - (pk_r2/pk_i2)
            bar.update(1)
        bar.close()
        return self.alpha_pk