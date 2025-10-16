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

# SMALL_SIZE = 11
# BIGGER_SIZE = 18
# #plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.family': 'sans-serif'})
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('legend', fontsize=BIGGER_SIZE)
# #plt.rc('title', fontsize=BIGGER_SIZE)
# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('figure', titlesize=BIGGER_SIZE)


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

    def __init__(self, p_mtx = None, controls = None, receivers = None, 
                 delta_x = 0.05, delta_y = 0.05, regu_par = 'L-curve', material = None):
        """
        Parameters
        ----------
        p_mtx : (N_rec x N_freq) numpy array
            A matrix containing the complex amplitudes of all the receivers
            Each column is a set of sound pressure at all receivers for a frequency.
            Each row is the spectrum for one receiver
        controls : object (AlgControls)
            Controls of the decomposition (frequency spam)
        material : object (PorousAbsorber)
            Contains the material properties (surface impedance).
        receivers : object (Receiver)
            The receivers in the field
        delta_x : float
            An estimative of sensor spacing in x direction. 
            This will control the spam of your kx, ky map, 
            which will go from -pi/delta_x to +pi/delta_x
        delta_y : float
            An estimative of sensor spacing in y direction. 
            This will control the spam of your ky, ky map, 
            which will go from -pi/delta_y to +pi/delta_y

        The objects are stored as attributes in the class (easier to retrieve).
        """
        self.controls = controls
        self.material = material
        self.receivers = receivers
        if hasattr(self.receivers, "ax") and hasattr(self.receivers, "ay"):
            self.delta_x = self.receivers.ax
            self.delta_y = self.receivers.ay
        else:
            self.delta_x = delta_x
            self.delta_y = delta_y
        
        self.pres_s = p_mtx
        # Choose regularization function
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
        self.prop_waves_dir = Receiver()
        self.prop_waves_dir.hemispherical_array(radius = 1, n_rec_target = n_waves)
        # self.pdir = self.prop_waves_dir.coord

        
        # rec_obj = Receiver()
        # rec_obj.hemispherical_array(radius = 1, n_rec_target = n_waves)
        # self.pdir = rec_obj.coord
        # self.n_prop = rec_obj.n_prop
        # self.conectivities_all = rec_obj.conectivities_all
        # self.conectivities = rec_obj.conectivities
        
        # elements = directions.indices
        # id_dir = np.where(rec_obj.coord[:,2]>=0)
        # self.id_dir = id_dir
        # self.pdir = rec_obj.coord
        # # self.pdir_all = rec_obj.coord
        # self.n_prop = rec_obj.n_prop
        # self.conectivities_all = rec_obj.conectivities_all
        # self.conectivities = rec_obj.conectivities
        # self.conectivity_correction()
        # self.compute_normals()
        # self.correct_normals()
        # directions = RayInitialDirections()
        # directions, n_sph, elements = directions.isotropic_rays(Nrays = int(n_waves))
        # # elements = directions.indices
        # id_dir = np.where(directions[:,2]>=0)
        # self.id_dir = id_dir
        # self.pdir = directions[id_dir[0],:]
        # self.pdir_all = directions
        # self.n_prop = len(self.pdir[:,0])
        # self.conectivities_all = elements
        # self.conectivity_correction()
        # self.compute_normals()
        # self.correct_normals()
        
        if plot:
            plt.figure()
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
        
    def compute_normals(self,):
        """ Compute normals of triangle
        """
        self.normals = np.zeros((self.conectivities.shape[0],3))
        for jrow in np.arange(self.conectivities.shape[0]):
            pt_1 = self.pdir[self.conectivities[jrow,0]]
            pt_2 = self.pdir[self.conectivities[jrow,1]]
            pt_3 = self.pdir[self.conectivities[jrow,2]]
            u_vec = pt_2 - pt_1
            v_vec = pt_3 - pt_1
            nx = u_vec[1]*v_vec[2] - u_vec[2]*v_vec[1]
            ny = u_vec[2]*v_vec[0] - u_vec[0]*v_vec[2]
            nz = u_vec[0]*v_vec[1] - u_vec[1]*v_vec[0]
            self.normals[jrow, :] = np.array([nx, ny, nz])
            
    def correct_normals(self,):
        """ correct the normals (to point outward)
        """
        for jrow in np.arange(self.normals.shape[0]):
            if self.normals[jrow, 2] < 0:
                self.normals[jrow, 2] = - self.normals[jrow, 2]
                self.conectivities[jrow, :] = np.flip(self.conectivities[jrow, :]) 
        
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

    def pk_tikhonov_ev_ig(self, f_ref = 1.0, f_inc = 1.0, factor = 1.5, 
                          zs_ref = 0.0, zs_inc = 1.5, num_of_grid_pts = None, 
                          plot_l = False, method = 'Tikhonov'):
        """ Wave number spectrum estimation using L2 norm regularization

        Estimate the wave number spectrum using regularized L2 norm inversion.
        The choice of the regularization parameter can be baded on the L-curve or GCV criterion.
        This sound field is modelled by a set of propagating and evanescent waves. We
        use a grid for the incident and another for the reflected sound field. This
        method is an adaptation of SONAH, implemented in:
            Hald, J. Basic theory and properties of statistically optimized near-field acoustical
            holography, J Acoust Soc Am. 2009 Apr;125(4):2105-20. doi: 10.1121/1.3079773

        The inversion steps are: (i) - Generate the evanescent wave regular grid;
        (ii) - filter out the propagating waves from this regular grid;
        (iii) - concatenate the filterd evanescent waves grid with the propagating grid
        created earlier, and; 
        (iv) - form the incident and reflected wave number grids;
        (v) - form the sensing matrix;
        (vi) - compute SVD of the sensing matix;
        (vii) - compute the regularization parameter; 
        (viii) - matrix inversion.

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
            Default value is 1.5 - optimal values are reported as between 1.0 and 1.5.
        zs_ref : float
            The location of the source plane for the reflected sound field.
            It should be a vaule somewhat near the surface of the sample.
        zs_inc : float
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
        
        """
        # Decomposition message (metadata)
        self.decomp_type = 'Tikhonov (transparent array) w/ evanescent waves - uses irregular grid'
        # Incident and reflected amplitudes (metadata)
        self.f_ref = f_ref
        self.f_inc = f_inc
        # reflected and incident virtual source plane distances  (metadata)
        self.zs_ref = zs_ref
        self.zs_inc = zs_inc
        self.zp = self.zs_ref - factor * np.amax([self.delta_x, self.delta_y])
        self.zm = self.zs_inc + factor * np.amax([self.delta_x, self.delta_y])
        # Generate kx and ky for evanescent wave grid
        if num_of_grid_pts is None:
            num_of_grid_pts = 3*int(self.prop_waves_dir.n_prop**0.5)
        self.num_of_grid_pts = np.zeros(len(self.controls.k0), dtype = int) + num_of_grid_pts
        kx, ky, kx_grid_f, ky_grid_f =\
            self.kxy_init_reg_grid(num_of_grid_pts = num_of_grid_pts)
        # self.kx = np.linspace(start = -np.pi/self.delta_x,
        #     stop = np.pi/self.delta_x, num = 3*int(self.n_prop**0.5))
        # self.ky = np.linspace(start = -np.pi/self.delta_y,
        #     stop = np.pi/self.delta_y, num = 3*int(self.n_prop**0.5))
        # kx_grid, ky_grid = np.meshgrid(self.kx, self.ky)
        # kx_e = kx_grid.flatten()
        # ky_e = ky_grid.flatten()
        # Initialize result variables
        self.cond_num = np.zeros(len(self.controls.k0))
        self.lambd_value_vec = np.zeros(len(self.controls.k0))
        self.problem_size = []
        self.pk = []        
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0),
            desc = 'Calculating Tikhonov inversion (with evanescent waves and irregular grid)...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # Filter propagating from evanescent wave grid
            # kx_eig, ky_eig, n_e = filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
            kx_eig, ky_eig, n_e = self.filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
            # compute evanescent kz
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            # Stack evanescent kz with propagating kz for incident and reflected grids
            k_vec_inc, k_vec_ref = self.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
            # k_vec_inc = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T,
            #     np.array([kx_eig, ky_eig, -kz_eig]).T))
            # k_vec_ref = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
            #     np.array([kx_eig, ky_eig, kz_eig]).T))
            
            # Form sensing matrix
            h_mtx, _, _ = self.build_hmtx_p(k_vec_inc, k_vec_ref, self.receivers)
            self.problem_size.append(h_mtx.shape)
            # self.h_mtx_shape = h_mtx.shape
            # print(self.h_mtx_shape)
            # # The receivers relative to the virtual source planes 
            # recs_inc = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
            #     self.receivers.coord[:,2]-self.zm]).T
            # recs_ref = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
            #     self.receivers.coord[:,2]-self.zp]).T
            # # Forming the sensing matrix
            # # fz_ref = np.sqrt(k0/np.abs(k_vec_inc[:,2]))
            # psi_inc = np.exp(-1j * recs_inc @ k_vec_inc.T)
            # psi_ref = np.exp(-1j * recs_ref @ k_vec_ref.T)
            # if f_inc == 0:
            #     h_mtx = psi_ref
            # else:
            #     h_mtx = np.hstack((psi_inc, psi_ref))
            # Condition number
            self.cond_num[jf] = np.linalg.cond(h_mtx)
            # Measured data as cplx
            pm = self.pres_s[:,jf].astype(complex)
            # Compute SVD
            u, sig, v = lc.csvd(h_mtx)
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
        
    def pk_tikhonov_ev_ig_balarea(self, f_ref = 1.0, f_inc = 1.0, factor = 1.5, 
                          zs_ref = 0.0, zs_inc = 1.5, plot_l = False, method = 'Tikhonov'):
        """ Wave number spectrum estimation using L2 norm regularization

        Estimate the wave number spectrum using regularized L2 norm inversion.
        The choice of the regularization parameter can be baded on the L-curve or GCV criterion.
        This sound field is modelled by a set of propagating and evanescent waves. We
        use a grid for the incident and another for the reflected sound field. This
        method is an adaptation of SONAH, implemented in:
            Hald, J. Basic theory and properties of statistically optimized near-field acoustical
            holography, J Acoust Soc Am. 2009 Apr;125(4):2105-20. doi: 10.1121/1.3079773

        The inversion steps are: (i) - Generate the evanescent wave regular grid;
        (ii) - filter out the propagating waves from this regular grid;
        (iii) - concatenate the filterd evanescent waves grid with the propagating grid
        created earlier, and; 
        (iv) - form the incident and reflected wave number grids;
        (v) - form the sensing matrix;
        (vi) - compute SVD of the sensing matix;
        (vii) - compute the regularization parameter; 
        (viii) - matrix inversion.

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
            Default value is 1.5 - optimal values are reported as between 1.0 and 1.5.
        zs_ref : float
            The location of the source plane for the reflected sound field.
            It should be a vaule somewhat near the surface of the sample.
        zs_inc : float
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
        
        """
        # Decomposition message (metadata)
        self.decomp_type = 'Tikhonov (transparent array) w/ evanescent waves - uses irregular grid'
        # Incident and reflected amplitudes (metadata)
        self.f_ref = f_ref
        self.f_inc = f_inc
        # reflected and incident virtual source plane distances  (metadata)
        self.zs_ref = zs_ref
        self.zs_inc = zs_inc
        self.zp = self.zs_ref - factor * np.amax([self.delta_x, self.delta_y])
        self.zm = self.zs_inc + factor * np.amax([self.delta_x, self.delta_y])
        # Pre-compute the areas inside the radiation circle for all frequencies
        self.compute_radiation_circle_areas()
        total_area_kxy_map = (2*np.pi/self.delta_x)*(2*np.pi/self.delta_y)
        # Initialize result variables
        self.cond_num = np.zeros(len(self.controls.k0))
        self.lambd_value_vec = np.zeros(len(self.controls.k0))
        self.num_of_grid_pts = np.zeros(len(self.controls.k0))
        self.problem_size = []
        self.pk = []
        
        
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0),
            desc = 'Calculating Tikhonov inversion (with evanescent waves and irregular grid)...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
        
            # mean_area_in_rad_circle = self.prop_waves_dir.triangle_areas_mean * k0**2
            mean_area_in_rad_circle = self.prop_waves_dir.triangle_areas_mean * k0**2
            # Generate kx and ky for evanescent wave grid
            # self.num_of_grid_pts[jf] = (total_area_kxy_map/mean_area_in_rad_circle)**0.5
            self.num_of_grid_pts[jf] = ((total_area_kxy_map-2*np.pi*k0**2)/mean_area_in_rad_circle)**0.5
            # self.num_of_grid_pts[jf] = (total_area_kxy_map/(2*mean_area_in_rad_circle))**0.5
            kx, ky, kx_grid_f, ky_grid_f =\
                self.kxy_init_reg_grid(num_of_grid_pts = int(self.num_of_grid_pts[jf]))
                
            # Filter propagating from evanescent wave grid
            kx_eig, ky_eig, n_e = self.filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
            # compute evanescent kz
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            # Stack evanescent kz with propagating kz for incident and reflected grids
            k_vec_inc, k_vec_ref = self.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
            
            # Form sensing matrix
            h_mtx, _, _ = self.build_hmtx_p(k_vec_inc, k_vec_ref, self.receivers)
            self.problem_size.append(h_mtx.shape)
            # Condition number
            self.cond_num[jf] = np.linalg.cond(h_mtx)
            # Measured data as cplx
            pm = self.pres_s[:,jf].astype(complex)
            
            # Compute SVD
            u, sig, v = lc.csvd(h_mtx)
            
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
        
    def compute_radiation_circle_areas(self,):
        """ Compute the areas of 2D triangles inside the radiation circle
        """
        self.radiation_circle_mean_areas = np.zeros(len(self.controls.k0))
        self.radiation_circle_max_areas = np.zeros(len(self.controls.k0))
        self.radiation_circle_min_areas = np.zeros(len(self.controls.k0))
        for jf, k0 in enumerate(self.controls.k0):
            rec_obj = Receiver()
            rec_obj.coord = k0 * np.array([self.prop_waves_dir.coord[:,0], 
                                      self.prop_waves_dir.coord[:,1]]).T
            triangle_obj = tri.Triangulation(rec_obj.coord[:,0], rec_obj.coord[:,1])
            rec_obj.connectivities = triangle_obj.triangles
            rec_obj.compute_triangle_areas()
            self.radiation_circle_mean_areas[jf] = rec_obj.triangle_areas_mean
            self.radiation_circle_max_areas[jf] = np.amax(rec_obj.triangle_areas)
            self.radiation_circle_min_areas[jf] = np.amin(rec_obj.triangle_areas)
        
    def kxy_init_reg_grid(self, num_of_grid_pts = 60):
        """ Get the initial regular kx, ky grid
        """
        # Generate kx and ky for evanescent wave grid
        kx = np.linspace(start = -np.pi/self.delta_x,
            stop = np.pi/self.delta_x, num = num_of_grid_pts)
        ky = np.linspace(start = -np.pi/self.delta_y,
            stop = np.pi/self.delta_y, num = num_of_grid_pts)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        kx_grid_f = kx_grid.flatten()
        ky_grid_f = ky_grid.flatten()
        return kx, ky, kx_grid_f, ky_grid_f

    def filter_evan(self, k0, kx_grid_f, ky_grid_f, plot=False):
        """ Filter the propagating waves

        This auxiliary function will exclude all propagating wave numbers from
        the evanescent wave numbers.
        
        Parameters
        ----------
        k0 : float
            your wave-number value in rad/m
        kx_grid_f : numpy 1dArray
            Flattened version of your kx regular meshgrid
        ky_grid_f : numpy 1dArray
            Flattened version of your ky regular meshgrid
        """
        ke_norm = (kx_grid_f**2 + ky_grid_f**2)**0.5
        kx_e_filtered = kx_grid_f[ke_norm > k0]
        ky_e_filtered = ky_grid_f[ke_norm > k0]
        n_evan = len(kx_e_filtered)
        if plot:
            plt.figure()
            plt.plot(kx_e_filtered, ky_e_filtered, 'o')
            plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
                k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
            plt.xlabel('kx')
            plt.ylabel('ky')
            plt.show()
        return kx_e_filtered, ky_e_filtered, n_evan
    
    def form_kxy_ig(self, k0, kx_ev, ky_ev, kz_ev):
        """ Form the kx,ky irregular grid
        
        Concatenates the created propagating k vectors with the filtered evanescent part
        
        Parameters
        ----------
        k0 : float
            your wave-number value in rad/m
        kx_ev : numpy 1dArray
            kx evanescent waves
        ky_ev : numpy 1dArray
            ky evanescent waves
        kz_ev : numpy 1dArray
            kz evanescent waves
        """
        # Stack evanescent kz with propagating kz for incident and reflected grids
        k_prop_inc = k0 * np.array([self.prop_waves_dir.coord[:,0],
                                    self.prop_waves_dir.coord[:,1],
                                    -self.prop_waves_dir.coord[:,2]]).T
        k_evan_inc = np.array([kx_ev, ky_ev, -kz_ev]).T
        k_vec_inc = np.vstack((k_prop_inc, k_evan_inc))
        # Reflected
        k_prop_ref = k0 * np.array([self.prop_waves_dir.coord[:,0], 
                                    self.prop_waves_dir.coord[:,1], 
                                    self.prop_waves_dir.coord[:,2]]).T
        k_evan_ref = np.array([kx_ev, ky_ev, kz_ev]).T
        k_vec_ref = np.vstack((k_prop_ref, k_evan_ref))
        return k_vec_inc, k_vec_ref
    
    def build_hmtx_p(self, k_vec_inc, k_vec_ref, receiver_obj):
        """ build h_mtx for pressure
        
        Parameters
        ----------
        k_vec_inc : numpy ndArray
            Wave number 3D vectors for incident sound field
        k_vec_ref : numpy ndArray
            Wave number 3D vectors for reflected (or radiated) sound field
        """
        # The receivers relative to the virtual source planes
        recs_inc = np.array([receiver_obj.coord[:,0], receiver_obj.coord[:,1],
            receiver_obj.coord[:,2]-self.zm]).T
        recs_ref = np.array([receiver_obj.coord[:,0], receiver_obj.coord[:,1],
            receiver_obj.coord[:,2]-self.zp]).T
        # Forming the sensing matrix
        psi_inc = self.kernel_p(k_vec = k_vec_inc, rec_coords = recs_inc)
        psi_ref = self.kernel_p(k_vec = k_vec_ref, rec_coords = recs_ref)
        # Check if diffraction or radiation problem
        if self.f_inc == 0: # radiation
            h_mtx = psi_ref
        else: # diffraction
            h_mtx = np.hstack((psi_inc, psi_ref))
        return h_mtx, psi_inc, psi_ref
        
    def kernel_p(self, k_vec, rec_coords):
        """ Sound pressure kernel
        """
        return np.exp(-1j * rec_coords @ k_vec.T)
    
    def separare_inc_ref_wns(self, pk):
        """ Separates the incident and reflected wave number spectra
        
        Returns the WNS for incident and reflected (scattered) sound fields. 
        Propagating and evanescent.
        """        
        pk_i = pk[:int(len(pk)/2)] # incident
        pk_r = pk[int(len(pk)/2):] # reflected
        return pk_i, pk_r
    
    def reconstruct_p(self, receivers, compute_inc_ref = False, 
                      compute_u_mode = 'z'):
        """ Reconstruct the sound pressure at a receiver object
        
        Parameters
        ----------
        compute_inc_ref : bool
            Whether to reconstruct the incident and reflected fields separatly
            
        compute_u_mode : str ('z', 'full', 'none')
            Whether to reconstruct the particle velocity. None or False means
            no particle velocity field is reconstructed. True or 'full' means
            ux, uy, and uz are reconstructed. 'z' means only uz is reconstructed
        """
        # initialize
        self.fpts = receivers
        self.initialize_p_recon(num_of_fpts = self.fpts.coord.shape[0], 
                                compute_inc_ref = compute_inc_ref)
        
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing pressure field...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # Generate kx and ky for evanescent wave grid
            kx, ky, kx_grid_f, ky_grid_f =\
                self.kxy_init_reg_grid(num_of_grid_pts = int(self.num_of_grid_pts[jf]))
            # Filter propagating from evanescent wave grid
            kx_eig, ky_eig, n_e = self.filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
            # compute evanescent kz
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            # Stack evanescent kz with propagating kz for incident and reflected grids
            k_vec_inc, k_vec_ref = self.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
            # Form sensing matrix
            h_mtx, psi_inc, psi_ref = self.build_hmtx_p(k_vec_inc, k_vec_ref, self.fpts)
            self.pt_recon[:,jf] = np.squeeze(np.asarray(h_mtx @ self.pk[jf].T))
            
            # Compute incident and reflected parts separetly
            if compute_inc_ref:
                pk_i, pk_r = self.separare_inc_ref_wns(self.pk[jf])
                self.pi_recon[:,jf] = np.squeeze(np.asarray(psi_inc @ pk_i.T))
                self.pr_recon[:,jf] = np.squeeze(np.asarray(psi_ref @ pk_r.T))
                
            bar.update(1)
        bar.close()
    
    def reconstruct_particle_velocity(self, compute_inc_ref = False,
                                      full_field = False):
        """ Reconstruct the sound pressure at a receiver object
        
        Parameters
        ----------
        compute_inc_ref : bool
            Whether to reconstruct the incident and reflected fields separatly
            
        """
        if full_field:
            self.initialize_ux_recon(num_of_fpts = self.fpts.coord.shape[0], 
                           compute_inc_ref = compute_inc_ref)
            self.initialize_uy_recon(num_of_fpts = self.fpts.coord.shape[0], 
                           compute_inc_ref = compute_inc_ref)
            self.initialize_uz_recon(num_of_fpts = self.fpts.coord.shape[0], 
                           compute_inc_ref = compute_inc_ref)
        else:
            self.initialize_uz_recon(num_of_fpts = self.fpts.coord.shape[0], 
                           compute_inc_ref = compute_inc_ref)
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing particle velocity field...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # Generate kx and ky for evanescent wave grid
            kx, ky, kx_grid_f, ky_grid_f =\
                self.kxy_init_reg_grid(num_of_grid_pts = int(self.num_of_grid_pts[jf]))
            # Filter propagating from evanescent wave grid
            kx_eig, ky_eig, n_e = self.filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
            # compute evanescent kz
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            # Stack evanescent kz with propagating kz for incident and reflected grids
            k_vec_inc, k_vec_ref = self.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
            # Form sensing matrix
            h_mtx, psi_inc, psi_ref = self.build_hmtx_p(k_vec_inc, k_vec_ref, self.fpts)
            
            if full_field:
                self.uxt_recon[:,jf] = self.compute_ut(k_vec_inc, k_vec_ref, k0, 
                                                      h_mtx, self.pk[jf], direction = 'x')
                self.uyt_recon[:,jf] = self.compute_ut(k_vec_inc, k_vec_ref, k0, 
                                                      h_mtx, self.pk[jf], direction = 'y')
                self.uzt_recon[:,jf] = self.compute_ut(k_vec_inc, k_vec_ref, k0, 
                                                      h_mtx, self.pk[jf], direction = 'z')
            else:
                self.uzt_recon[:,jf] = self.compute_ut(k_vec_inc, k_vec_ref, k0, 
                                                      h_mtx, self.pk[jf], direction = 'z')   
            # Compute incident and reflected parts separetly
            if compute_inc_ref:
                pk_i, pk_r = self.separare_inc_ref_wns(self.pk[jf])
                if full_field:
                    self.uxi_recon[:,jf] = self.compute_u_inc_or_ref(k_vec_inc, k0, 
                                                          psi_inc, pk_i, direction = 'x')
                    self.uxr_recon[:,jf] = self.compute_u_inc_or_ref(k_vec_ref, k0, 
                                                          psi_ref, pk_r, direction = 'x')
                    self.uyi_recon[:,jf] = self.compute_u_inc_or_ref(k_vec_inc, k0, 
                                                          psi_inc, pk_i, direction = 'y')
                    self.uyr_recon[:,jf] = self.compute_u_inc_or_ref(k_vec_ref, k0, 
                                                          psi_ref, pk_r, direction = 'y')
                    self.uzi_recon[:,jf] = self.compute_u_inc_or_ref(k_vec_inc, k0, 
                                                          psi_inc, pk_i, direction = 'z')
                    self.uzr_recon[:,jf] = self.compute_u_inc_or_ref(k_vec_ref, k0, 
                                                          psi_ref, pk_r, direction = 'z')
                else:
                    self.uzi_recon[:,jf] = self.compute_u_inc_or_ref(k_vec_inc, k0, 
                                                          psi_inc, pk_i, direction = 'z')
                    self.uzr_recon[:,jf] = self.compute_u_inc_or_ref(k_vec_ref, k0, 
                                                          psi_ref, pk_r, direction = 'z')
            bar.update(1)
        bar.close()
    
    def get_total_intensity(self,):
        """ Gets the total sound field intensity at a given direction
        """
        self.Ix = np.zeros(self.pt_recon.shape)
        self.Iy = np.zeros(self.pt_recon.shape)
        self.Iz = np.zeros(self.pt_recon.shape)
        self.It = np.zeros(self.pt_recon.shape)
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            self.Ix[:,jf] = 0.5 * np.real(self.pt_recon[:, jf] * self.uxt_recon[:,jf])
            self.Iy[:,jf] = 0.5 * np.real(self.pt_recon[:, jf] * self.uyt_recon[:,jf])
            self.Iz[:,jf] = 0.5 * np.real(self.pt_recon[:, jf] * self.uzt_recon[:,jf])
            self.It[:,jf] = np.sqrt(self.Ix[:,jf]**2 + self.Iy[:,jf]**2 + self.Iz[:,jf]**2)
    
    def get_incident_intensity(self,):
        """ Gets the total sound field intensity at a given direction
        """
        self.Ix_i = np.zeros(self.pt_recon.shape)
        self.Iy_i = np.zeros(self.pt_recon.shape)
        self.Iz_i = np.zeros(self.pt_recon.shape)
        self.It_i = np.zeros(self.pt_recon.shape)
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            self.Ix_i[:,jf] = 0.5 * np.real(self.pi_recon[:, jf] * self.uxi_recon[:,jf])
            self.Iy_i[:,jf] = 0.5 * np.real(self.pi_recon[:, jf] * self.uyi_recon[:,jf])
            self.Iz_i[:,jf] = 0.5 * np.real(self.pi_recon[:, jf] * self.uzi_recon[:,jf])
            self.It_i[:,jf] = np.sqrt(self.Ix_i[:,jf]**2 + self.Iy_i[:,jf]**2 + self.Iz_i[:,jf]**2)
    
    def get_reflected_intensity(self,):
        """ Gets the total sound field intensity at a given direction
        """
        self.Ix_r = np.zeros(self.pt_recon.shape)
        self.Iy_r = np.zeros(self.pt_recon.shape)
        self.Iz_r = np.zeros(self.pt_recon.shape)
        self.It_r = np.zeros(self.pt_recon.shape)
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            self.Ix_r[:,jf] = 0.5 * np.real(self.pr_recon[:, jf] * self.uxr_recon[:,jf])
            self.Iy_r[:,jf] = 0.5 * np.real(self.pr_recon[:, jf] * self.uyr_recon[:,jf])
            self.Iz_r[:,jf] = 0.5 * np.real(self.pr_recon[:, jf] * self.uzr_recon[:,jf])
            self.It_r[:,jf] = np.sqrt(self.Ix_r[:,jf]**2 + self.Iy_r[:,jf]**2 + self.Iz_r[:,jf]**2)
            
    # def reconstruct_ux(self, compute_inc_ref = False):
    #     """ Reconstruct the sound pressure at a receiver object
        
    #     Parameters
    #     ----------
    #     compute_inc_ref : bool
    #         Whether to reconstruct the incident and reflected fields separatly
            
    #     """
        
    #     self.initialize_ux_recon(num_of_fpts = self.fpts.coord.shape[0], 
    #                        compute_inc_ref = compute_inc_ref)
    #     # Initializa bar
    #     bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing z-velocity field...')
    #     # Freq loop
    #     for jf, k0 in enumerate(self.controls.k0):
    #         # Generate kx and ky for evanescent wave grid
    #         kx, ky, kx_grid_f, ky_grid_f =\
    #             self.kxy_init_reg_grid(num_of_grid_pts = int(self.num_of_grid_pts[jf]))
    #         # Filter propagating from evanescent wave grid
    #         kx_eig, ky_eig, n_e = self.filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
    #         # compute evanescent kz
    #         kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
    #         # Stack evanescent kz with propagating kz for incident and reflected grids
    #         k_vec_inc, k_vec_ref = self.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
            
    #         self.uxt_recon[:,jf] = self.compute_u(k_vec_inc, k_vec_ref, k0, 
    #                                               self.pt_recon[:,jf], direction = 'x')
    #         # Compute incident and reflected parts separetly
    #         if compute_inc_ref:
    #             self.uxi_recon[:,jf] = self.compute_u(k_vec_inc, k_vec_ref, k0, 
    #                                                   self.pi_recon[:,jf], direction = 'x')
    #             self.uxr_recon[:,jf] = self.compute_u(k_vec_inc, k_vec_ref, k0, 
    #                                                   self.pr_recon[:,jf], direction = 'x')
    #         bar.update(1)
    #     bar.close()
    
    # def reconstruct_uy(self, compute_inc_ref = False):
    #     """ Reconstruct the sound pressure at a receiver object
        
    #     Parameters
    #     ----------
    #     compute_inc_ref : bool
    #         Whether to reconstruct the incident and reflected fields separatly
            
    #     """
        
    #     self.initialize_uy_recon(num_of_fpts = self.fpts.coord.shape[0], 
    #                        compute_inc_ref = compute_inc_ref)
    #     # Initializa bar
    #     bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing z-velocity field...')
    #     # Freq loop
    #     for jf, k0 in enumerate(self.controls.k0):
    #         # Generate kx and ky for evanescent wave grid
    #         kx, ky, kx_grid_f, ky_grid_f =\
    #             self.kxy_init_reg_grid(num_of_grid_pts = int(self.num_of_grid_pts[jf]))
    #         # Filter propagating from evanescent wave grid
    #         kx_eig, ky_eig, n_e = self.filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
    #         # compute evanescent kz
    #         kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
    #         # Stack evanescent kz with propagating kz for incident and reflected grids
    #         k_vec_inc, k_vec_ref = self.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
            
    #         self.uyt_recon[:,jf] = self.compute_u(k_vec_inc, k_vec_ref, k0, 
    #                                               self.pt_recon[:,jf], direction = 'y')
    #         # Compute incident and reflected parts separetly
    #         if compute_inc_ref:
    #             self.uyi_recon[:,jf] = self.compute_u(k_vec_inc, k_vec_ref, k0, 
    #                                                   self.pi_recon[:,jf], direction = 'y')
    #             self.uyr_recon[:,jf] = self.compute_u(k_vec_inc, k_vec_ref, k0, 
    #                                                   self.pr_recon[:,jf], direction = 'y')
    #         bar.update(1)
    #     bar.close()
        
    # def reconstruct_uz(self, compute_inc_ref = False):
    #     """ Reconstruct the sound pressure at a receiver object
        
    #     Parameters
    #     ----------
    #     compute_inc_ref : bool
    #         Whether to reconstruct the incident and reflected fields separatly
            
    #     """
        
    #     self.initialize_uz_recon(num_of_fpts = self.fpts.coord.shape[0], 
    #                        compute_inc_ref = compute_inc_ref)
    #     # Initializa bar
    #     bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing z-velocity field...')
    #     # Freq loop
    #     for jf, k0 in enumerate(self.controls.k0):
    #         # Generate kx and ky for evanescent wave grid
    #         kx, ky, kx_grid_f, ky_grid_f =\
    #             self.kxy_init_reg_grid(num_of_grid_pts = int(self.num_of_grid_pts[jf]))
    #         # Filter propagating from evanescent wave grid
    #         kx_eig, ky_eig, n_e = self.filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
    #         # compute evanescent kz
    #         kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
    #         # Stack evanescent kz with propagating kz for incident and reflected grids
    #         k_vec_inc, k_vec_ref = self.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
            
    #         self.uzt_recon[:,jf] = self.compute_u(k_vec_inc, k_vec_ref, k0, 
    #                                               self.pt_recon[:,jf], direction = 'z')
    #         # Compute incident and reflected parts separetly
    #         if compute_inc_ref:
    #             self.uzi_recon[:,jf] = self.compute_u(k_vec_inc, k_vec_ref, k0, 
    #                                                   self.pi_recon[:,jf], direction = 'z')
    #             self.uzr_recon[:,jf] = self.compute_u(k_vec_inc, k_vec_ref, k0, 
    #                                                   self.pr_recon[:,jf], direction = 'z')
    #         bar.update(1)
    #     bar.close()
        
    def compute_ut(self, k_vec_inc, k_vec_ref, k0, h_mtx, pk, direction = 'z'):
        """ Compute the particle velocity at a given direction
        Parameters
        ----------
        kvec_inc : numpy 1DArray 
            Incident Wave-number vector
        kvec_ref : numpy 1DArray 
            Reflected Wave-number vector
        k0 : float
            Magnitude of wave number
        h_mtx : numpy ndArray
            Sensing matrix for reconstruction
        pk : numpy 1D Array
            WNS
        direction : str
            'x', 'y', or 'z' directions
        """
        if direction == 'z':
            k_vec = np.concatenate((k_vec_inc[:,2], k_vec_ref[:,2]))
        elif direction == 'x':
            k_vec = np.concatenate((k_vec_inc[:,0], k_vec_ref[:,0]))
        elif direction == 'y':
            k_vec = np.concatenate((k_vec_inc[:,1], k_vec_ref[:,1]))
        
        u_dot_specific = np.squeeze(np.asarray(((np.divide(k_vec, k0)) * h_mtx)  @ pk.T))
        return u_dot_specific
    
    def compute_u_inc_or_ref(self, k_vec_inc_or_ref, k0, h_mtx, pk, direction = 'z'):
        """ Compute the incident or reflected particle velocity at a given direction
        Parameters
        ----------
        k_vec_inc_or_ref : numpy 1DArray 
            Incident or reflected Wave-number vector
        k0 : float
            Magnitude of wave number
        h_mtx : numpy ndArray
            Sensing matrix for reconstruction
        pk : numpy 1D Array
            WNS
        direction : str
            'x', 'y', or 'z' directions
        """

        if direction == 'z':
            k_vec = k_vec_inc_or_ref[:,2]
        elif direction == 'x':
            k_vec = k_vec_inc_or_ref[:,0]
        elif direction == 'y':
            k_vec = k_vec_inc_or_ref[:,1]
            
        u_dot_specific = np.squeeze(np.asarray(((np.divide(k_vec, k0)) * h_mtx)  @ pk.T))
        return u_dot_specific
        
    def initialize_p_recon(self, num_of_fpts, compute_inc_ref = False):
        """ Initializes reconstructed pressure vectors
        
        Parameters
        ----------
        num_of_fpts : int
            number of field points to reconstruct at
        compute_inc_ref : bool
            Whether to reconstruct the incident and reflected fields separatly        
        """
        # Total
        self.pt_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                 dtype=complex)
        if compute_inc_ref:
            self.pi_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                 dtype=complex) # Incident
            self.pr_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                 dtype=complex) # Reflected
    
    def initialize_ux_recon(self, num_of_fpts, compute_inc_ref = False):
        """ Initializes reconstructed pressure vectors
        
        Parameters
        ----------
        num_of_fpts : int
            number of field points to reconstruct at
        compute_inc_ref : bool
            Whether to reconstruct the incident and reflected fields separatly
        compute_u_mode : str ('z', 'full')
            'full' means ux, uy, and uz are reconstructed. 'z' means only uz is reconstructed
        """
        self.uxt_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                 dtype=complex)

        if compute_inc_ref:
            self.uxi_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                     dtype=complex)
            self.uxr_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                     dtype=complex)
            
    def initialize_uy_recon(self, num_of_fpts, compute_inc_ref = False):
        """ Initializes reconstructed pressure vectors
        
        Parameters
        ----------
        num_of_fpts : int
            number of field points to reconstruct at
        compute_inc_ref : bool
            Whether to reconstruct the incident and reflected fields separatly
        compute_u_mode : str ('z', 'full')
            'full' means ux, uy, and uz are reconstructed. 'z' means only uz is reconstructed
        """
        self.uyt_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                 dtype=complex)

        if compute_inc_ref:
            self.uyi_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                     dtype=complex)
            self.uyr_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                     dtype=complex)
    
    def initialize_uz_recon(self, num_of_fpts, compute_inc_ref = False):
        """ Initializes reconstructed pressure vectors
        
        Parameters
        ----------
        num_of_fpts : int
            number of field points to reconstruct at
        compute_inc_ref : bool
            Whether to reconstruct the incident and reflected fields separatly
        compute_u_mode : str ('z', 'full')
            'full' means ux, uy, and uz are reconstructed. 'z' means only uz is reconstructed
        """
        self.uzt_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                 dtype=complex)

        if compute_inc_ref:
            self.uzi_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                     dtype=complex)
            self.uzr_recon = np.zeros((num_of_fpts, len(self.controls.k0)), 
                                     dtype=complex)
    
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
        self.pt_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        self.uzt_recon = np.zeros((self.fpts.coord.shape[0], len(self.controls.k0)), dtype=complex)
        if compute_uxy:
            self.uxt_recon = np.zeros((self.fpts.coord.shape[0], len(self.controls.k0)), dtype=complex)
            self.uyt_recon = np.zeros((self.fpts.coord.shape[0], len(self.controls.k0)), dtype=complex)
        # # Generate kx and ky for evanescent wave grid
        # kx_grid, ky_grid = np.meshgrid(self.kx, self.ky)
        # kx_e = kx_grid.flatten()
        # ky_e = ky_grid.flatten()
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing sound field...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # Generate kx and ky for evanescent wave grid
            kx, ky, kx_grid_f, ky_grid_f =\
                self.kxy_init_reg_grid(num_of_grid_pts = int(self.num_of_grid_pts[jf]))
            # Filter propagating from evanescent wave grid
            kx_eig, ky_eig, n_e = filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
            # # compute evanescent kz
            # kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            # # Stack evanescent kz with propagating kz for incident and reflected grids
            # k_vec_inc = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T,
            #     np.array([kx_eig, ky_eig, -kz_eig]).T))
            # k_vec_ref = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
            #     np.array([kx_eig, ky_eig, kz_eig]).T))
            # # The receivers relative to the virtual source planes 
            # recs_inc = np.array([self.fpts.coord[:,0], self.fpts.coord[:,1],
            #     self.fpts.coord[:,2]-self.zm]).T
            # recs_ref = np.array([self.fpts.coord[:,0], self.fpts.coord[:,1],
            #     self.fpts.coord[:,2]-self.zp]).T
            # # Forming the sensing matrix
            # psi_inc = np.exp(-1j * recs_inc @ k_vec_inc.T)
            # psi_ref = np.exp(-1j * recs_ref @ k_vec_ref.T)
            # h_mtx = np.hstack((psi_inc, psi_ref))
            
            # compute evanescent kz
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            # Stack evanescent kz with propagating kz for incident and reflected grids
            k_vec_inc, k_vec_ref = self.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
            # Form sensing matrix
            h_mtx, psi_inc, psi_ref = self.build_hmtx_p(k_vec_inc, k_vec_ref, self.fpts)
            
            # Compute p and uz
            self.pt_recon[:,jf] = np.squeeze(np.asarray(h_mtx @ self.pk[jf].T))
            self.uzt_recon[:,jf] = np.squeeze(np.asarray(-((np.divide(np.concatenate((k_vec_inc[:,2], k_vec_ref[:,2])), k0)) *\
                h_mtx) @ self.pk[jf].T))
            if compute_uxy:
                self.uxt_recon[:,jf] = np.squeeze(np.asarray(((np.divide(np.concatenate(
                    (k_vec_inc[:,0], k_vec_ref[:,0])), k0)) * h_mtx) @ self.pk[jf].T))
                self.uyt_recon[:,jf] = np.squeeze(np.asarray(((np.divide(np.concatenate(
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

    # def plot_colormap(self, freq = 1000, total_pres = True, dinrange = 20):
    #     """Plots a color map of the pressure field.

    #     Parameters
    #     ----------
    #     freq : float
    #         desired frequency of the color map. If the frequency does not exist
    #         on the simulation, then it will choose the frequency just before the target.
    #     total_pres : bool
    #         Whether to plot the total sound pressure (Default = True) or the reflected only.
    #         In the later case, we use the reflectd grid only
    #     dinrange : float
    #         Dinamic range of the color map

    #     Returns
    #     ---------
    #     plt : Figure object
    #     """
    #     id_f = np.where(self.controls.freq <= freq)
    #     id_f = id_f[0][-1]
    #     # color parameter
    #     # color_par = 20*np.log10(np.abs(self.p_recon[:, id_f])/np.amax(np.abs(self.p_recon[:, id_f])))
    #     if total_pres:
    #         color_par = 20*np.log10(np.abs(self.p_recon[:, id_f])/np.amax(np.abs(self.p_recon[:, id_f])))
    #     else:
    #         color_par = 20*np.log10(np.abs(self.p_scat[:, id_f])/np.amax(np.abs(self.p_scat[:, id_f])))
    #         # color_par = np.real(self.p_scat[:, id_f])
    #     # Create triangulazition
    #     triang = tri.Triangulation(self.fpts.coord[:,0], self.fpts.coord[:,2])
    #     # Figure
    #     fig = plt.figure() #figsize=(8, 8)
    #     # fig = plt.figure()
    #     fig.canvas.set_window_title('pressure color map')
    #     plt.title('|P(f)| - reconstructed')
    #     # p = plt.tricontourf(triang, color_par, cmap = 'seismic')
    #     p = plt.tricontourf(triang, color_par, np.linspace(-dinrange, 0, int(dinrange)), cmap = 'seismic')

    #     fig.colorbar(p)
    #     plt.xlabel(r'$x$ [m]')
    #     plt.ylabel(r'$z$ [m]')
    #     return plt

    # def plot_intensity(self, freq = 1000):
    #     """Plots a vector map of the intensity field.

    #     Parameters
    #     ----------
    #     freq : float
    #         desired frequency of the color map. If the frequency does not exist
    #         on the simulation, then it will choose the frequency just before the target.

    #     Returns
    #     ---------
    #     plt : Figure object
    #     """
    #     id_f = np.where(self.controls.freq <= freq)
    #     id_f = id_f[0][-1]
    #     c0 = self.controls.c0
    #     rho0 = self.material.rho0
    #     # Intensities
    #     Ix = 0.5*np.real(self.p_recon[:,id_f] *\
    #         np.conjugate(self.ux_recon[:,id_f]))
    #     Iy = 0.5*np.real(self.p_recon[:,id_f] *\
    #         np.conjugate(self.uy_recon[:,id_f]))
    #     Iz = 0.5*np.real(self.p_recon[:,id_f] *\
    #         np.conjugate(self.uz_recon[:,id_f]))
    #     I = np.sqrt(Ix**2+Iy**2+Iz**2)
    #     # # Figure
    #     fig = plt.figure() #figsize=(8, 8)
    #     fig.canvas.set_window_title('Recon. Intensity distribution map')
    #     cmap = 'viridis'
    #     plt.title('Reconstructed |I|')
    #     # if streamlines:
    #     #     q = plt.streamplot(self.receivers.coord[:,0], self.receivers.coord[:,2],
    #     #         Ix/I, Iz/I, color=I, linewidth=2, cmap=cmap)
    #     #     fig.colorbar(q.lines)
    #     # else:
    #     q = plt.quiver(self.fpts.coord[:,0], self.fpts.coord[:,2],
    #         Ix/I, Iz/I, I, cmap = cmap, width = 0.010)
    #     #fig.colorbar(q)
    #     plt.xlabel(r'$x$ [m]')
    #     plt.ylabel(r'$z$ [m]')
    #     return plt
    
    def get_kxy_data2plot(self, freq = 1000, db = False, dinrange = 20):
        """ Retrieves desired kxy wave-number spectrum for plotting map
        
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
        """
        # Find desired index       
        id_f = utils_insitu.find_freq_index(self.controls.freq, freq_target = freq)
        # Wave number spectrum and k0 value
        k0 = self.controls.k0[id_f]
        pk = np.squeeze(np.asarray(self.pk[id_f]))
        pk_i, pk_r = self.separare_inc_ref_wns(pk)
        # Grid to interpolate on
        kx, ky, kx_grid_f, ky_grid_f =\
            self.kxy_init_reg_grid(num_of_grid_pts = int(self.num_of_grid_pts[id_f]))
        kx_grid, ky_grid = np.meshgrid(kx,ky)
        # Original kxy grid
        kx_eig, ky_eig, n_e = self.filter_evan(k0, kx_grid_f, ky_grid_f, plot=False)
        kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
        kxy, _ = self.form_kxy_ig(k0, kx_eig, ky_eig, kz_eig)
        # Interpolate
        pk_i_grid = griddata(np.real(kxy[:,0:2]), np.abs(pk_i), (kx_grid, ky_grid),
            method='cubic', fill_value=np.finfo(float).eps, rescale=False)
        pk_r_grid = griddata(np.real(kxy[:,0:2]), np.abs(pk_r), (kx_grid, ky_grid),
            method='cubic', fill_value=np.finfo(float).eps, rescale=False)
        # Calculate colors
        if db:
            np.seterr(divide='ignore')
            color_par_i = utils_insitu.get_color_par_dBnorm(pk_i_grid, dinrange = dinrange)
            color_par_r = utils_insitu.get_color_par_dBnorm(pk_r_grid, dinrange = dinrange)           
            # color_range = np.arange(-dinrange, 0.1, 0.1)
            # vmin, vmax = (-dinrange, 0)
            # colorbar_ticks = np.arange(-dinrange, 3, 3)
        else:
            color_par_i = utils_insitu.pressure_normalize(pk_i_grid)
            color_par_r = utils_insitu.pressure_normalize(pk_r_grid)
            # color_range = np.linspace(0, 1, 21)
            # vmin, vmax = (0, 1)
            # colorbar_ticks = np.arange(0, 1.2, 0.2)
        return k0, kx_grid, ky_grid, color_par_i, color_par_r
    
    def get_kxy_scales2plot(self, db = False, dinrange = 20):
        """ Get the vmin, vmax, color range and color ticks of kxy map
        Parameters
        ----------
        db : bool
            Whether to plot in linear scale (default) or decibel scale.
        dinrange : float
            You can specify a dinamic range for the decibel scale. It will not affect the
            linear scale.
        """
        if db:
            vmin, vmax = (-dinrange, 0)
            color_range = np.arange(-dinrange, 0.1, 0.1)
            colorbar_ticks = np.arange(-dinrange, 3, 3)
            colorbar_label = r'$|\bar{P}(k_x, k_y)|$ [dB]'
        else:
            vmin, vmax = (0, 1)
            color_range = np.linspace(0, 1, 250)
            colorbar_ticks = np.arange(0, 1.2, 0.2)
            colorbar_label = r'$|\bar{P}(k_x, k_y)|$ [-]'
        return vmin, vmax, color_range, colorbar_ticks, colorbar_label
            
    def create_pkmap(self, ax, k0, kx_grid, ky_grid, color_par, vmin, vmax, color_range,
                     colorbar_ticks, color_code):
        # Incident
        ax.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
            k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'grey')
        pax = ax.contourf(kx_grid, ky_grid, color_par,
            color_range, vmin = vmin, vmax = vmax, extend='both', cmap = color_code)
        for c in pax.collections:
            c.set_edgecolor("face")
        return pax
    
    def plot_inc_pkmap(self, ax, freq = 1000, db = False, dinrange = 20,
        color_code = 'viridis'):
        """ Plot incident wave number spectrum as a 2D maps (vs. kx and ky)

        Plot the magnitude of the incident wave number spectrum (WNS) as two 2D maps of
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
            color_code : str
                Can be anything that matplotlib supports. Some recomendations given below:
                'viridis' (default) - Perceptually Uniform Sequential
                'Greys' - White (cold) to black (hot)
                'seismic' - Blue (cold) to red (hot) with a white in the middle
            figsize : tuple
                size of the figure
        """
        k0, kx_grid, ky_grid, color_par_i, color_par_r = self.get_kxy_data2plot(freq = freq,
                                                                db = db, dinrange = dinrange)
        
        vmin, vmax, color_range, colorbar_ticks, colorbar_label =\
            self.get_kxy_scales2plot(db = db, dinrange = dinrange)
        
        # fig, ax = plt.subplots(1, 1, figsize = figsize, sharey = True)
        cbar = self.create_pkmap(ax, k0, kx_grid, ky_grid, color_par_i, vmin, vmax, color_range,
                         colorbar_ticks, color_code)
        # ax.set_xlabel(r'$k_x$ [rad/m]')
        # ax.set_ylabel(r'$k_y$ [rad/m]')
        # fig.colorbar(pax, shrink = 0.7, ticks = colorbar_ticks, label = colorbar_label)
        # plt.tight_layout()
        return ax, cbar

    def plot_ref_pkmap(self, ax, freq = 1000, db = False, dinrange = 20,
        color_code = 'viridis'):
        """ Plot the reflected wave number spectrum as a 2D maps (vs. kx and ky)

        Plot the magnitude of the reflected wave number spectrum (WNS) as two 2D maps of
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
            color_code : str
                Can be anything that matplotlib supports. Some recomendations given below:
                'viridis' (default) - Perceptually Uniform Sequential
                'Greys' - White (cold) to black (hot)
                'seismic' - Blue (cold) to red (hot) with a white in the middle
                'inferno' - Nice hot and black map with a Blacksabath plotwist
        """
        k0, kx_grid, ky_grid, color_par_i, color_par_r = self.get_kxy_data2plot(freq = freq,
                                                                db = db, dinrange = dinrange)
        
        vmin, vmax, color_range, colorbar_ticks, colorbar_label =\
            self.get_kxy_scales2plot(db = db, dinrange = dinrange)
        
        # fig, ax = plt.subplots(1, 1, figsize = figsize, sharey = True)
        cbar = self.create_pkmap(ax, k0, kx_grid, ky_grid, color_par_r, vmin, vmax, color_range,
                         colorbar_ticks, color_code)
        #ax.set_xlabel(r'$k_x$ [rad/m]')
        #ax.set_ylabel(r'$k_y$ [rad/m]')
        #fig.colorbar(pax, shrink = 0.7, ticks = colorbar_ticks, label = colorbar_label)
        #plt.tight_layout()
        return ax, cbar
    
    def plot_inc_ref_pkmap(self, freq = 1000, db = False, dinrange = 20,
        color_code = 'viridis', figsize=(10, 5), fine_tune_subplt = [0.1, 0, 0.9, 0.99]):
        """ Plot the incident and reflected wave number spectrum as a 2D maps (vs. kx and ky)

        Plot the magnitude of the incident and reflected wave number spectrum (WNS) as two 2D maps of
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
            color_code : str
                Can be anything that matplotlib supports. Some recomendations given below:
                'viridis' (default) - Perceptually Uniform Sequential
                'Greys' - White (cold) to black (hot)
                'seismic' - Blue (cold) to red (hot) with a white in the middle
            figsize : tuple
                size of the figure
        """
        k0, kx_grid, ky_grid, color_par_i, color_par_r = self.get_kxy_data2plot(freq = freq,
                                                                db = db, dinrange = dinrange)
        
        vmin, vmax, color_range, colorbar_ticks, colorbar_label =\
            self.get_kxy_scales2plot(db = db, dinrange = dinrange)
        
        fig, axs = plt.subplots(1, 2, figsize = figsize, sharey = True)
        cbar = self.create_pkmap(axs[0], k0, kx_grid, ky_grid, color_par_i, vmin, vmax, color_range,
                         colorbar_ticks, color_code)
        
        cbar = self.create_pkmap(axs[1], k0, kx_grid, ky_grid, color_par_r, vmin, vmax, color_range,
                         colorbar_ticks, color_code)
        for i in range(axs.shape[1]):
            axs[i].set_xlabel(r'$k_x$ [rad/m]')
        axs[0].set_ylabel(r'$k_y$ [rad/m]')
        
        cbar_ax_start = 0.2
        cbar_ax = fig.add_axes([0.91, cbar_ax_start, 0.010, 1-2*cbar_ax_start])
        fig.colorbar(cbar, cax = cbar_ax, shrink = 0.7, ticks = colorbar_ticks, 
                     label = colorbar_label)
        fig.subplots_adjust(left = fine_tune_subplt[0], bottom = fine_tune_subplt[1],
                            right = fine_tune_subplt[2], top = fine_tune_subplt[3])
        return fig, axs
    
    def plot_inc_pkmap_severalfreq(self, freqs2plot = [1000], db = False, dinrange = 20,
        color_code = 'viridis', figsize=(10, 5), fine_tune_subplt = [0.1, 0, 0.9, 0.99]):
        
        
        fig, axs = plt.subplots(1, len(freqs2plot), figsize = figsize, 
                                sharey = True, squeeze = False)
        for jf, f2p in enumerate(freqs2plot):
            _, cbar = self.plot_inc_pkmap(axs[0, jf], freq = f2p, db = db, dinrange = dinrange, 
                                             color_code=color_code)
        self.include_cbar_axinfo(fig, axs, cbar, db = db, dinrange = dinrange)
        # finetune subplot
        fig.subplots_adjust(left = fine_tune_subplt[0], bottom = fine_tune_subplt[1],
                            right = fine_tune_subplt[2], top = fine_tune_subplt[3])
        
    def plot_ref_pkmap_severalfreq(self, freqs2plot = [1000], db = False, dinrange = 20,
        color_code = 'viridis', figsize=(10, 5), fine_tune_subplt = [0.1, 0, 0.9, 0.99]):
        
        
        fig, axs = plt.subplots(1, len(freqs2plot), figsize = figsize, 
                                sharey = True, squeeze = False)
        for jf, f2p in enumerate(freqs2plot):
            _, cbar = self.plot_ref_pkmap(axs[0, jf], freq = f2p, db = db, dinrange = dinrange, 
                                             color_code=color_code)
        self.include_cbar_axinfo(fig, axs, cbar, db = db, dinrange = dinrange)
        # finetune subplot
        fig.subplots_adjust(left = fine_tune_subplt[0], bottom = fine_tune_subplt[1],
                            right = fine_tune_subplt[2], top = fine_tune_subplt[3])
        
    def plot_inc_ref_pkmap_severalfreq(self, freqs2plot = [1000], db = False, dinrange = 20,
        color_code = 'viridis', figsize=(10, 5), fine_tune_subplt = [0.1, 0, 0.9, 0.99]):
        
        
        fig, axs = plt.subplots(2, len(freqs2plot), figsize = figsize, 
                                sharey = True, squeeze = False)
        for jf, f2p in enumerate(freqs2plot):
            _, cbar = self.plot_inc_pkmap(axs[0, jf], freq = f2p, db = db, dinrange = dinrange, 
                                             color_code=color_code)
            _, cbar = self.plot_ref_pkmap(axs[1, jf], freq = f2p, db = db, dinrange = dinrange, 
                                             color_code=color_code)
            
        self.include_cbar_axinfo(fig, axs, cbar, db = db, dinrange = dinrange)
        # finetune subplot
        fig.subplots_adjust(left = fine_tune_subplt[0], bottom = fine_tune_subplt[1],
                            right = fine_tune_subplt[2], top = fine_tune_subplt[3])
        return fig, axs

    def include_cbar_axinfo(self, fig, axs, cbar, db = False, dinrange = 20):
        for i in range(axs.shape[1]):
            axs[-1, i].set_xlabel(r'$k_x$ [rad/m]')
        for i in range(axs.shape[0]):
            axs[i, 0].set_ylabel(r'$k_y$ [rad/m]')
        
        vmin, vmax, color_range, colorbar_ticks, colorbar_label =\
            self.get_kxy_scales2plot(db = db, dinrange = dinrange)         
        cbar_ax_start = 0.1#0.2
        cbar_ax = fig.add_axes([0.91, cbar_ax_start, 0.010, 1-2*cbar_ax_start])
        fig.colorbar(cbar, cax = cbar_ax, shrink = 0.7, ticks = colorbar_ticks, 
                     label = colorbar_label)
        # return fig, axs
        
    
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
            
    def get_reflected_wns(self, freq = 1000, true_directivity = False):
        """ Get one frequency of the reflected wave-number spectrum
        """
        # get freq index
        id_f = utils_insitu.find_freq_index(self.controls.freq, freq)
        
        # wavenumber in air
        k0 = self.controls.k0[id_f]
        # propagating plane wave directions
        directions = k0 * np.array([self.prop_waves_dir.coord[:,0], 
                                    self.prop_waves_dir.coord[:,1], 
                                    -self.prop_waves_dir.coord[:,2]]).T
        
        # k-space (total)
        pk = np.squeeze(np.asarray(self.pk[id_f]))
        
        # Directivity k-space (reflected)
        pk_r = pk[int(len(pk)/2):int(len(pk)/2)+ self.prop_waves_dir.n_prop] # reflected
        if true_directivity:
            pk_r = (directions[:,2]) * pk_r
        
        return pk_r
    
    def get_reflected_wns_allfreq(self, true_directivity = False):
        """ Get all frequencies of the reflected wave-number spectrum
        """
        pk_r = np.zeros((self.prop_waves_dir.n_prop, len(self.controls.freq)), dtype = complex)
        for jf, f in enumerate(self.controls.freq):
            pk_r[:, jf] = self.get_reflected_wns(freq = f, true_directivity = true_directivity)
        return pk_r
            
        
        

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
        # id_f = utils_insitu.find_freq_index(self.controls.freq, freq)
        
        # # wavenumber in air
        # k0 = self.controls.k0[id_f]
        # # propagating plane wave directions
        # directions = k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T
        
        # # k-space (total)
        # pk = np.squeeze(np.asarray(self.pk[id_f]))
        
        # # Directivity k-space (reflected)
        # pk_r = pk[int(len(pk)/2):int(len(pk)/2)+self.n_prop] # reflected
        # if true_directivity:
        #     pk_r = (directions[:,2]) * pk_r
        
        pk_r = self.get_reflected_wns(freq = freq, true_directivity = true_directivity)
        
        # Figure
        plt3Ddir = utils_insitu.Plot3Ddirectivity(pressure = pk_r, 
                                                  coords = self.prop_waves_dir.coord, 
            connectivities =  self.prop_waves_dir.connectivities, dinrange = dinrange, 
            color_method = color_method, radius_method = radius_method, 
            color_map = color_code, view = view, eye_dict = eye, 
            renderer = renderer, remove_cart_axis = True, create_sph_axis = True, 
            azimuth_grid_color = 'grey', elevation_grid_color = 'grey',
            num_of_radius = 3, delta_azimuth = 45, delta_elevation = 15, line_style = 'dot',
            plot_elevation_grid = True, font_family = "Palatino Linotype", font_size = 14,
            colorbar_title = 'Normalized Scattered Pressure [dB]', fig_size=dpi)
        plt3Ddir.plot_3d_polar()
        # #return balloon_data #x, y, z, conectivities, r, plot_pressure
        # fig, trace = utils_insitu.plot_3d_polar(self.pdir, self.conectivities,
        #      pk_r, dinrange = dinrange, 
        #      color_method = color_method, radius_method = radius_method,
        #      color_code = color_code, view = view, eye = eye, renderer = renderer,
        #      remove_axis = remove_axis)
        return plt3Ddir.fig, plt3Ddir.trace


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
                 delta_x = 0.05, delta_y = 0.05, regu_par = 'L-curve'):
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
        DecompositionEv2.__init__(self, p_mtx, controls,  receivers,
                                  delta_x, delta_y, regu_par, material)
        super().__init__(p_mtx, controls, receivers, delta_x, delta_y, regu_par, material)

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
        Zs_pt = np.divide(self.pt_recon, self.uzt_recon)
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
    
################ DUMP ###############
# def plot_ref_pkmap(self, freq = 1000, db = False, dinrange = 20,
#     save = False, fig_title = '', path = '', fname='', color_code = 'viridis',
#     plot_incident = True, dpi = 600, figsize=(8, 8), fileformat='png'):
#     """ Plot wave number spectrum as a 2D maps (vs. kx and ky)

#     Plot the magnitude of the wave number spectrum (WNS) as two 2D maps of
#     evanescent and propagating waves. The map is first interpolated into
#     a regular grid. It is a normalized version of the magnitude, either between
#     0 and 1 or between -dinrange and 0. The maps are ploted as color as function
#     of kx and ky. The radiation circle is also ploted.

#     Parameters
#     ----------
#         freq : float
#             Which frequency you want to see. If the calculated spectrum does not contain it
#             we plot the closest frequency before the target.
#         db : bool
#             Whether to plot in linear scale (default) or decibel scale.
#         dinrange : float
#             You can specify a dinamic range for the decibel scale. It will not affect the
#             linear scale.
#         save : bool
#             Whether to save or not the figure. PDF file with simple standard name
#         fig_title : str
#             Title of the figure file #FixMe
#         path : str
#             Path to save the figure file
#         fname : str
#             File name to save the figure file
#         color_code : str
#             Can be anything that matplotlib supports. Some recomendations given below:
#             'viridis' (default) - Perceptually Uniform Sequential
#             'Greys' - White (cold) to black (hot)
#             'seismic' - Blue (cold) to red (hot) with a white in the middle
#         plot_incident : bool
#             Whether to plot incident WNS or not
#         dpi : float
#             dpi of figure - to save
#         figsize : tuple
#             size of the figure
#     """
#     k0, kx_grid, ky_grid, color_par_i, color_par_r = self.get_kxy_data2plot(freq = freq,
#                                                             db = db, dinrange = dinrange)
    
#     vmin, vmax, color_range, colorbar_ticks, colorbar_label =\
#         self.get_kxy_scales2plot(db = db, dinrange = dinrange)
    
#     fig, ax = plt.subplots(1, 1, figsize = figsize, sharey = True)
#     pax = self.create_pkmap(ax, k0, kx_grid, ky_grid, color_par_r, vmin, vmax, color_range,
#                      colorbar_ticks, color_code)
#     fig.colorbar(pax, shrink=1.0, ticks = colorbar_ticks, label = colorbar_label)
#     plt.tight_layout()
#     return fig, ax

#     # # Figure
#     # if plot_incident:
#     #     fig = plt.figure(figsize=figsize)
#     #     # Incident
#     #     plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
#     #         k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'grey')
#     #     p = plt.contourf(kx_grid, ky_grid, color_par_i,
#     #         color_range, vmin = vmin, vmax = vmax, extend='both', cmap = color_code)
#     #     fig.colorbar(p, shrink=1.0, ticks = colorbar_ticks, label = r'$|\bar{P}_i(x, y)|$ dB')
#     #     for c in p.collections:
#     #         c.set_edgecolor("face")
#     #     plt.xlabel(r'$k_x$ [rad/m]')
#     #     plt.ylabel(r'$k_y$ [rad/m]')
#     #     plt.tight_layout()
#     #     if save:
#     #         filename = path + fname + '_' + str(int(freq)) + 'Hz_i'
#     #         plt.savefig(fname = filename, format=fileformat, dpi = dpi)
#     #     # Reflected
#     #     fig = plt.figure(figsize=figsize)
#     #     plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
#     #             k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'grey')
#     #     p = plt.contourf(kx_grid, ky_grid, color_par_r,
#     #         color_range, vmin = vmin, vmax = vmax, extend='both', cmap = color_code)
#     #     fig.colorbar(p, shrink=1.0, ticks = colorbar_ticks, label = r'$|\bar{P}_r(x, y)|$ dB')
#     #     for c in p.collections:
#     #         c.set_edgecolor("face")
#     #     plt.xlabel(r'$k_x$ [rad/m]')
#     #     plt.ylabel(r'$k_y$ [rad/m]')
#     #     plt.tight_layout()
#     #     if save:
#     #         filename = path + fname + '_' + str(int(freq)) + 'Hz_r'
#     #         plt.savefig(fname = filename, format=fileformat, dpi = dpi)
#     # else:
#     #     fig = plt.figure(figsize=figsize)
#     #     #fig.canvas.set_window_title('Reflected WNS - PEIG')
#     #     # Reflected
#     #     #plt.title('Reflected: ' + fig_title)
#     #     plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
#     #             k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'grey')
#     #     p = plt.contourf(kx_grid, ky_grid, color_par_r,
#     #         color_range, extend='both', cmap = color_code)
#     #     fig.colorbar(p, shrink=1.0, ticks=np.arange(-dinrange, 3, 3), label = r'$|\bar{P}_r(x, y)|$ dB')
#     #     for c in p.collections:
#     #         c.set_edgecolor("face")
#     #     plt.xlabel(r'$k_x$ [rad/m]')
#     #     plt.ylabel(r'$k_y$ [rad/m]')
#     #     plt.tight_layout()
#     #     if save:
#     #         filename = path + fname + '_' + str(int(freq)) + 'Hz_r.pdf'
#     #         plt.savefig(fname = filename, format=fileformat, dpi = dpi, bbox_inches='tight')