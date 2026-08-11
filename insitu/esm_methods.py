# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
import plotly
from tqdm import tqdm
import scipy

import lcurve_functions as lc
import utils_insitu as ut_is
# from controlsair import AlgControls, AirProperties
# from sources import Source
from receivers import Receiver

class ESMBase(object):
    """ Decomposition of the sound field using the Equivalent Source Method.
    
    Base class for several others. 
    The intention is to have classes for: free-field radiation, radiation over baffle, 
    free-field rigid scattering, free-field non-rigid scattering, 
    rigid scattering over baffle, non-rigid scattering over baffle.
    """
    def __init__(self, p_mtx=None, air = None, controls=None, receivers=None, source = None, 
                 regu_par = 'L-curve', geometry = None):
        """ Class Init
        
        Our origin will be assumed as the x-y center of the scatterer (radiating) 
        object. Its z-coordinate is max(z).
        
        Parameters
        ----------
        p_mtx : numpy ndArray 
            A matrix (N_rec x N_freq) containing the complex amplitudes of all the receivers
            Each column is a set of sound pressure at all receivers for a frequency.
        controls : object (AlgControls)
            Controls of the decomposition (frequency spam)
        receivers : object (Receiver)
            The receivers in the field
        source : object (Source)
            The sound source properties
        regu_par : str
            Automatic choice of regularization parameter. Default is "L-curve". It can
            be "L-curve" or l-curve for L-curve choice; or "gcv" or "GCV" for generalized
            cross-validation. Any other choice reverts do default.
        bbox_size : numpy 1dArray
            Box size in x, y, z dimensions. Updates authomatically if the user provides
            the diffuser geometry
        rec_bbox : bool
            Wether the bounding box is rectangular (True - default) or cylindrical
        geometry : dict
            Dictionary containing the coordinates and the connectivities of the diffuser. 
            Used to update the bounding box properties.

        The objects are stored as attributes in the class (easier to retrieve).
        """
        # Inputs
        self.pres_s = p_mtx
        self.air = air
        self.controls = controls
        self.receivers = receivers
        self.source = source
        self.geometry = geometry        

        # Regularization strategy
        if regu_par == 'L-curve' or regu_par == 'l-curve':
            self.regu_par_fun = lc.l_curve
            print("You choose L-curve to find optimal regularization parameter")
        elif regu_par == 'gcv' or regu_par == 'GCV':
            self.regu_par_fun = lc.gcv_lambda
            print("You choose GCV to find optimal regularization parameter")
        elif regu_par == 'ncp' or regu_par == 'NCP':
            self.regu_par_fun = lc.ncp
            print("You choose NCP to find optimal regularization parameter")
        else:
            self.regu_par_fun = lc.l_curve
            print("Returning to default L-curve to find optimal regularization parameter")
        
        # Other variables
        self.source_es = None # Equiv. Sources - sound source
        self.eq_sources = None  # Equiv. Sources - main scattering/radiating object
        self.num_sources = 0 # For radiation problems
           
    def get_dx(self, n_s_lam = 4, freq = 1000):
        """ computes minimum samplind distace based on a frequency and n_s_lam
        
        Parameters
        ----------
        n_s_lam : int
            number of elements per wavelength
        freq : float
            frequency value
            
        Returns
        ----------
        dx : float
            sampling distance
        """
        lambda_min = self.air.c0/freq
        dx = lambda_min/n_s_lam
        return dx
            
    def get_rec_bbox_size(self):
        """ Get rectangular bounding box size with the geometry information
        
        Returns
        ---------
        self.bbox_size : numpy1dArray
            Bounding box size with size Lx, Ly, Lz    
        """
        Lx = np.amax(self.geometry['coords'][:,0]) - np.amin(self.geometry['coords'][:,0])
        Ly = np.amax(self.geometry['coords'][:,1]) - np.amin(self.geometry['coords'][:,1])
        Lz = np.amax(self.geometry['coords'][:,2]) - np.amin(self.geometry['coords'][:,2])
        self.bbox_size = np.array([Lx, Ly, Lz])
        return self.bbox_size
    
    def get_cyl_bbox_size(self):
        """ Get cylindrical bounding box size with the geometry information
        
        Returns
        ---------
        self.bbox_size : numpy1dArray
            Bounding box size with size radii, Lz
        """
        radii = np.amax(self.geometry['coords'][:,0])
        Lz = np.abs(np.amax(self.geometry['coords'][:,2]) - np.amin(self.geometry['coords'][:,2]))
        self.bbox_size = np.array([radii, Lz])
        return self.bbox_size
    
    def rectangular_bbox(self, bbox_size = [0.6, 0.6, 0.5]):
        """ Get the rectangular bounding box information - given its size  
        
        Rectangular bounding box information cosists in its vertices, faces, normals,
        and edges. Such information will be used for plottting and sampling the 
        equivalent sources.
        
        Parameters
        ----------
        bbox_size : numpy 1dArray
            Box size in x, y, z dimensions.
        """
        # Set rectangular bounding box to True
        self.rectangular_bbox = True
        
        # Choose between specified dimetions or geometry
        if self.geometry is None:
            self.bbox_size = np.array(bbox_size)
        else:
            self.bbox_size = self.get_rec_bbox_size()
        
        Lx, Ly, Lz = self.bbox_size[0], self.bbox_size[1], self.bbox_size[2]
        # Verticies
        self.bbox_verts = np.array([
            [-Lx/2, -Ly/2, 0], 
            [-Lx/2, Ly/2, 0],
            [Lx/2, Ly/2, 0],
            [Lx/2, -Ly/2, 0],
            [-Lx/2, -Ly/2, -Lz], 
            [-Lx/2, Ly/2, -Lz],
            [Lx/2, Ly/2, -Lz],
            [Lx/2, -Ly/2, -Lz]])
        # Faces (connectivity)
        self.bbox_faces = np.array([
            [0, 3, 2, 1], # top
            [4, 7, 3, 0], # x side 1
            [5, 1, 2, 6], # x side 2
            [4, 0, 1, 5], # y side 1
            [7, 6, 2, 3], # y side 2
            [4, 5, 6, 7], # bottom
            ])
        # Normals
        self.bbox_normals = np.array([
            [0, 0, 1], # top
            [0, -1, 0], # x side 1
            [0, 1, 0], # x side 2
            [-1, 0, 0], # y side 1
            [1, 0, 0], # y side 2
            [0, 0, -1], # bottom
            ])
        # Edges (connectivity)
        self.bbox_edges = np.array([
            [0,3],[3,2],[2,1],[1,0],
            [4,5],[5,6],[6,7],[7,4],
            [4,0],[5,1],[6,2], [7,3]])
        
    def cylindrical_bbox(self, bbox_size = [0.6, 0.25]):
        """ Get the cylindrical bounding box information - given its size
        
        Cylindrical bounding box information cosists in its vertices, faces, 
        and edges. Such information will be used for plottting and sampling the 
        equivalent sources.
        
        Parameters
        ----------
        bbox_size : numpy 1dArray
            Box size in x, y, z dimensions.
        """
        # Set rectangular bounding box to False
        self.rectangular_bbox = False
        
        # Choose between specified dimetions or geometry
        if self.geometry is None:
            self.bbox_size = np.array(bbox_size)
        else:
            self.bbox_size = self.get_cyl_bbox_size()
        
        radii, Lz = self.bbox_size[0], self.bbox_size[1]
        # Sampling Phi
        n_phi = 72
        phi = np.linspace(0, 2*np.pi, n_phi)
        
        # Top cilinder vertices (across circunference)
        self.bbox_verts = np.array([radii*np.cos(phi),
                                    radii*np.sin(phi),
                                    np.zeros(n_phi)]).T
        # Bottom cilinder vertices (across circunference)
        self.bbox_verts = np.vstack((self.bbox_verts,
                                     np.array([radii*np.cos(phi),
                                               radii*np.sin(phi),
                                               np.zeros(n_phi)-Lz]).T))
        self.bbox_verts = np.vstack((self.bbox_verts, np.array([0.0,0.0,0.0]),
                                     np.array([0.0,0.0,-Lz])))
        # Bounding box edges (connectivity)
        bbox_edges = []
        for je in range(n_phi-1):
            bbox_edges.append(np.array([je, je+1])) # top
            bbox_edges.append(np.array([je+n_phi, je+n_phi+1])) # bottom
            bbox_edges.append(np.array([je, je+n_phi])) # sides         
        self.bbox_edges = np.array(bbox_edges)
        # Bounding box faces (connectivity)
        bbox_faces = []
        for je in range(n_phi-1):
            bbox_faces.append(np.array([je, je+1, 2*n_phi])) # top
            bbox_faces.append(np.array([je+n_phi+1, je+n_phi,  2*n_phi+1])) # top
            bbox_faces.append(np.array([je, je+n_phi, je+1])) # sides
            bbox_faces.append(np.array([je+n_phi, je+n_phi+1, je+1])) # sides
        self.bbox_faces = np.array(bbox_faces)
    
    def sample_bbox(self, dx = 0.05, while_step = 100):
        """ Sample specified bounding box
        """
        if self.rectangular_bbox:
            _ = self.sample_rec_bbox(dx = dx)
        else:
            _ = self.sample_cyl_bbox(dx = dx, while_step = while_step)
    
    def sample_straight_edges(self, verts, idp = [0,1], dx = 0.05):
        """ Sample pts across 2 straight edges of rectangular bounding box
        
        Parameters
        ----------
        verts : numpyndArray
            Vetices of the plane to be sampled.
        idp : list
            list of 2 indexes of our plane to be sampled.
        dx : float
            sampling distance
        """
        # loop indexes to get the edges sampled
        edges_samples = []
        for jp in idp:
            dmin, dmax = np.amin(verts[:,jp]), np.amax(verts[:,jp])
            n_samples = int(np.ceil(np.abs(dmax-dmin)/dx))
            edges_samples.append(np.linspace(dmin, dmax, n_samples)) 
        # Meshgrid it
        x_g, y_g = np.meshgrid(edges_samples[0], edges_samples[1])
        # Flatten
        x_s, y_s = x_g.flatten(), y_g.flatten()
        return x_s, y_s
    
    def sample_rec_face(self, faceid = 0, dx = 0.05, tol = 1e-6):
        """ sample points on a face of the rectangular bounding box
        
        Parameters
        ----------
        faceid : int
            face index to sample
        dx : float
            sampling distance
        tol : float
            tolerance for determining unique pts
        """
        # Get relevant vertices
        verts = self.bbox_verts[self.bbox_faces[faceid]]  
        # get indices of zero normal (that is our plane) and non-zero normal 
        idp = np.where(self.bbox_normals[faceid,:] == 0)[0]
        idn = np.where(self.bbox_normals[faceid,:] != 0)[0]
        x_s, y_s = self.sample_straight_edges(verts = verts, dx = dx, idp = idp)
        # Init eq_sources
        eq_sources = np.zeros((x_s.shape[0],3))
        # fill eq_sources
        eq_sources[:, idp[0]] = x_s
        eq_sources[:, idp[1]] = y_s
        eq_sources[:, idn[0]] = verts[0, idn[0]] * np.ones(x_s.shape)
        # rounding + unique
        eq_sources = np.round(eq_sources / tol) * tol
        return np.unique(eq_sources, axis=0)
    
    def sample_rec_bbox(self, dx = 0.05, tol = 1e-6):
        """ sample points on all faces of the rectangular bounding box
        
        Parameters
        ----------
        dx : float
            sampling distance
        tol : float
            tolerance for determining unique pts
        """
        # zero-th face
        eq_sources = self.sample_rec_face(faceid = 0, dx = dx, tol = tol)
        for jf in range(1,6):
            eq_sources = np.vstack((eq_sources, 
                                   self.sample_rec_face(faceid = jf, dx = dx, tol = tol)))
        # rounding + unique
        eq_sources = np.round(eq_sources / tol) * tol
        self.eq_sources = np.unique(eq_sources, axis=0)
        return self.eq_sources
    
    def sample_cyl_bbox(self, dx = 0.05, tol = 1e-6, while_step = 50):
        """ sample points on all faces of the cylindrical bounding box
        
        Parameters
        ----------
        dx : float
            sampling distance
        tol : float
            tolerance for determining unique pts
        while_step : int
            step to increase the number of receivers in the sunflower array, 
            which samples the top and bottom of the cylinder.
        """
        # number of pts accross the circunference and side
        n_phi = int(np.ceil(2*np.pi*self.bbox_size[0]/dx))
        n_Lz = int(np.ceil(self.bbox_size[1]/dx))
        phi_vec = np.linspace(0, 2*np.pi, n_phi)
        # Sample side
        eq_sources_side = np.zeros((n_phi*n_Lz, 3))
        for jp, phi in enumerate(phi_vec): 
            x = self.bbox_size[0]*np.cos(phi) * np.ones(n_Lz)
            y = self.bbox_size[0]*np.sin(phi) * np.ones(n_Lz)
            z = np.linspace(-self.bbox_size[1] + 0.5*dx, -0.5*dx, n_Lz)      
            eq_sources_side[jp*n_Lz:(jp+1)*n_Lz,:] = np.array([x, y, z]).T
            
        # sample top
        max_min_dist = 2*dx
        n_sf = int(0.9*n_phi)
        sf = Receiver()
        while max_min_dist > dx:
            sf.sunflower_circular_array(n_recs = n_sf, radius = self.bbox_size[0], 
                                        alpha = 2, zr = 0.0)
            sf.rotate_array(axis = 'z', theta_deg = 22.5)
            max_min_dist, _, _ = sf.compute_min_distances(sf.coord)
            n_sf += while_step
        eq_sources = np.vstack((eq_sources_side, sf.coord))
        # Sample bottom
        sf.translate_array(axis = 'z', delta = -self.bbox_size[1])
        eq_sources = np.vstack((eq_sources, sf.coord))
        # rounding + unique 
        eq_sources = np.round(eq_sources / tol) * tol
        self.eq_sources = np.unique(eq_sources, axis=0)
        return self.eq_sources
    
    def get_rm_rs_dist(self,):
        """ Computes the Euclidian norm between all receiver to source (equivalent)
        """
        if self.source_es is None:
            rs_mtx = np.copy(self.source.coord)
        else:
            rs_mtx = np.copy(self.source_es.coord)
        rm_rs = scipy.spatial.distance.cdist(self.receivers.coord, rs_mtx, metric='euclidean')
        return rm_rs
    
    def get_rm_rq_dist(self, receivers):
        """ Computes the Euclidian norm between all receiver to equivalent sources
        """
        rm_rq = scipy.spatial.distance.cdist(receivers.coord, self.eq_sources, metric='euclidean')
        return rm_rq
    
    def get_cosmq_dist(self, receivers, rm_rq):
        """ Computes the cosine between all receiver to equivalent sources
        """
        zm = np.repeat(np.array([receivers.coord[:,2]]).T, rm_rq.shape[1], axis = 1)
        zq = np.repeat(np.array([self.eq_sources[:,2]]), rm_rq.shape[0], axis = 0)
        cos_mq = (zm-zq)/rm_rq
        return cos_mq
    
    def kernel_g_m(self, k0, r):  
        """ Green's function (Monopole version)
        
        Parameters
        ----------
        k0 : float
            wave-number magnitude value in [rad/m]
        r : numpyndArray
            MxN matrix with the Euclidan norm distances
        """
        g_fun = (np.exp(-1j * k0 * r)) / r
        return g_fun
                
    def kernel_g_d(self, k0, r, cosmq):  
        """ Green's function (Dipole version)
        
        Parameters
        ----------
        k0 : float
            wave-number magnitude value in [rad/m]
        r : numpyndArray
            MxN matrix with the Euclidan norm distances
        cosmq : numpyndArray
            MxN matrix with the Cosine values
        """
        dg_fun = (-1j*k0/(4*np.pi))*cosmq*((np.exp(-1j * k0 * r)) / r)*(1 + (1 / (1j * k0 * r)))
        return dg_fun
    
    def solve_freq(self, g_mtx, pm, method='tikhonov', plot_reg_curve = False):
        """ Regularized single frequency solver
        
        Parameters
        ----------
        g_mtx : numpyndArray
            Problem sensing matrix
        pm : numpy1dArray
            Measured data vs. space at a frequency
        method : str
            solving method. Default is 'tikhonov',
        plot_reg_curve : bool
            whether to plot the regularization curve.
        
        Returns
        ----------
        x : numpy1dArray
            estimated solution
        lambd_value: estimated regularization parameter
        """
        # Measured data
        pm = pm.astype(complex)
        # Compute SVD
        u, sig, v = lc.csvd(g_mtx.astype(complex))
        cond_num = sig[0]/sig[-1]
        # Find the optimal regularization parameter.
        lambd_value = self.regu_par_fun(u, sig, pm, plot_reg_curve)
        # Solve system          
        if method == 'direct':
            Hm = np.matrix(g_mtx)
            x = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
        elif method == 'Ridge':
            x = lc.ridge_solver(g_mtx,pm,lambd_value)
        elif method == 'Tikhonov':
            x = lc.tikhonov(u,sig,v,pm,lambd_value)
        elif method == 'cvx':
            x = lc.cvx_tikhonov(g_mtx, pm, lambd_value, l_norm = 2)
        else:
            x = lc.tikhonov(u,sig,v,pm,lambd_value)
        return x, lambd_value, cond_num

    def plot_sensing_mtx(self, g_mtx):
        """ Plots the sensing matrix
        
        Parameters
        ----------
        g_mtx : numpyndArray
            MxN sensing Matrix
        """
        # Normalize
        g_mtx_max = np.amax(np.abs(g_mtx.flatten()))
        # Figure fig size
        g_size = np.array(g_mtx.shape)
        fax = g_size.max()/g_size.min()
        if g_mtx.shape[0] <= g_mtx.shape[1]: # Overdetermined case
            figsize = (3*fax, 3)
        else:
            figsize = (3, 3*fax)
        plt.figure(figsize=figsize)
        # imshow displays the matrix; vmin/vmax anchor the color scale limits
        im = plt.imshow(20*np.log10(np.abs(g_mtx/g_mtx_max)), vmin = -40, vmax = 0, 
                        cmap='bwr')
        # im = plt.imshow(np.abs(g_mtx/g_mtx_max), vmin = 0.5, vmax =1, 
        #                 cmap='bwr')
        cbar = plt.colorbar(im, fraction=0.02, pad=0.02)
        cbar.set_label(r'$|G|$ [dB]', rotation=90, labelpad=15)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.tight_layout()
    
    def plotly_scene(self, renderer = 'browser', xyz_size = [1,1,1],
                     plot_bbox = True):
        """ plot scene using plotly
        
        Parameters
        ----------
        renderer : str
            plotly renderer (browser, notebook)
        """
        # renderer
        plotly.io.renderers.default = renderer
        
        # Figure isntantiation
        xmax, ymax = xyz_size[0], xyz_size[1]
        fig = plotly.graph_objs.Figure()
        
        # Bounding box
        if plot_bbox:
            # BBox vertices
            vertex_trace = self.get_verts_trace()
            # BBox edges
            edge_trace =  self.get_edges_trace()
            # BBox Faces
            face_trace =  self.get_faces_trace()
            fig.add_traces([vertex_trace, edge_trace, face_trace])

        # Diffuser
        if self.geometry is not None:
            geo_trace = self.get_geo_trace()
            fig.add_traces(geo_trace)
        
        # Source
        if self.source is not None:
            source_trace = self.get_source_trace()
            fig.add_traces(source_trace)
            xyz_size[2] = 1.1*self.source.coord[0,2]            
            
        # Receivers
        if self.receivers is not None:
            rec_trace = self.get_receivers_trace()
            fig.add_traces(rec_trace)
            x = np.amax(np.array([np.amax(self.receivers.coord[:,0]), xyz_size[0]]))
            y = np.amax(np.array([np.amax(self.receivers.coord[:,1]), xyz_size[1]]))
            xmax, ymax = np.sqrt(x**2 + y**2), np.sqrt(x**2 + y**2)
        
        # Source - equiv.sources
        if self.source_es is not None:
            ses_trace = self.get_souce_es_trace()
            fig.add_traces(ses_trace)
        
        # Main scattering/radiating - Equivalent sources
        if self.eq_sources is not None:
            eqs_trace =  self.get_es_trace()
            fig.add_traces(eqs_trace)
        # x, y, z - limits
        fig.update_layout(scene=dict(xaxis=dict(range=[-xmax/2, xmax/2]),
                                     yaxis=dict(range=[-ymax/2, ymax/2]),
                                     zaxis=dict(range=[np.amin(self.bbox_verts[:,2])-0.1, xyz_size[2]])))
        return fig
    
    def get_verts_trace(self):
        """ Bbox plotly vertices trace
        """
        # BBox vertices
        vertex_trace = plotly.graph_objs.Scatter3d(
            x = self.bbox_verts[:,0], y = self.bbox_verts[:,1], z = self.bbox_verts[:,2],
            mode='markers', marker=dict(size=3, color='grey'), name='Bbox-V')
        return vertex_trace
    
    def get_edges_trace(self):
        """ Bbox plotly edges trace
        """
        Xe, Ye, Ze = [], [], []
        for i, j in self.bbox_edges:
            Xe += [self.bbox_verts[i,0], self.bbox_verts[j,0], None]
            Ye += [self.bbox_verts[i,1], self.bbox_verts[j,1], None]
            Ze += [self.bbox_verts[i,2], self.bbox_verts[j,2], None]
        
        edge_trace =  plotly.graph_objs.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines',
                                                  line=dict(color='rgba(0, 50, 0, 0.5)', 
                                                            width=4), name='Bbox-E')
        return edge_trace
    
    def get_faces_trace(self):
        """ Bbox plotly faces trace
        """
        face_tri = []
        for f in self.bbox_faces:
            face_tri.append([f[0], f[1], f[2]])
            if len(f) == 4:
                face_tri.append([f[0], f[2], f[3]])
        face_tri = np.array(face_tri)
        face_trace =  plotly.graph_objs.Mesh3d(
            x=self.bbox_verts[:,0], y=self.bbox_verts[:,1], z=self.bbox_verts[:,2],
            i=face_tri[:,0], j=face_tri[:,1], k=face_tri[:,2],
            color='aquamarine',      # or 'lightgrey'
            opacity=0.3,           # nice and transparent
            flatshading=True, hoverinfo='skip', showscale=False, showlegend=True, name='Bbox-F')
        return face_trace
        
    def get_geo_trace(self):
        """ Bbox plotly geometry trace
        """
        # coords and connectivities
        coords = self.geometry['coords']
        connectivities = self.geometry['connectivities']
        trisurf = plotly.figure_factory.create_trisurf(
            x = coords[:, 0], y = coords[:, 1], z = coords[:, 2],
            simplices = connectivities, show_colorbar = False, 
            color_func = connectivities.shape[0] * ["rgb(222, 184, 135)"], edges_color = 'black')
        for tr in trisurf.data:
            tr.showlegend = True
            tr.name = "Geometry"
        return trisurf.data
    
    def get_es_trace(self,):
        """ Bbox plotly equivalent sources trace
        """
        eqs_trace =  plotly.graph_objs.Scatter3d(
            x = self.eq_sources[:,0], y = self.eq_sources[:,1], z = self.eq_sources[:,2], 
            mode='markers', marker=dict(size=2, color='blue'), name='Eq. Sources')
        return eqs_trace
    
    def get_source_trace(self):
        """ Source trace
        """
        # Source coords
        source_trace = plotly.graph_objs.Scatter3d(
            x = self.source.coord[:,0], y = self.source.coord[:,1], z = self.source.coord[:,2],
            mode='markers', marker=dict(size=7, color='red', symbol='diamond'), name='Source',
            opacity=0.6)
        return source_trace
    
    def get_receivers_trace(self):
        """ Source trace
        """
        # Receiver coords
        rec_trace = plotly.graph_objs.Scatter3d(
            x = self.receivers.coord[:,0], y = self.receivers.coord[:,1], z = self.receivers.coord[:,2],
            mode='markers', marker=dict(size=3, color='mediumorchid'), name='Receivers',
            opacity=0.6)
        return rec_trace
        
    def get_souce_es_trace(self):
        """ Source trace
        """
        # Receiver coords
        ses_trace = plotly.graph_objs.Scatter3d(
            x = self.source_es.coord[:,0], y = self.source_es.coord[:,1], z = self.source_es.coord[:,2],
            mode='markers', marker=dict(size=2, color='blue'), name='Eq. Sources',
            opacity=0.6)
        return ses_trace
    
    def plot_directivity(self, semi_sphere, freq = 1000, dinrange = 45,
        save = False, fig_title = '', path = '', fname='', color_code = 'turbo',
        dpi = 600, figsize=(8, 8), fileformat='png',
        color_method = 'dB', radius_method = 'dB',
        view = 'iso_z', eye = None, renderer = 'browser',
        remove_axis = False):
        """ Plot directivity as a 3D maps 
        
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
        idf = ut_is.find_freq_index(self.fc, freq)
        
        # Figure
        plt3Ddir = ut_is.Plot3Ddirectivity(pressure = self.dir_oct[:, idf],
                                           coords = semi_sphere.coord, 
                                           connectivities =  semi_sphere.connectivities, 
                                           dinrange = dinrange,
                                           color_method = color_method, radius_method = radius_method, 
            color_map = color_code, view = view, eye_dict = eye, 
            renderer = renderer, remove_cart_axis = True, create_sph_axis = True, 
            azimuth_grid_color = 'grey', elevation_grid_color = 'grey',
            num_of_radius = 3, delta_azimuth = 45, delta_elevation = 15, line_style = 'dot',
            plot_elevation_grid = True, font_family = "Palatino Linotype", font_size = 14,
            colorbar_title = 'Normalized Scattered Pressure [dB]', fig_size=dpi)
        plt3Ddir.plot_3d_polar()
        return plt3Ddir.fig, plt3Ddir.trace

class RISESM(ESMBase):
    def __init__(self, p_mtx=None, air = None, controls=None, receivers=None, source = None, 
                 regu_par = 'L-curve', geometry = None):
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
        super().__init__(p_mtx, air, controls, receivers, source, regu_par, geometry)
    
    def sample_source(self, source_radii = 0.1, sampling_scheme = 'octahedron'):
        """ sample equivalent sources arond the sound source
        
        Parameters
        ----------
        source_radii : float
            source radius
        sampling_scheme : str
            type of samplig scheme. It can be 'single', tetrahedron, 'octahedron', 'cube',
            'dodecahedron', 'icosahedron'. Default is 'octahedron'
        """
        self.source_es = Receiver()
        if sampling_scheme == 'single':
            self.source_es.coord = np.copy(self.source.coord)
        elif sampling_scheme == 'tetrahedron':
            self.source_es.tetrahedron(radius = source_radii, center = self.source.coord[0])
        elif sampling_scheme == 'octahedron':
            self.source_es.octahedron(radius = source_radii, center = self.source.coord[0])
        elif sampling_scheme == 'cube':
            self.source_es.cube(radius = source_radii, center = self.source.coord[0])
        elif sampling_scheme == 'dodecahedron':
            self.source_es.dodecahedron(radius = source_radii, center = self.source.coord[0])
        elif sampling_scheme == 'icosahedron':
            self.source_es.icosahedron(radius = source_radii, center = self.source.coord[0])
        else:
            self.source_es.octahedron(radius = source_radii, center = self.source.coord[0])
        self.num_sources = self.source_es.coord.shape[0]
    
    def get_sens_mtx(self, k0, rm_rs, rm_rq, cosmq):
        """ Forms matrix for the free-field rigid scattering problem.
        
        Parameters
        ----------
        k0 : float
            wave-number magnitude value in [rad/m]
        rm_rs : numpyndArray
            MxNs matrix with the Euclidan norm distances from mics to original source (eq)
        rm_rq : numpyndArray
            MxL matrix with the Euclidan norm distances from mics to equiv sources (diffuser)
        cosmq : numpyndArray
            MxL matrix with the Cosine values of mics to equiv sources (diffuser)
        """
        # Terms associated with the original source
        g_fun = self.kernel_g_m(k0, rm_rs)
        # Terms associated with the equivalent sources
        dg_fun = self.kernel_g_d(k0, rm_rq, cosmq)
        g_mtx = np.hstack((g_fun, dg_fun))
        # Pre-conditioner vector
        # rm_rs_mean = np.mean(rm_rs, axis = 0)
        # rm_rq_mean = np.mean(rm_rq**2 * np.abs(cosmq), axis = 0)
        # rm_rq_mean = np.mean(rm_rq**2, axis = 0)
        rm_rs_mean = np.mean(rm_rs)
        rm_rq_mean = np.mean(rm_rq)**2
        cond_vec = np.hstack((rm_rs_mean*np.ones(rm_rs.shape[1]), 
                              rm_rq_mean*np.ones(rm_rq.shape[1])))
        # cond_vec = np.linalg.norm(g_mtx, axis = 0)
        return g_mtx, cond_vec
    
    def pk_ff_rigid_static(self, ):
        """ Regularized multi frequency solver
        
        Static solver - the equivalent sources are fixed (do not vary with freq).
        We use the max freq to derive discretization
        """
        self.problem_type = 'Free-fild rigid scattering'
        # Reg. parameter / cond num init
        self.lambd_value_vec = np.zeros(len(self.controls.k0))
        self.cond_num = np.zeros(len(self.controls.k0))
        # Get distances                    
        rm_rs = self.get_rm_rs_dist()
        rm_rq = self.get_rm_rq_dist(self.receivers)
        cosmq = self.get_cosmq_dist(self.receivers, rm_rq)
        # Monopole init
        self.num_cols = rm_rs.shape[1]+rm_rq.shape[1]
        self.pk = []        
        # Initialize bar
        bar = tqdm(total=len(self.controls.k0),
                   desc='Calc. Regularized inversion (' +\
                       self.problem_type + ')...', ascii=False)
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            g_mtx, cond_vec = self.get_sens_mtx(k0, rm_rs, rm_rq, cosmq)
            g_mtx = g_mtx @ np.diag(cond_vec)
            # Condition number
            # self.cond_num[jf] = np.linalg.cond(g_mtx)
            x, self.lambd_value_vec[jf], self.cond_num[jf] =\
                self.solve_freq(g_mtx, self.pres_s[:,jf])
            self.pk.append(np.diag(cond_vec) @ x)
            bar.update(1)
        bar.close()
        self.problem_size = g_mtx.shape
        
    def pk_ff_rigid_dynamic(self, n_s_lam = 4):
        """ Regularized multi frequency solver
        
        Dynamic solver - the equivalent sources vary with freq.
        """
        self.problem_type = 'Free-fild rigid scattering'
        # Reg. parameter / cond num init
        self.lambd_value_vec = np.zeros(len(self.controls.k0))
        self.cond_num = np.zeros(len(self.controls.k0))
        # # Get distances                    
        # rm_rs = self.get_rm_rs_dist()
        # rm_rq = self.get_rm_rq_dist(self.receivers)
        # cosmq = self.get_cosmq_dist(self.receivers, rm_rq)
        # Monopole init
        # self.num_cols = rm_rs.shape[1]+rm_rq.shape[1]
        self.pk = []        
        # Initialize bar
        bar = tqdm(total=len(self.controls.k0),
                   desc='Calc. Regularized inversion (' +\
                       self.problem_type + ')...', ascii=False)
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # Get freq discretization
            dx = self.get_dx(n_s_lam = n_s_lam, freq = self.controls.freq[jf])
            self.sample_bbox(dx = dx)
            rm_rs = self.get_rm_rs_dist()
            rm_rq = self.get_rm_rq_dist(self.receivers)
            cosmq = self.get_cosmq_dist(self.receivers, rm_rq)
            g_mtx, cond_vec = self.get_sens_mtx(k0, rm_rs, rm_rq, cosmq)
            g_mtx = g_mtx @ np.diag(cond_vec)
            # Condition number
            # self.cond_num[jf] = np.linalg.cond(g_mtx)
            x, self.lambd_value_vec[jf], self.cond_num[jf] =\
                self.solve_freq(g_mtx, self.pres_s[:,jf])
            self.pk.append(np.diag(cond_vec) @ x)
            bar.update(1)
        bar.close()
        self.problem_size = g_mtx.shape
    
    def recon_pt_static(self, receivers):
        """ Reconstruct total pressure.

        Reconstruction of scattered pressure with a receiver object. 

        Parameters
        ----------
        receivers: object
            receiver array
        """
        # Get receiver data (frequency independent)
        rm_rs = self.get_rm_rs_dist()
        rm_rq = self.get_rm_rq_dist(receivers)
        cosmq = self.get_cosmq_dist(receivers, rm_rq)
        # Initialize variables
        self.pt_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        # Initialize bar
        bar = tqdm(total=len(self.controls.k0), desc='Reconstructing total pressure...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # Forming the reconstruction matrix
            g_mtx,_ = self.get_sens_mtx(k0, rm_rs, rm_rq, cosmq)
            # pressure reconstruction
            self.pt_recon[:,jf] = g_mtx @ self.pk[jf]  # total pressure
            bar.update(1)
        bar.close()    
    
    def recon_ps_static(self, receivers, directivity = True):
        """ Reconstruct scattered pressure.

        Reconstruction of scattered pressure with a receiver object. 

        Parameters
        ----------
        receivers: object
            receiver array
        """
        # Get receiver data (frequency independent)
        # rm_rs = self.get_rm_rs_dist()
        rm_rq = self.get_rm_rq_dist(receivers)
        cosmq = self.get_cosmq_dist(receivers, rm_rq)
        # Initialize variables
        self.ps_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        # Initialize bar
        bar = tqdm(total=len(self.controls.k0), desc='Reconstructing scattered pressure...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # Forming the reconstruction matrix
            dg_fun = self.kernel_g_d(k0, rm_rq, cosmq)
            # pressure reconstruction
            self.ps_recon[:,jf] = dg_fun @ self.pk[jf][self.num_sources:]  # total pressure
            bar.update(1)
        bar.close()
        
        if directivity:
            self.dir_spk = self.ps_recon
    
    def check_decomp(self,):
        """ check decomposition quality
        """
        self.recon_pt_static(self.receivers)
        
        self.mae = np.zeros(len(self.controls.k0))
        self.error_db = np.zeros(len(self.controls.k0))
        bar = tqdm(total = len(self.controls.k0), desc = 'Computing decomposition quality...')
        # loop over frequencies
        for jf, k0 in enumerate(self.controls.k0):
            self.mae[jf] = np.mean(np.abs(self.pt_recon[:,jf] - self.pres_s[:,jf]))
            self.error_db[jf] = 20*np.log10(self.mae[jf])
            bar.update(1)
        bar.close()
        # return self.mae, self.error_db
    
    def octave_avg(self,):
        """ Octave avareging directivities
        """
        self.dir_oct, self.fc, _, _ = ut_is.third_octave_avg(self.controls.freq, self.dir_spk, 
                                                   magnitude = True)
        self.gamma = ut_is.diffusion_coef_equiang(self.fc, np.abs(self.dir_oct))
        
    def plot_reg_par(self,):
        """ Plots regularization parameter as a function of freq.
        """
        ymin,ymax = 0.9*self.lambd_value_vec.min(),  1.1*self.lambd_value_vec.max(),
        _, ax = ut_is.give_me_an_ax(figsize=(6,3))
        ut_is.plot_1d_curve(self.controls.freq, self.lambd_value_vec, ax = ax[0,0], 
                            xlims = None, ylims = (ymin, ymax), color = 'dodgerblue', 
                            linewidth = 1.5, marker = None, linestyle = '-', 
                            alpha = 1.0, label = None, xlabel = "Frequency [Hz]", 
                            ylabel = r"$\lambda$ [-]", linx = False, 
                            liny = True, xticks = None)
    