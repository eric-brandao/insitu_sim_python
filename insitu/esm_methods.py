# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import plotly
from tqdm import tqdm
import scipy

import lcurve_functions as lc
import utils_insitu as ut_is
from receivers import Receiver

class ESM(object):
    """ Decomposition of the sound field using the Equivalent Source Method.
    
    Several types will be available for radiation and scattering problems 
    (rigid and non-rigid scatterers)
    """
    def __init__(self, p_mtx=None, air = None, controls=None, receivers=None, source = None, 
                 regu_par = 'L-curve', bbox_size = [0.6, 0.6, 0.5], rec_bbox = True,
                 geometry = None):
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
        self.pres_s = p_mtx
        self.air = air
        self.controls = controls
        self.receivers = receivers
        self.source = source
        self.source_es = None
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
        
        # bounding box and geometry
        self.bbox_size = bbox_size
        
        if rec_bbox:
            self.get_rec_bbox(bbox_size = self.bbox_size)
        else:
            self.get_cyl_bbox(bbox_size = self.bbox_size)
            
        self.geometry = geometry        
        if self.geometry is not None:
            if rec_bbox:
                self.bbox_size = self.get_rec_bbox_size()
                self.get_rec_bbox(bbox_size = self.bbox_size)
            else:
                self.bbox_size = self.get_cyl_bbox_size()
                self.get_cyl_bbox(bbox_size = self.bbox_size)
                
    def get_min_dx_from_maxfreq(self, n_s_lam = 4, freq = 1000):
        """ computes minimum samplind distace based on max frequency and n_s_lam
        
        Parameters
        ----------
        n_s_lam : int
            number of elements per wavelength
        freq : float
            frequency value
        """
        lambda_min = self.air.c0/freq
        dx = lambda_min/n_s_lam
        return dx
            
    def get_rec_bbox_size(self):
        """ Get rectangular bounding box size with the geometry information
        """
        Lx = np.amax(self.geometry['coords'][:,0]) - np.amin(self.geometry['coords'][:,0])
        Ly = np.amax(self.geometry['coords'][:,1]) - np.amin(self.geometry['coords'][:,1])
        Lz = np.amax(self.geometry['coords'][:,2]) - np.amin(self.geometry['coords'][:,2])
        return np.array([Lx, Ly, Lz])
    
    def get_cyl_bbox_size(self):
        """ Get cylindrical bounding box size with the geometry information
        """
        radii = np.amax(self.geometry['coords'][:,0])
        Lz = np.abs(np.amax(self.geometry['coords'][:,2]) - np.amin(self.geometry['coords'][:,2]))
        return np.array([radii, Lz])
    
    def get_rec_bbox(self, bbox_size = [0.6, 0.6, 0.5]):
        """ Get the bounding box - given its size in x, y, z.
        
        
        Parameters
        ----------
        bbox_size : numpy 1dArray
            Box size in x, y, z dimensions.
        """
        Lx = bbox_size[0]
        Ly = bbox_size[1]
        Lz = bbox_size[2]
        self.bbox_verts = np.array([
            [-Lx/2, -Ly/2, 0], 
            [-Lx/2, Ly/2, 0],
            [Lx/2, Ly/2, 0],
            [Lx/2, -Ly/2, 0],
            [-Lx/2, -Ly/2, -Lz], 
            [-Lx/2, Ly/2, -Lz],
            [Lx/2, Ly/2, -Lz],
            [Lx/2, -Ly/2, -Lz]])
        
        self.bbox_faces = np.array([
            [0, 3, 2, 1], # top
            [4, 7, 3, 0], # x side 1
            [5, 1, 2, 6], # x side 2
            [4, 0, 1, 5], # y side 1
            [7, 6, 2, 3], # y side 2
            [4, 5, 6, 7], # bottom
            ])
        
        self.bbox_normals = np.array([
            [0, 0, 1], # top
            [0, -1, 0], # x side 1
            [0, 1, 0], # x side 2
            [-1, 0, 0], # y side 1
            [1, 0, 0], # y side 2
            [0, 0, -1], # bottom
            ])

        self.bbox_edges = np.array([
            [0,3],[3,2],[2,1],[1,0],
            [4,5],[5,6],[6,7],[7,4],
            [4,0],[5,1],[6,2], [7,3]])
        
    def get_cyl_bbox(self, bbox_size = [0.6, 0.6, 0.5]):
        """ Get the bounding box - given its size in x, y, z.
        
        Used for visualization purposes (sampling will be done by another method)
        
        Parameters
        ----------
        bbox_size : numpy 1dArray
            Box size in x, y, z dimensions.
        """
        self.radii = bbox_size[0]
        self.Lz = bbox_size[1]
        # Sampling Phi
        n_phi = 72
        phi = np.linspace(0, 2*np.pi, n_phi)
        
        # Top cilinder vertices (across circunference)
        self.bbox_verts = np.array([self.radii*np.cos(phi),
                                    self.radii*np.sin(phi),
                                    np.zeros(n_phi)]).T
        # Bottom cilinder vertices (across circunference)
        self.bbox_verts = np.vstack((self.bbox_verts,
                                     np.array([self.radii*np.cos(phi),
                                               self.radii*np.sin(phi),
                                               np.zeros(n_phi)-self.Lz]).T))
        self.bbox_verts = np.vstack((self.bbox_verts, np.array([0.0,0.0,0.0]),
                                     np.array([0.0,0.0,-self.Lz])))
        # Bounding box edges
        bbox_edges = []
        for je in range(n_phi-1):
            bbox_edges.append(np.array([je, je+1])) # top
            bbox_edges.append(np.array([je+n_phi, je+n_phi+1])) # bottom
            bbox_edges.append(np.array([je, je+n_phi])) # sides         
        self.bbox_edges = np.array(bbox_edges)
        
        # Bounding box faces
        bbox_faces = []
        for je in range(n_phi-1):
            bbox_faces.append(np.array([je, je+1, 2*n_phi])) # top
            bbox_faces.append(np.array([je+n_phi+1, je+n_phi,  2*n_phi+1])) # top
            bbox_faces.append(np.array([je, je+n_phi, je+1])) # sides
            bbox_faces.append(np.array([je+n_phi, je+n_phi+1, je+1])) # sides
        self.bbox_faces = np.array(bbox_faces)
    
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
        return np.unique(eq_sources, axis=0)
    
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
        n_phi = int(np.ceil(2*np.pi*self.radii/dx))
        n_Lz = int(np.ceil(self.Lz/dx))
        phi_vec = np.linspace(0, 2*np.pi, n_phi)
        # Sample side
        eq_sources_side = np.zeros((n_phi*n_Lz, 3))
        for jp, phi in enumerate(phi_vec): 
            x = self.radii*np.cos(phi) * np.ones(n_Lz)
            y = self.radii*np.sin(phi) * np.ones(n_Lz)
            z = np.linspace(-self.Lz + 0.5*dx, -0.5*dx, n_Lz)      
            eq_sources_side[jp*n_Lz:(jp+1)*n_Lz,:] = np.array([x, y, z]).T
            
        # sample top
        max_min_dist = 2*dx
        n_sf = int(0.9*n_phi)
        sf = Receiver()
        while max_min_dist > dx:
            sf.sunflower_circular_array(n_recs = n_sf, radius = self.radii, 
                                        alpha = 2, zr = 0.0)
            sf.rotate_array(axis = 'z', theta_deg = 22.5)
            max_min_dist, _, _ = sf.compute_min_distances(sf.coord)
            n_sf += while_step
        eq_sources = np.vstack((eq_sources_side, sf.coord))
        # Sample bottom
        sf.translate_array(axis = 'z', delta = -self.Lz)
        eq_sources = np.vstack((eq_sources, sf.coord))
        # rounding + unique 
        eq_sources = np.round(eq_sources / tol) * tol
        return np.unique(eq_sources, axis=0)
        
    def sample_source(self, source_radii = 0.1, sampling_scheme = 'octahedron'):
        """ sample equivalent sources arond the sound source
        
        Parameters
        ----------
        source_radii : float
            source radius
         sampling_scheme : str
             type of samplig scheme. It can be 'tetrahedron, 'octahedron', 'cube',
             'dodecahedron', 'icosahedron'. Default is 'octahedron'
        """
        self.source_es = Receiver()
        if sampling_scheme == 'tetrahedron':
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
    
    def plotly_scene(self, renderer = 'browser', xyz_size = [1,1,1],
                     plot_bbox = True, eq_sources = None):
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
            
        if self.source_es is not None:
            ses_trace = self.get_souce_es_trace()
            fig.add_traces(ses_trace)
            
        # Receivers
        if self.receivers is not None:
            rec_trace = self.get_receivers_trace()
            fig.add_traces(rec_trace)
            x = np.amax(np.array([np.amax(self.receivers.coord[:,0]), xyz_size[0]]))
            y = np.amax(np.array([np.amax(self.receivers.coord[:,1]), xyz_size[1]]))
            xmax, ymax = np.sqrt(x**2 + y**2), np.sqrt(x**2 + y**2)
          
        # Equivalent sources
        if eq_sources is not None:
            eqs_trace =  self.get_es_trace(eq_sources)
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
        # face_trace = plotly.graph_objs.Surface(
        #     x=self.bbox_verts[:,0], y=self.bbox_verts[:,1], z=self.bbox_verts[:,2],
        #     opacity=0.12, surfacecolor = 'aquamarine',
        #     showscale=False, name='Bbox-F')
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
    
    def get_es_trace(self, eq_sources):
        """ Bbox plotly equivalent sources trace
        """
        eqs_trace =  plotly.graph_objs.Scatter3d(
            x = eq_sources[:,0], y = eq_sources[:,1], z = eq_sources[:,2], 
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
        
        