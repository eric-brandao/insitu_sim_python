# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:56:44 2022

@author: ericb
"""
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import griddata
from controlsair import cart2sph, sph2cart
import gmsh
import meshio

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
#import plotly.figure_factory as ff
#import plotly.graph_objs as go
from plotly.subplots import make_subplots


def find_freq_index(freq_vec, freq_target = 1000):
    """Find the index of a frequency closer to a target frequency.
    
    Parameters
    ----------
        freq_vec : np1dArray
            frequecy vector of measurement or simulation.
        freq_target : float
            the disered frequency
    
    Returns
    ----------
        id_f : int
            index in freq_vec of the closest frequency to freq_target
    """
    id_f = np.where(freq_vec <= freq_target)
    id_f = id_f[0][-1]
    return id_f

def create_angle_grid(n_pts_original, grid_factor = 2, limit_phi = (-180,180),
                      limit_theta = (-90,90)):
    """ Create a regular grid of angles to interpolate on.
    
    Parameters
    ----------
        n_pts_original : int
            The number of points in the original grid
        grid_factor : int
            factor multiplying the n_pts_original (the new grid will have 
           grid_factor * n_pts_original X grid_factor * n_pts_original points)
        limit_phi : tuple of 2 (in degrees)
            tuple containing the inferior and superior values of phi (new grid)
        limit_theta : tuple of 2 (in degrees)
            tuple containing the inferior and superior values of theta (new grid)
    
    Returns
    ----------
        grid_phi : numpy ndArray
            phi grid in radians
        grid_theta : numpy ndArray
            theta grid in radians
    """
    
    # limiting angles in radians
    limit_phi_rad = np.deg2rad(limit_phi)
    limit_theta_rad = np.deg2rad(limit_theta)
    # Create a grid to interpolate on
    #nphi = int(2*(npts+1))
    #ntheta = int(npts+1)
    # Create theta and phi grid
    new_phi = np.linspace(limit_phi_rad[0], limit_phi_rad[1], grid_factor*n_pts_original)
    new_theta = np.linspace(limit_theta_rad[0], limit_theta_rad[1], grid_factor*n_pts_original)
    grid_phi, grid_theta = np.meshgrid(new_phi, new_theta)
    return grid_phi, grid_theta

def interpolate2regulargrid(sph, data_sph, grid_phi, grid_theta):
    """Interpolate data on spherical grid to a regular grid
    
    This can be problematic
    
    Parameters
    ----------
        sph : numpy ndArray
            original spherical grid: size (n_pts x 3)
        data_sph: numpy 1dArray
            original data on the spherical grid: (size n_pts)
        grid_phi : numpy ndArray
            phi grid in radians
        grid_theta : numpy ndArray
            theta grid in radians
    Returns
    ----------
        grid_data : numpy ndArray
            data on the grid
    """
    # theta phi representation of original spherical points
    _, theta, phi = cart2sph(sph[:,0], sph[:,1], sph[:,2])
    thetaphi_pts = np.transpose(np.array([phi, theta]))
    # interpolate using griddata from scipy
    grid_data = griddata(thetaphi_pts, data_sph, (grid_phi, grid_theta), 
                            method='cubic', fill_value=np.finfo(float).eps, rescale=False)
    return grid_data

def plot3D_directivity(grid_phi, grid_theta, grid_data, figsize = (7,8)):
    
    # Angular to spatial domain
    x, y, z = sph2cart(grid_data, grid_theta, grid_phi)
    
    # Create a figure
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize = figsize)
    # Plot the surface.
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('inferno'))
    surf = ax.plot_surface(x, y, z, facecolors=cmap.to_rgba(grid_data),
                           linewidth=1, antialiased=False)
# =============================================================================
#     ax.set_xlim((-1,1))
#     ax.set_ylim((-1,1))
    #ax.set_zlim((-50,0))
# =============================================================================
    #fig.colorbar(surf, shrink=0.5, aspect=5)

def plot3D_directivity_tri(phi, theta, data, figsize = (7,8)):
    
    # Angular to spatial domain
    x, y, z = sph2cart(data, theta, phi)
    tri = mtri.Triangulation(x, y)
    # Create a figure
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize = figsize)
    # Plot the surface.
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('inferno'))
    surf = ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)# linewidth = 0.2, antialiased = True)
# =============================================================================
#     ax.set_xlim((-1,1))
#     ax.set_ylim((-1,1))
    #ax.set_zlim((-50,0))
# =============================================================================
    #fig.colorbar(surf, shrink=0.5, aspect=5)


def extract_mesh_data_from_gmsh(factory):
    """
    Extracts mesh data from Gmsh object.

    Parameters
    ----------
    dim : int
        Mesh dimension.
    order : int
        Mesh order.
    mesh_data : dict
        Dictionary that the mesh data will be written into.
    """
    # gmsh.fltk.run()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)

    _, _, node_tags_surface = gmsh.model.mesh.getElements(2)
    elements = (np.array(node_tags_surface, dtype=int).reshape(-1, 3) - 1).T.astype(
        "uint32")
    _, vertices_xyz, _ = gmsh.model.mesh.getNodes()
    vertices = vertices_xyz.reshape((-1, 3)).T
    return vertices, elements

def init_gmsh(h, tol, mesh_algorithms):
    """
    Initialize Gmsh Python API with the desired parameters.

    Parameters
    ----------
    h : list
        List of floats with minimum and maximum element sizes.
    tol : float
        Boolean operation tolerance in meters.
    mesh_algorithms : list
        List of integers containing 2D and 3D mesh algorithm indexes.

    Returns
    -------
    Gmsh OpenCASCADE CAD engine.
    """
    gmsh.initialize(sys.argv)
    gmsh.model.add("mesh")
    gmsh.option.setNumber("Mesh.Algorithm", mesh_algorithms[0])
    gmsh.option.setNumber("Mesh.Algorithm3D", mesh_algorithms[1])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h[0])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h[1])
    gmsh.option.setNumber("Geometry.ToleranceBoolean", tol)
    factory = gmsh.model

    return factory

def gmsh_hemisphere(radius, h, axis):

    angle1 = 0; angle2 =np.pi / 2; angle3 = 2*np.pi
    axis_dict = {"x": [0,1,0], "y": [-1,0,0], "z": [0,1,0]}
    factory = init_gmsh([h[0]/radius, h[1]/radius], 0.001, [6, 1])
    factory.occ.add_sphere(0, 0, 0, radius, tag=5, angle1=angle1, angle2=angle2, angle3=angle3)
    factory.occ.synchronize()
    pg = gmsh.model.getPhysicalGroups(2)
    tg2 = gmsh.model.occ.getEntities(2)

    surface_loop = gmsh.model.occ.addSurfaceLoop([i[1] for i in tg2], sewing=False)
    gmsh.model.occ.add_volume([surface_loop])
    gmsh.model.occ.synchronize()
    tg3 = gmsh.model.occ.getEntities(3)

    gmsh.model.occ.rotate(tg3,0,0,0,axis_dict[axis][0],axis_dict[axis][1],axis_dict[axis][2],np.pi/2)
    gmsh.model.occ.synchronize()
    vertices, elements = extract_mesh_data_from_gmsh(factory)
    vertices = vertices.T
    elements = elements.T
    init_elem = []
    init_tags = []
    for ii in tg2:
        elem_tag = gmsh.model.mesh.getElements(2, ii[1])[1][0]
        pgones = np.ones_like(elem_tag) * ii[1]
        init_elem = np.hstack((init_elem, elem_tag))
        init_tags = np.hstack((init_tags, pgones))
    vas = np.argsort(init_elem)
    tags = np.asarray(init_tags[vas],dtype=int)
    elem_index = np.argwhere(tags == 1).ravel()
    elem = elements[elem_index]
    hemi_index = np.sort(np.unique(elem))# np.argwhere(vertices[:, axis_dict[axis]] > 0.01).ravel()
    vertices_filter = vertices[hemi_index]

    if axis == 'z':
        vertices_filter[:, [0, 2]] = vertices_filter[:, [2, 0]]
        vertices[:, [0, 2]] = vertices[:, [2, 0]]
    gmsh.finalize()
    return vertices_filter, elem, vertices, hemi_index

def pressure_normalize(pressure):
    """Normalize sound pressure
    
    Parameters
    ----------
        pressure: ndArray
            Complex pressure array
    Returns
    -------
        pressure_norm : ndArray
            Normalized pressure level    
    """
    p_abs = np.abs(pressure)
    pressure_norm = p_abs/np.amax(p_abs)
    return pressure_norm

def pressure2spl(pressure, p_ref = 2e-5):
    """Computes SPL from complex pressure
    
    Parameters
    ----------
        pressure: ndArray
            Complex pressure array
        p_ref : float
            reference value
    Returns
    -------
        spl : ndArray
            Sound pressure level with specified reference
    """
    #spl = 10 * np.log10(0.5 * pressure * np.conj(pressure) / p_ref ** 2)
    spl = 10 * np.log10((np.abs(pressure) / p_ref) ** 2)
    return np.real(spl)

def pre_balloon2(all_coords, receiver_indexes, conectivities, pressure, pressure_lim):
    pressure = np.real(pressure)
    max_pressure = np.amax(pressure)

    if pressure_lim > 0:
        pressure = pressure2spl(pressure)
        min_pressure = np.amin(pressure)
        norm_pressure = pressure - min_pressure + pressure_lim
        norm_pressure[norm_pressure<0] = 0
        plot_pressure = pressure - np.amax(pressure)
    else:
        norm_pressure = 20 * pressure / max_pressure
        plot_pressure = pressure2spl(norm_pressure)
        plot_pressure = plot_pressure - np.amax(plot_pressure)

    plot_pressure = np.abs(pressure)/np.amax(np.abs(pressure))

    all_coords = all_coords / np.amax(all_coords)
    r, theta, phi = cart2sph(all_coords[:, 0], all_coords[:, 1], all_coords[:, 2])
    # r_ = r.copy()
    r[receiver_indexes] = plot_pressure#norm_pressure #plot_pressure#
    r[np.setdiff1d(np.arange(len(r)), receiver_indexes)] = np.zeros_like(
        r[np.setdiff1d(np.arange(len(r)), receiver_indexes)])
    x, y, z = sph2cart(r, theta, phi)

    return x, y, z, conectivities, r, plot_pressure


def pre_balloon(coords, pressure, dinrange = 18, 
                color_method = 'normalized',
                radius_method = 'normalized'):
    # normalize
    pressure_norm = pressure_normalize(pressure)
    # dB
    pressure_spl = pressure2spl(pressure_norm, p_ref = 1)
    pressure_spl[pressure_spl < -dinrange] = -dinrange
    # Compute pressure return from color_method
    if color_method == 'normalized':
        color_data = pressure_norm
        color_limits = np.array([0, 1])
    elif color_method == 'dB':
        color_data = pressure_spl
        color_limits = np.array([-dinrange, 0])
    else:
        color_data = pressure_spl
        color_limits = np.array([-dinrange, 0])
    # Transform to spherical coordinates to get radius of the plot
    radius_data, theta, phi = cart2sph(coords[:, 0], coords[:, 1], coords[:, 2])
    # The radius of the plot using radius_method
    if radius_method == 'normalized':
        radius_data = pressure_norm #pressure_spl + dinrange
    elif radius_method == 'dB':
        radius_data = pressure_spl + dinrange
    elif radius_method == 'dBnormalized':
        pres_positive = pressure_spl + dinrange
        radius_data = pres_positive / np.amax(pres_positive)
    else:
        radius_data = pressure_spl + dinrange
    # Transform back to caterian coordinates
    x, y, z = sph2cart(radius_data, theta, phi)

    return x, y, z, color_data, color_limits

def pre_balloon_list(all_coords, receiver_indexes, conectivities, pressure, pressure_lim,
                     arc_theta):
    balloon_data = {"x": [], "y": [], "z": [], "elements": [], "r": [], "intensity": []}
    #for i in range(len(pressure[:, 0])):
    a, b, c, d, e, f = pre_balloon(all_coords, receiver_indexes, conectivities,
                                   pressure, pressure_lim)
    balloon_data["x"].append(a)
    balloon_data["y"].append(b)
    balloon_data["z"].append(c)
    balloon_data["elements"].append(d)
    balloon_data["r"].append(e)
    balloon_data["intensity"].append(f)

    return balloon_data


def plot_3d_polar(coords, conectivities, pressure, dinrange = 18,
                  color_method = 'dB', radius_method = 'normalized',
                  color_code = 'jet', view = 'iso_z', eye = None,
                  renderer = 'notebook', remove_axis = False):#, axis):
    
    # renderer
    pio.renderers.default = renderer
    # Balloon data
    x, y, z, color_data, color_limits = pre_balloon(coords,
         pressure, dinrange = dinrange,
         color_method = color_method,
         radius_method = radius_method)
    
    colorbar_dict = {'title': color_method,
                     'titlefont': {'color': 'black'},
                     'title_side': 'right',
                     'tickangle': -90,
                     'tickcolor': 'black',
                     'tickfont': {'color': 'black'},
                     'x': -0.1}
    
    fig = go.Figure()   
    trace = go.Mesh3d(x=x, y=y, z=z,
                            i=conectivities.T[0, :], j=conectivities.T[1, :],
                            k=conectivities.T[2, :], intensity=color_data,
                            colorscale=color_code, intensitymode='vertex', showlegend=False,
                            visible=True, opacity=1,
                            showscale=True, colorbar=colorbar_dict, 
                            cmin=color_limits[0], cmax=color_limits[1])
    fig.add_trace(trace)
    # fig = px.scatter_3d(x=x, y=y, z=z, color = color_data)
    # trace = 0
    
    eye = set_camera_eye(fig, view = view, eye_user = eye)
    camera = dict(eye=eye)

    fig.update_layout(font=dict(family="Times New Roman", size=14),
                      scene_camera = camera)
    
    if remove_axis:
        fig = remove_bg_and_axis(fig, 1)
    #fig = set_all_cameras(fig, 1, axis='z')
    return fig, trace

def plot_3d_polar2(coords, conectivities, pressure, dinrange = 18,
                  color_method = 'dB', radius_method = 'normalized',
                  color_code = 'jet', view = 'iso_z', eye = None,
                  figsize = (7,5)):
    # Balloon data
    x, y, z, color_data, color_limits = pre_balloon(coords,
         pressure, dinrange = dinrange,
         color_method = color_method,
         radius_method = radius_method)
    # set colormap
    my_cmap = plt.get_cmap(color_code)
    
    # set color limits
    if radius_method == 'dB':
        vmax = dinrange
        cbar_ticks = np.linspace(0, vmax, 4)
        cbar_ticks_label = cbar_ticks-dinrange 
    else:
        vmax = 1
        cbar_ticks = np.linspace(0, vmax, 4)
        cbar_ticks_label = cbar_ticks
        
    # Figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, projection='3d')
    trisurf = ax.plot_trisurf(x, y, z, triangles=conectivities,
                    cmap = my_cmap, antialiased = False, shade = False,
                    linewidth = 0, edgecolor='Gray', vmin = 0, vmax = vmax)
    
    
    cbar = fig.colorbar(trisurf, shrink=0.7, aspect=50, 
                 ticks = cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticks_label)
    ax.set_xlim(-vmax/2, vmax/2)
    ax.set_ylim(-vmax/2, vmax/2)
    ax.set_zlim(0, vmax)
    # View
    if eye != None:
        r_view, theta_view, phi_view = cart2sph(eye['x'],eye['y'],eye['z'])
        ax.view_init(np.rad2deg(theta_view), np.rad2deg(phi_view))
    
    return fig, ax
    

def remove_bg_and_axis(fig, len_scene):
    for ic in range(len_scene):
        scene_text = f'scene{ic + 1}' if ic > 0 else 'scene'
        fig.layout[scene_text]['xaxis']['showbackground'] = False
        fig.layout[scene_text]['xaxis']['visible'] = False
        fig.layout[scene_text]['yaxis']['showbackground'] = False
        fig.layout[scene_text]['yaxis']['visible'] = False
        fig.layout[scene_text]['zaxis']['showbackground'] = False
        fig.layout[scene_text]['zaxis']['visible'] = False
    return fig

def set_camera_eye(fig, view = 'iso_z', eye_user = None):
    """Set a view
    """
    if view == 'iso_z':
        eye = dict(x=1.2, y=-1.1, z=0.5)
    elif view == 'z':
        eye = dict(x=0, y=0, z=2)
    elif view == 'x':
        eye = dict(x=0, y=-2, z=1)
    elif view == 'y':
        eye = dict(x=-2, y=0, z=1)
    else:
        eye = dict(x=1.2, y=-1.1, z=0.5)
    if eye != None:
        eye = eye_user
    return eye

def set_all_cameras(fig, len_scene, camera_dict=None, axis='z'):
    eye_dict = {'x': [0., 0., -1.75],
                'y': [0., 0., -1.75],
                'z': [-0.5, -1.5, 0],
                "iso_z": [-1.2, -1.1, 0.4]}

    up_dict = {'x': [1, 0., 0.],
               'y': [0, 1, 0.],
               'z': [0., 0., 1],
               'iso_z': [0., 0., 1]}

    if camera_dict is None:
        camera_dict = dict(eye=dict(x=eye_dict[axis][0], y=eye_dict[axis][1], z=eye_dict[axis][2]),
                           up=dict(x=up_dict[axis][0], y=up_dict[axis][1], z=up_dict[axis][2]),
                           center=dict(x=0, y=0, z=0), projection_type="perspective")

    for ic in range(len_scene):
        scene_text = f'scene{ic + 1}_camera' if ic > 0 else 'scene_camera'
        fig.layout[scene_text] = camera_dict
    return fig

# Diffusion coefficients

def diffusion_coef_equiang(frequency, ps_abs):
    """ computes diffusion coefficient for a hemisphere sampled with equi solid angles (no area correction need)
    
    Inputs:
    ------------------
    frequency : numpy1dArray
        frequency vector
    ps_abs : numpyndArray
        absolute of scattered pressure with shape Nrecs x Nfreqs
    """
    d_coef = np.zeros(len(frequency))
    num_of_recs = ps_abs.shape[0]
    
    for ic in range(len(frequency)):     
        d_coef[ic] = (np.sum((ps_abs[:,ic])**2)**2 - np.sum(((ps_abs[:,ic])**2) ** 2)) /\
            ((num_of_recs - 1) * np.sum(((ps_abs[:,ic])**2) ** 2))        
    d_coef[d_coef < 0] = 0  
    return d_coef

def area_factors(receivers):
    # Cartesian to spherical coords
    r, theta, phi = cart2sph(receivers[:, 0], receivers[:, 1], receivers[:, 2])
    theta_deg = 90-np.round(np.rad2deg(theta), decimals=0)
    phi_deg = np.round(np.rad2deg(phi), decimals=0)
    
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    # dtheta
    theta_shifted = np.roll(theta, 1)
    d_theta_vec = np.abs(theta_shifted[1:]-theta[1:])
    
    d_theta = np.amax(d_theta_vec)
    #dphi
    d_phi = np.abs(phi[-1] - phi[-2])
    
    #theta = np.pi/2-theta
    # Me
    n_imag = np.cos(theta)/np.cos(np.deg2rad(85)) - 1
    
    # ISO
    ai = np.zeros(len(theta))
    ai = 2*np.sin(theta)*np.sin(d_theta/2)
    ai[theta_deg == 0] = (4*np.pi/d_phi)*(np.sin(d_theta/4))**2
    ai[theta_deg == 90] = np.sin(d_theta/2)
    n_i_areas = ai / np.amin(ai)

    return n_imag, n_i_areas

def diffusion_coef_gauss(frequency, ps_abs, receivers):
    """ computes diffusion coefficient for a hemisphere sampled with gaussian grid (area correction need)
    
    Inputs:
    ------------------
    frequency : numpy1dArray
        frequency vector
    ps_abs : numpyndArray
        absolute of scattered pressure with shape Nrecs x Nfreqs
    """
    _, ni = area_factors(receivers)
    d_coef = np.zeros(len(frequency))
    num_of_recs = ps_abs.shape[0]
    
    for ic in range(len(frequency)):     
        d_coef[ic] = (np.sum(ni*(ps_abs[:,ic])**2)**2 - np.sum(ni*((ps_abs[:,ic])**2) ** 2)) /\
            ((np.sum(ni)-1) * np.sum(ni*((ps_abs[:,ic])**2) ** 2))        
    d_coef[d_coef < 0] = 0  
    return d_coef

def diffusion_coef_norm(d_coef_sample, d_coef_ref):
    """ Computes normalized diffusion coefficient
    """
    norm_gamma = (d_coef_sample - d_coef_ref)/(1-d_coef_ref)
    return norm_gamma

def save(obj, filename = 'fname', path = ''):
    """ To save the decomposition object as pickle

    Parameters
    ----------
    filename : str
        name of the file
    pathname : str
        path of folder to save the file
    """
    filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
    path_filename = path + filename + '.pkl'
    f = open(path_filename, 'wb')
    pickle.dump(obj.__dict__, f, 2)
    f.close()

def load(obj, filename = 'fname', path = ''):
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
    obj.__dict__.update(tmp_dict)