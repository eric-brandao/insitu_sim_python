# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:56:44 2022

@author: ericb
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import griddata
from controlsair import cart2sph, sph2cart
import gmsh
import meshio

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.graph_objs as go
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
                  color_method = 'dB',
                  radius_method = 'normalized'):#, axis):
    # balloon_data = pre_balloon_list(all_coords, receiver_indexes, conectivities,
    #                                 pressure, pressure_lim, arc_theta)
    
    x, y, z, color_data, color_limits = pre_balloon(coords,
         pressure, dinrange = dinrange,
         color_method = color_method,
         radius_method = radius_method)
    
    colorbar_dict = {'title': 'SPL [dB]',
                     'titlefont': {'color': 'black'},
                     'title_side': 'right',
                     'tickangle': -90,
                     'tickcolor': 'black',
                     'tickfont': {'color': 'black'},
                     'x': -0.1}
    
    fig = go.Figure()

    # for i in range(len(balloon_data["x"])):      
    #     fig.add_trace(go.Mesh3d(x=balloon_data["x"][i], y=balloon_data["y"][i], z=balloon_data["z"][i],
    #                             i=balloon_data["elements"][i].T[0, :], j=balloon_data["elements"][i].T[1, :],
    #                             k=balloon_data["elements"][i].T[2, :], intensity=balloon_data["intensity"][i],
    #                             colorscale='jet', intensitymode='vertex', showlegend=False,
    #                             visible=True, opacity=1,
    #                             showscale=True, colorbar=colorbar_dict, cmin=-18, cmax=0))
    
       
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z,
                            i=conectivities.T[0, :], j=conectivities.T[1, :],
                            k=conectivities.T[2, :], intensity=color_data,
                            colorscale='inferno', intensitymode='vertex', showlegend=False,
                            visible=True, opacity=1,
                            showscale=True, colorbar=colorbar_dict, 
                            cmin=color_limits[0], cmax=color_limits[1]))
    
    #fig = remove_bg_and_axis(fig, 1)
    #fig = set_all_cameras(fig, 1, axis=axis)

    return fig