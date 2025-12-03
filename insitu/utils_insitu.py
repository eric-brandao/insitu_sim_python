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
import matplotlib as mpl
import scipy
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

from PIL import Image, ImageChops
import io

import PyPDF2
import pdf2image 
# from PyPDF2 import PdfReader, PdfWriter

# from receivers import Receiver
import receivers

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

def get_color_par_dBnorm(data, dinrange = 10):
    """ Get color values to plot a colormap
    
    Parameters
    ----------
    data : 1dArray
        Pressure or other type of data that you want to create a color data from.
    dinrange : float
        Dinamic range to adjust your color data to.
    """
    data = pressure_normalize(data)
    color_par = pressure2spl(data, p_ref = 1)
    color_par[color_par<-dinrange] = -dinrange
    return color_par

def third_octave_freqs(freq):
    """ Returns a list of center frequencies, lower and upper bonds in 1/3 oct band
    
    The list will be limitted to your frequency vector
    
    Parameters
    ----------
    freq : numpy 1dArray
        frequency vector in discrete band
    """
     
    # Center frequencies
    # fcentre  = (10**nband) * (2**(np.arange(-18, 13) / nband))
    fcentre = np.array([16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
                        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                        4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000])
    # Lower and upper frequencies
    fd = 2**(1/(2 * 3))
    fupper = np.round(fcentre * fd, 1)
    flower = np.round(fcentre / fd, 1)
    
    # find indexes of lower and upper bands in your freq vector
    id_lower = np.where(fcentre <= np.amin(freq))[0][-1] + 1
    id_upper = np.where(fcentre <= np.amax(freq))[0][-1]# - 1
    
    return fcentre[id_lower:id_upper], flower[id_lower:id_upper], fupper[id_lower:id_upper]

def third_octave_avg(freq, data, magnitude = True):
    """ Returns a list of center frequencies, lower and upper bonds in 1/3 oct band
    
    The list will be limitted to your frequency vector
    
    Parameters
    ----------
    freq : numpy 1dArray
        frequency vector in discrete band
    magnitude : bool
        Wether to use magnitude (True) or complex (False) averaging
    """
    
    # Get 1/3 octave frequencies
    fcentre, flower, fupper = third_octave_freqs(freq)
    # Choose transformation
    if magnitude:
        data = np.abs(data)
        data_oct = np.zeros((data.shape[0], len(fcentre)))
    else:
        data_oct = np.zeros((data.shape[0], len(fcentre)), dtype = complex)
    
    # Loop over each freq band and average
    for jfc, fc in enumerate(fcentre):
        # find indexes belonging to the frequency band in question
        id_f = np.where(np.logical_and(freq >= flower[jfc], freq <= fupper[jfc]))
        #print('f_c  = {} Hz, f_low  = {} Hz, f_up  = {} Hz'.format(fc, flower[jfc], fupper[jfc]))
        data_oct[:, jfc] = np.mean(data[:, id_f[0]], axis = 1)
    return data_oct, fcentre, flower, fupper

def plotly_mesh(coordinates, connectivities, renderer = 'notebook', edges_color = 'black',
                mesh_color = ["rgb(222, 184, 135)"]):
    """ Plot a mesh using plotly
    
    Parameters
    ----------
    coordinates : numpy ndArray
        coordinates of the points of the mesh (Ncoords x 3)
    connectivities : numpy ndArray
        triagle connectivities of the mesh (Ntri x 3)
    renderer : str
        plotly renderer: 'notebook', 'browser'
    """
    # renderer
    if renderer != 'notebook' and renderer != 'browser':
        print("invalid Plotly renderer. Changing it to browser")
        renderer = 'browser'
    pio.renderers.default = renderer
    # Figure isntantiation
    fig = go.Figure()
    # Mesh 
    fig = ff.create_trisurf(x = coordinates[:, 0], y = coordinates[:, 1], z = coordinates[:, 2],
                            simplices = connectivities, show_colorbar = False, 
                            color_func = connectivities.shape[0] * mesh_color, 
                            edges_color = edges_color, title = '')
    return fig

def plot_map_countourf_ax(ax, x, z, color_par, dinrange, cmap = 'plasma'):
    """ Plot a 2D color map
    
    Parameters
    ----------
    ax : matplotlib axes
        axes of matplotlib figure
    x : numpy 1dArray
        x coordinates to be plotted
    z : numpy 1dArray
        z coordinates to be plotted
    color_par : numpy 1dArray
        Color scale of scalar data
    dinrange : float
        Value of dinamic range   
    cmap : str
        Color map string: ex: 'plasma', 'jet', inferno
    """
    triang = mtri.Triangulation(x, z)
    p = ax.tricontourf(triang, color_par, np.arange(-dinrange, 0.1, 0.1),
        vmin = -dinrange, vmax = 0, cmap = cmap)
    for c in p.collections:
        c.set_edgecolor("face")
    return p

def plot_map_scatter_ax(ax, coords, pressure, freq, title = ''):
    """ 3D scatter plot with color
    """
    pres2plot = pressure2spl(pressure, p_ref = 1)
    ax.set_proj_type("persp")
    cb = ax.scatter3D(coords[:,0], coords[:,1], coords[:,2], c = pres2plot)
    cbar = ax.figure.colorbar(cb, ax=ax, shrink=0.6, aspect=15, pad=0.2)
    cbar.set_label(r'$|p_t|$')
    #ax.set_title(r"{} Hz".format(freq))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

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
    
    # fig = polar_axis_3d(fig, dinrange, 'z')
    if remove_axis:
        fig = remove_bg_and_axis(fig, 1)
    fig = add_spherical_grid(fig, radius=np.abs(dinrange), delta_radius = 15, delta_azimuth= 30, 
                             delta_elevation = 30)
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
                    cmap = my_cmap, antialiased = True, shade = True,
                    linewidth = 0, edgecolor='Grey', vmin = 0, vmax = vmax)
    
    # trisurf = ax.tricontourf(x, y, z, triangles=conectivities,
    #                 cmap = my_cmap, levels = int(dinrange))
    
    # fig, ax = plot_3d_mesh(x, y, z, conectivities, cmap='jet')
    
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
    plt.tight_layout()
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

def add_spherical_grid(fig, radius=45, delta_radius = 15, delta_azimuth=90, delta_elevation=30,
                        r_annotation_offset = 5, az_color = 'black', el_color = 'black'):
    """
    Adds a spherical grid (elevation and azimuth lines) to a 3D Plotly figure.

    Parameters:
        fig: Plotly figure object to overlay the grid on.
        radius: float
            Radius of the spherical grid.
        delta_radius: float
            The spacing between circular grids in elevation and azimuth
        delta_azimuth: float
            The spacing between azimuth angles
        delta_elevation: float
            The spacing between elevation angles
        r_annotation_offset : float
            an offset applied to the annotations in the grid to make it readable
        az_color : str
            Color of the azimuth grid
        el_color : str
             Color of the elevation grid
    """
    
    # Add azimuth circle on z=0
    azimuths_circunference = np.linspace(-np.pi, np.pi, 100)
    # Add circles with different levels
    all_radius = np.linspace(delta_radius, radius, int(radius/delta_radius))#np.arange(delta_radius, radius+1, delta_radius)
    for radii in all_radius:
        x = radii * np.cos(azimuths_circunference) 
        y = radii * np.sin(azimuths_circunference) 
        z = np.zeros(len(azimuths_circunference)) 
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', 
                                   line=dict(color=az_color, width=1, dash = 'dash'), showlegend=False))

        # Add annotation to radial line
        r, theta, phi = cart2sph(x[-1], y[-1], z[-1])
        # r += r_annot_offset
        phi -= np.rad2deg(-130)
        x, y, z = sph2cart(r, theta, phi)        
        fig.add_trace(go.Scatter3d(x = [x], y = [y], z = [z],
                                    mode='text', text=[str(np.round(radii-radius,1)) + ' dB'], textfont_color=az_color,
                                    textposition='middle center', showlegend=False))
    
    
    # Azimuth radius
    azimuths_radius = np.arange(-180, 180, delta_azimuth)#np.linspace(0, 2*np.pi, 10)
    for jaz, az in enumerate(azimuths_radius):
        dummy_rec = receivers.Receiver()
        dummy_rec.line_array(line_len = radius, step = radius/100, axis = 'x', 
                        start_at = 0, zr = 0)
        dummy_rec.rotate_array(axis = 'z', theta_deg = az)
        # Add radial line
        fig.add_trace(go.Scatter3d(x = dummy_rec.coord[:,0], y = dummy_rec.coord[:,1], 
                                    z = dummy_rec.coord[:,2],
                                    mode='lines', line=dict(color=az_color, width=1, dash = 'longdash'), 
                                    showlegend=False))
        
        
        
        # Add annotation to radial line
        r, theta, phi = cart2sph(dummy_rec.coord[-1,0], dummy_rec.coord[-1,1], dummy_rec.coord[-1,2])
        r += r_annotation_offset
        x, y, z = sph2cart(r, theta, phi)        
        fig.add_trace(go.Scatter3d(x = [x], y = [y], z = [z],
                                    mode='text', text=[str(az) + 'º'], textfont_color=az_color,
                                    textposition='middle center', showlegend=False))
        

        #Add elevation lines
        azimuths_el = np.arange(0, 360, 360)
        elevation_angles = np.linspace(0, np.pi / 2, 100)
        for radii in all_radius:
            x = radii * np.cos(elevation_angles) * np.cos(np.deg2rad(azimuths_el))
            y = radii * np.cos(elevation_angles) * np.sin(np.deg2rad(azimuths_el))
            z = radii * np.sin(elevation_angles)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', 
                                        line=dict(color = el_color, width=1, dash = 'longdash'), 
                                        showlegend=False))
        # Elevation radius
        elevation_radius = np.arange(0, 90, delta_elevation)#np.linspace(0, 2*np.pi, 10)
        for jel, el in enumerate(elevation_radius):
            dummy_rec = receivers.Receiver()
            dummy_rec.line_array(line_len = radii, step = radius/100, axis = 'z', 
                            start_at = 0, zr = 0)
            dummy_rec.rotate_array(axis = 'y', theta_deg = -el)
            # Add radial line
            fig.add_trace(go.Scatter3d(x = dummy_rec.coord[:,0], y = dummy_rec.coord[:,1], 
                                        z = dummy_rec.coord[:,2],
                                        mode='lines', line=dict(color=el_color, width=1, dash = 'longdash'), 
                                        showlegend=False))
            
                
            # Add annotation to radial line
            r, theta, phi = cart2sph(dummy_rec.coord[-1,0], dummy_rec.coord[-1,1], dummy_rec.coord[-1,2])
            r += r_annotation_offset
            x, y, z = sph2cart(r, theta, phi)        
            fig.add_trace(go.Scatter3d(x = [x], y = [y], z = [z],
                                        mode='text', text=[str(el) + 'º'], textfont_color=el_color,
                                        textposition='middle center', showlegend=False))
        
        

    return fig

def add_azimuthal_grid(fig, radius=45, delta_radius = 15, delta_azimuth=90, az_color = 'black'):
    # Define a well discretized grid to plot the azimuth circunferences
    azimuths_circunference = np.linspace(-np.pi, np.pi, 100)
    # Add azimuth circles with different radius indicating the level value
    all_radius = np.linspace(delta_radius, radius, int(radius/delta_radius))
    for radii in all_radius:
        # set of x, y, z
        x = radii * np.cos(azimuths_circunference) 
        y = radii * np.sin(azimuths_circunference) 
        z = np.zeros(len(azimuths_circunference))
        # Add each circunference
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', 
                                   line=dict(color=az_color, width=1, dash = 'dash'), showlegend=False))
        # get the annotations for the radial line (the levels)
        x_annot, y_annot, z_annot = get_annotation(x[-1], y[-1], z[-1],
                                                   r_annotation_offset = 0, phi_annotation_loc = 130)
        # Add the annotations to the figure
        fig.add_trace(go.Scatter3d(x = [x_annot], y = [y_annot], z = [z_annot],
                                    mode='text', text=[str(np.round(radii-radius, 1)) + ' dB'], 
                                    textfont_color=az_color, textposition='middle center', showlegend=False))
        
    # Azimuth radial lines
    azimuths_radius = np.arange(-180, 180, delta_azimuth)
    for jaz, az in enumerate(azimuths_radius):
        # Create a dummy receiver line array spanning x-axis (to be rotated, so you create several lines)
        dummy_rec = receivers.Receiver()
        dummy_rec.line_array(line_len = radius, step = radius/100, axis = 'x', 
                        start_at = 0, zr = 0)
        dummy_rec.rotate_array(axis = 'z', theta_deg = az)
        # Add radial line to the figure
        fig.add_trace(go.Scatter3d(x = dummy_rec.coord[:,0], y = dummy_rec.coord[:,1], 
                                    z = dummy_rec.coord[:,2],
                                    mode='lines', line=dict(color=az_color, width=1, dash = 'longdash'), 
                                    showlegend=False))
        # get the annotations for each azimuthal angle (azimuth)
        x_annot, y_annot, z_annot = get_annotation(dummy_rec.coord[-1,0], dummy_rec.coord[-1,1], 
                                                   dummy_rec.coord[-1,2],
                                                   r_annotation_offset = 0, phi_annotation_loc = 130)
        # Add the annotations to the figure
        fig.add_trace(go.Scatter3d(x = [x_annot], y = [y_annot], z = [z_annot],
                                    mode='text', text=[str(np.round(radii-radius, 1)) + ' dB'], 
                                    textfont_color=az_color, textposition='middle center', showlegend=False))

def get_annotation(x, y, z, r_annotation_offset = 0, phi_annotation_loc = None):
    # Add annotation to radial line
    r, theta, phi = cart2sph(x, y, z)
    r += r_annotation_offset
    if phi_annotation_loc is not None:
        phi += phi_annotation_loc
    x_annot, y_annot, z_annot = sph2cart(r, theta, phi)
    return x_annot, y_annot, z_annot       
    

def set_camera_eye(fig, view = 'iso_z', eye_user = None):
    """Set a view
    """
    d_iso_val = 2/np.sqrt(3)
    if view == 'iso_z' and eye_user is None:
        eye = dict(x=d_iso_val, y=d_iso_val, z=d_iso_val)
    elif view == 'x' and eye_user is None:
        eye = dict(x=0, y=-2, z=0)
    elif view == 'y' and eye_user is None:
        eye = dict(x=-2, y=0, z=0)
    elif view == 'z' and eye_user is None:
        eye = dict(x=0, y=0, z=2)    
    elif view != 'iso_z' or view != 'x' or view != 'y' or view != 'z' and eye != None:
        eye = eye_user
    else: 
        eye = dict(x=d_iso_val, y=d_iso_val, z=d_iso_val)
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

def scattering_coef_angdep(frequency, p_sample, p_ref):
    """ Computes angle dependent scattering coefficient
    """
    n_freq = len(frequency)
    scatt_coeff = np.zeros(n_freq)
    for jf in range(n_freq):
        sum_mod2_sample = np.sum(np.abs(p_sample[:, jf])**2)
        sum_mod2_ref = np.sum(np.abs(p_ref[:, jf])**2)
        sum_corr = np.abs(np.sum(p_sample[:, jf]*np.conj(p_ref[:, jf])))**2
        scatt_coeff[jf] = 1-sum_corr/(sum_mod2_sample * sum_mod2_ref)
    return scatt_coeff

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
    
class Plot3Ddirectivity(object):
    """ Class to plot the 3D directivity
    """

    def __init__(self, pressure, coords = None, connectivities = None, dinrange = 18,
                      color_method = 'dB', radius_method = 'dB',
                      color_map = 'jet', view = 'iso_z', eye_dict = None,
                      renderer = 'notebook', remove_cart_axis = False,
                      create_sph_axis = True, r_annotation_offset = 5, 
                      azimuth_grid_color = 'black', elevation_grid_color = 'black',
                      num_of_radius = 3, delta_azimuth = 45, delta_elevation = 30,
                      line_style = 'dot', plot_elevation_grid = True,
                      font_family = "Palatino Linotype", font_size = 14,
                      colorbar_title = '', colorbar_shrink =  0.7, colorbar_thickness = 20,
                      fig_title = '', fig_size = 800):
        
        """

        Parameters
        ----------
        pressure : 1dArray
            Pressure vector containing the pressure in space for a single frequency
        coords : numpy ndArray
            The spherical receiver coordinates
        connectivities : numpy ndArray
            The spherical receiver connectivities
        dinrange : flat
            The dinamic range in your plot (for both color and radius)
        color_method : str
            Color scale can be either in 'dB' or 'normalized'
        radius_method : str
            Radial scale can be either in 'dB' or 'normalized' or 'dBnormalized' 
        color_map : str
            Type of your color map used. Default is 'jet'
        view : str
            Standardized view of your directivity plot. Default is 'iso_z'
        eye_dict : dict
            Dictionary containing the new view. Overwrites view varibale. Default is None
        renderer : str
            Rederer used by plotly. Can be 'notebook', 'browser', 'colab'
        remove_cart_axis : bool
            Remove or not the cartesian axis
        create_sph_axis : bool
            Create or not the spherical axis
        azimuth_grid_color : str
            Color of the azimuth grid
        elevation_grid_color : str
            Color of the elevation grid
        num_of_radius : int 
            number of radial lines on azimuth grid
        delta_azimuth : float 
            azimuth radial lines separation
        delta_elevation : float 
            elevation radial lines separation
        line_style : str
            Line style of polar grid
        plot_elevation_grid : bool
            Whether to plot the elevation grid or not
        font_family : str
            Choose your font family
        font_size : int
            choose font size
        color_bar_title : str
            Title of the color bar
        fig_size : int
            Choose your figure size (square for single directivity plot)
        """
        self.pressure = pressure
        self.coords = coords
        self.connectivities = connectivities
        self.dinrange = dinrange
        self.color_method = color_method
        self.radius_method = radius_method
        self.color_map = color_map
        self.view = view
        self.eye_dict = eye_dict
        self.renderer = renderer
        self.remove_cart_axis = remove_cart_axis
        self.create_sph_axis = create_sph_axis
        # For the spherical grid
        # self.r_annotation_offset = r_annotation_offset
        self.azimuth_grid_color = azimuth_grid_color 
        self.elevation_grid_color = elevation_grid_color
        
        self.num_of_radius = num_of_radius
        self.delta_azimuth = delta_azimuth
        self.delta_elevation = delta_elevation
        self.line_style = line_style
        self.plot_elevation_grid = plot_elevation_grid
        
        self.font_family = font_family
        self.font_size = font_size
        self.colorbar_title = colorbar_title
        self.colorbar_shrink = colorbar_shrink
        self.colorbar_thickness = colorbar_thickness
        self.fig_title = fig_title
        self.fig_size = fig_size
        
        # self.plot_3d_polar(font_family = font_family, font_size = font_size, 
        #                    color_bar_title = color_bar_title)
        
    
    def plot_3d_polar(self, use_matplotlib = False):
        """ Plots the 3D directivity
        
        By default the method uses plotly, but we can parse the plotly figure object
        to Matplolib (as a static image). This is done to achieve a nice croping of
        the figure (something you should not try to do in plotly, unless you want to
        spend painful and frustrating hours).
        
        Parameters
        ----------
        use_matplotlib : bool
            Whether to use matplotlib or not. If True, the plotly colorbar will
            be disabled.
        
        """
        # basic setup
        if use_matplotlib:
            showscale = False
        else:
            showscale = True
        # renderer
        pio.renderers.default = self.renderer
        # Balloon data
        x, y, z, color_data, color_limits = self.pre_balloon()
        # colorbar dictionary
        colorbar_dict = {'title': self.colorbar_title,
                         'len': self.colorbar_shrink, # want control
                         'thickness' : self.colorbar_thickness, # want control
                         'xanchor' : 'right',
                         'yanchor' : 'middle',
                         'titlefont': {'color': 'black'}, # want control
                         'title_side': 'right',
                         'tickangle': -90, 
                         'tickcolor': 'black',
                         'tickfont': {'color': 'black'}, # want control
                         'x': 0, 'y': 0.5}

        # Instatiate the figure and plot the mesh
        self.fig = go.Figure()
        self.trace = go.Mesh3d(x=x, y=y, z=z,
                          i = self.connectivities.T[0, :], 
                          j = self.connectivities.T[1, :],
                          k = self.connectivities.T[2, :], 
                          intensity = color_data, colorscale = self.color_map, 
                          intensitymode='vertex', showlegend = False,
                          visible = True, opacity = 1, showscale = showscale, 
                          colorbar=colorbar_dict, cmin = color_limits[0], cmax = color_limits[1])
        self.fig.add_trace(self.trace)
        
        # Update the camera view of figure
        eye = set_camera_eye(self.fig, view = self.view, eye_user = self.eye_dict)
        camera = dict(eye = eye)
        
        # remove catesian axis or not of figure
        if self.remove_cart_axis:
            self.remove_bg_and_cartaxis(len_scene = 1) # this is done at a single scene.
        # Create spherical grid or not
        if self.create_sph_axis:
            self.add_spherical_grid()
            
        _, max_radius, _ = self.get_discretizing_info()
        
        self.fig.update_layout(
            # title = dict(text = self.fig_title, x=0.5, y = 0.8, xanchor="auto", yanchor="middle" ),
            margin = dict(l=0, r=0, t=0, b=0),
            height = self.fig_size, width = self.fig_size,
            font = dict(family = self.font_family, size = self.font_size),
            scene = dict(aspectmode='data',
                      # xaxis=dict(range=[-max_radius, max_radius]),
                      # yaxis=dict(range=[-max_radius, max_radius]),
                      # zaxis=dict(range=[0, max_radius]),
                      aspectratio = dict(x=1, y=1, z=0.5)),
            scene_camera = camera, 
            paper_bgcolor="white", plot_bgcolor="white") # BG color helps parsing to matplotlib
            
    def pre_balloon(self, ):
        # normalize pressure
        pressure_norm = pressure_normalize(self.pressure)
        # normalized dB (setting the dinamic range as minimum and zero dB as max)
        pressure_spl = pressure2spl(pressure_norm, p_ref = 1)
        pressure_spl[pressure_spl < -self.dinrange] = -self.dinrange
        # Compute pressure return from color_method
        if self.color_method == 'normalized':
            color_data = pressure_norm
            color_limits = np.array([0, 1])
        elif self.color_method == 'dB':
            color_data = pressure_spl
            color_limits = np.array([-self.dinrange, 0])
        else:
            color_data = pressure_spl
            color_limits = np.array([-self.dinrange, 0])
        # Transform the spherical coordinates of the receivers to get radius of the plot
        radius_data, theta, phi = cart2sph(self.coords[:, 0], self.coords[:, 1], 
                                           self.coords[:, 2])
        # The radius of the plot using radius_method
        if self.radius_method == 'normalized':
            radius_data = pressure_norm #pressure_spl + dinrange
        elif self.radius_method == 'dB':
            radius_data = pressure_spl + self.dinrange
        elif self.radius_method == 'dBnormalized':
            pres_positive = pressure_spl + self.dinrange
            radius_data = pres_positive / np.amax(pres_positive)
        else:
            radius_data = pressure_spl + self.dinrange
        # Transform back the pressure data (radius) to caterian coordinates. Elevation and azimuth comes from receivers
        x, y, z = sph2cart(radius_data, theta, phi)
        return x, y, z, color_data, color_limits
    
    def remove_bg_and_cartaxis(self, len_scene = 1):
        """ remove the cartesian axis. It also works for subplots with len_scene
        
        Parameters
        ----------
        len_scene : int
            How many plots do you have in the scene.
        """
        for ic in range(len_scene):
            scene_text = f'scene{ic + 1}' if ic > 0 else 'scene'
            self.fig.layout[scene_text]['xaxis']['showbackground'] = False
            self.fig.layout[scene_text]['xaxis']['visible'] = False
            self.fig.layout[scene_text]['yaxis']['showbackground'] = False
            self.fig.layout[scene_text]['yaxis']['visible'] = False
            self.fig.layout[scene_text]['zaxis']['showbackground'] = False
            self.fig.layout[scene_text]['zaxis']['visible'] = False
    
    def add_spherical_grid(self, ):
        """
        Adds a spherical grid (elevation and azimuth lines) to a 3D Plotly figure.

        Parameters:
            fig: Plotly figure object to overlay the grid on.
            radius: float
                Radius of the spherical grid.
            delta_radius: float
                The spacing between circular grids in elevation and azimuth
            delta_azimuth: float
                The spacing between azimuth angles
            delta_elevation: float
                The spacing between elevation angles
            r_annotation_offset : float
                an offset applied to the annotations in the grid to make it readable
            az_color : str
                Color of the azimuth grid
            el_color : str
                 Color of the elevation grid
        """
        self.add_azimuthal_circunferences()
        self.add_azimuthal_radial_lines()
        
        if self.plot_elevation_grid:
            self.add_elevation_circunferences()
            self.add_elevation_radial_lines()
    
    def get_discretizing_info(self,):
        """ Get discretization information needed for computing the directivity
        
        Has to do with which radius method you chose.
        """
        if self.radius_method == 'normalized':
            delta_radius = 1 / self.num_of_radius
            all_radius = np.linspace(delta_radius, 1.0, int(1/delta_radius))
            max_radius = 1.0
            all_radius_str = [str(np.round(value,1)) for value in all_radius]
        elif self.radius_method == 'dB':
            delta_radius = self.dinrange / self.num_of_radius
            all_radius = np.linspace(delta_radius, self.dinrange, int(self.dinrange/delta_radius))
            # all_radius = np.linspace(delta_radius, 30, int(30/delta_radius))
            max_radius = self.dinrange
            all_radius_str = [str(np.round(value,1))+ ' dB' for value in all_radius-max_radius]
        else:
            delta_radius = 1 / self.num_of_radius
            all_radius = np.linspace(delta_radius, 1.0, int(self.dinrange/delta_radius))
            max_radius = 1.0
            all_radius_str = [str(np.round(value,1)) for value in all_radius]
            
        return all_radius, max_radius, all_radius_str
    
    def add_azimuthal_circunferences(self,):
        """ Add the azimuthal circunferences at several radii
        """
        # Add azimuth circle on z=0
        azimuths_circunference = np.linspace(-np.pi, np.pi, 100)
        # Add circles with different levels
                
        # delta_radius = self.dinrange / self.num_of_radius
        # all_radius = np.linspace(delta_radius, self.dinrange, int(self.dinrange/delta_radius))
        all_radius, max_radius, all_radius_str = self.get_discretizing_info()
        r_annotation_offset = 0.08 * max_radius
        for j, radii in enumerate(all_radius):
            # Circunferences
            x = radii * np.cos(azimuths_circunference) 
            y = radii * np.sin(azimuths_circunference) 
            z = np.zeros(len(azimuths_circunference))
                       
            self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', 
                                            line=dict(color = self.azimuth_grid_color, 
                                                      width=2, dash = self.line_style), showlegend=False))
            # Annotations
            x_annot, y_annot, z_annot = self.get_annotation(x[-1], y[-1], z[-1], 
                                r_annotation_offset = r_annotation_offset, 
                                phi_annotation_loc = -130)
            self.fig.add_trace(go.Scatter3d(x = [x_annot], y = [y_annot], z = [z_annot],
                                            mode='text', text=[all_radius_str[j]], 
                                            textfont_color = self.azimuth_grid_color,
                                            textposition='middle center', showlegend=False))
    
    def add_azimuthal_radial_lines(self,):
        """ Add the azimuthal radial lines
        """
        # Compute the azimutal angles at which to generate the radial lines
        azimuths_radial_lines_at = np.arange(0, 360, self.delta_azimuth)
        _, max_radius, _ = self.get_discretizing_info()
        r_annotation_offset = 0.08 * max_radius
        for jaz, az in enumerate(azimuths_radial_lines_at):
            dummy_rec = receivers.Receiver()
            dummy_rec.line_array(line_len = max_radius, step = max_radius/100, axis = 'x', 
                                 start_at = 0, zr = 0)
            dummy_rec.rotate_array(axis = 'z', theta_deg = az)
            # Add radial line
            self.fig.add_trace(go.Scatter3d(x = dummy_rec.coord[:,0], y = dummy_rec.coord[:,1], 
                                        z = dummy_rec.coord[:,2],
                                        mode='lines', line=dict(color=self.azimuth_grid_color, 
                                                                width=2, 
                                        dash = self.line_style), showlegend=False))
            # Annotations
            x_annot, y_annot, z_annot = self.get_annotation(dummy_rec.coord[-1,0], 
                                                            dummy_rec.coord[-1,1],
                                                            dummy_rec.coord[-1,2], 
                                                            r_annotation_offset = r_annotation_offset,
                                                            phi_annotation_loc = None)
            
            self.fig.add_trace(go.Scatter3d(x = [x_annot], y = [y_annot], z = [z_annot],
                                            mode='text', text=[str(az) + 'º'], 
                                            textfont_color=self.azimuth_grid_color,
                                            textposition='middle center', showlegend=False))
            
    def add_elevation_circunferences(self,):
        """ Add the elevation circunferences at several radii
        """
        # delta_radius = self.dinrange / self.num_of_radius
        # all_radius = np.linspace(delta_radius, self.dinrange, int(self.dinrange/delta_radius))
        all_radius, _, _ = self.get_discretizing_info()
    
        #Add elevation lines
        azimuths_el = np.arange(0, 360, 360)
        elevation_angles = np.linspace(0, np.pi / 2, 100)
        for radii in all_radius:
            x = radii * np.cos(elevation_angles) * np.cos(np.deg2rad(azimuths_el))
            y = radii * np.cos(elevation_angles) * np.sin(np.deg2rad(azimuths_el))
            z = radii * np.sin(elevation_angles)
            self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', 
                                        line=dict(color = self.elevation_grid_color, 
                                                  width=2, dash = self.line_style), 
                                        showlegend=False))
        
    def add_elevation_radial_lines(self,):
        """ Add the elevation radial lines
        """
        # Elevation radius
        elevation_radius = np.arange(0, 90, self.delta_elevation)
        _, max_radius, _ = self.get_discretizing_info()
        r_annotation_offset = 0.08 * max_radius
        for jel, el in enumerate(elevation_radius):
            dummy_rec = receivers.Receiver()
            dummy_rec.line_array(line_len = max_radius, step = max_radius/100, axis = 'z', 
                            start_at = 0, zr = 0)
            dummy_rec.rotate_array(axis = 'y', theta_deg = -el)
            # Add radial line
            self.fig.add_trace(go.Scatter3d(x = dummy_rec.coord[:,0], y = dummy_rec.coord[:,1], 
                                        z = dummy_rec.coord[:,2],
                                        mode='lines', line=dict(color=self.elevation_grid_color, 
                                                                width=2, dash = self.line_style), 
                                        showlegend=False))
            
            x_annot, y_annot, z_annot = self.get_annotation(dummy_rec.coord[-1,0], 
                                                            dummy_rec.coord[-1,1], 
                                                            dummy_rec.coord[-1,2],
                                                            r_annotation_offset = r_annotation_offset)
            self.fig.add_trace(go.Scatter3d(x = [x_annot], y = [y_annot], z = [z_annot],
                                        mode='text', text=[str(el) + 'º'], 
                                        textfont_color=self.elevation_grid_color,
                                        textposition='middle center', showlegend=False))
        
    def get_annotation(self, x, y, z, r_annotation_offset = 0, phi_annotation_loc = None):
        # Add annotation to radial line
        r, theta, phi = cart2sph(x, y, z)
        r += r_annotation_offset
        if phi_annotation_loc is not None:
            phi += phi_annotation_loc
        x_annot, y_annot, z_annot = sph2cart(r, theta, phi)
        return x_annot, y_annot, z_annot
    
    def plotly2matplotlib(self, ax = None, pixels_to_crop = [40, 40, 40, 100],
                          figsize = (7,5), keep_frame = False):
        """ Parse the plotly figure to matplotlib as an image
        
        The figure is then croped (remove the white spaces)
        """
        # Parse plotly figure to an  png image (increasing resolution leads to scaling down fonts)
        pdf_bytes = self.fig.to_image(format="pdf", width = self.fig_size, 
                                      height = self.fig_size)
        cropped_pdf_bytes = self.crop_pdf(pdf_bytes, pixels_to_crop = pixels_to_crop)
        
        # Convert to image
        images = pdf2image.convert_from_bytes(cropped_pdf_bytes, dpi = 300)
        cropped_img = images[0]
        
        # Create the figure and main axis for the image
        if ax is None:
            _, ax = plt.subplots(figsize = figsize)

        # Display the image
        ax.imshow(cropped_img, aspect='equal')
        # Add padding to make the spines visible
        # ax.set_xlim(-1, cropped_img.size[0])
        # ax.set_ylim(cropped_img.size[1], -1)
        # ax.axis("off")  # Turn off the axis for the image
        if keep_frame:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
        
        plt.tight_layout()
        return ax
    
    def add_colorbar_to_mplfigure(self, fig, font_size = 14, subplots_adjust = True):
        """
        Add a colorbar to an existing Matplotlib figure.
        
        Parameters:
            fig: Matplotlib figure object.
            vmin: Minimum value for the color scale.
            vmax: Maximum value for the color scale.
            cmap_name: Name of the colormap to use (e.g., 'viridis', 'plasma').
            shrink: Factor to shrink the colorbar height (1.0 is default).
        """
        # update font for the labels
        plt.rcParams["font.family"] = self.font_family
        
        if self.radius_method == 'normalized':
            ticks = np.round(np.arange(0, 1.2, 0.2),1)
            vmin, vmax = 0, 1
        else:
            ticks = np.arange(-self.dinrange, 5, 5)
            vmin, vmax = -self.dinrange, 0
        
        # Define the colormap and normalization
        cmap = plt.get_cmap(self.color_map)
        norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
        # Calculate colorbar dimensions with shrink applied
        width = 0.009  # Width of the colorbar (in figure coordinates)
        height = self.colorbar_shrink# * 0.6  # Height of the colorbar (adjusted by shrink)
        x = 0.85  # X position of the colorbar (near the right edge of the figure)
        y = (1 - height) / 2  # Center the colorbar vertically
        cbar_ax = fig.add_axes([x, y, width, height])  # [x, y, width, height] in figure coordinates
        # Create the colorbar        
        cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, ticks = ticks)
        cbar.ax.set_yticks(ticks)
        cbar.ax.set_yticklabels(list(map(str, ticks)), fontsize = font_size)
        cbar.set_label(self.colorbar_title, fontsize = font_size)  # Add a label (optional)
        if subplots_adjust:
            fig.subplots_adjust(left=0.01, bottom=0.01, right=0.8, top=0.99, wspace = 0.01)

    def crop_pdf(self, pdf_bytes, pixels_to_crop = [40, 30, 40, 100]):
        """ Crop PDF image
        
        The croping is done with the aid of pixels_to_crop
        
        Parameters
        ----------
        pixels_to_crop : list
            list with 4 elements: x0, y0 pixel coords of the crop, and 
            x1, y1 pixel coords of the crop
        """
        # Load the PDF from bytes
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        writer = PyPDF2.PdfWriter()

        # Access the first page (assumes a single-page PDF)
        page = reader.pages[0]

        # Get the current crop box (left, bottom, right, top)
        crop_box = page.mediabox
        crop_box = page.cropbox
        x0, y0, x1, y1 = map(float, [crop_box.left, crop_box.bottom, crop_box.right, crop_box.top])
        
        # Adjust the crop box to add/remove padding
        page.mediabox.lower_left = (x0 + pixels_to_crop[0], y0 + pixels_to_crop[1])
        page.mediabox.upper_right = (x1 - pixels_to_crop[2], y1 - pixels_to_crop[3])

        # Add the modified page to the writer
        writer.add_page(page)

        # Save cropped PDF to bytes
        cropped_pdf_bytes = io.BytesIO()
        writer.write(cropped_pdf_bytes)
        return cropped_pdf_bytes.getvalue()
    
def crop_pdf(pdf_bytes, buffer = [40, 30, 40, 100]):
    # Load the PDF from bytes
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    writer = PyPDF2.PdfWriter()

    # Access the first page (assumes a single-page PDF)
    page = reader.pages[0]

    # Get the current crop box (left, bottom, right, top)
    crop_box = page.mediabox
    crop_box = page.cropbox
    x0, y0, x1, y1 = map(float, [crop_box.left, crop_box.bottom, crop_box.right, crop_box.top])
    print('x0, y0, x1,y1: {},{},{},{}'.format(x0,y0,x1,y1))
    # Apply padding to crop the image (adjust these values as needed)
    new_x0 = x0 + buffer[0]
    new_y0 = y0 + buffer[1]
    new_x1 = x1 - buffer[2]
    new_y1 = y1 - buffer[3]

    # Print the adjusted crop box
    
    # Adjust the crop box to add/remove padding
    page.mediabox.lower_left = (x0 + buffer[0], y0 + buffer[1])
    page.mediabox.upper_right = (x1 - buffer[2], y1 - buffer[3])

    # Add the modified page to the writer
    writer.add_page(page)

    # Save cropped PDF to bytes
    cropped_pdf_bytes = io.BytesIO()
    writer.write(cropped_pdf_bytes)
    return cropped_pdf_bytes.getvalue()

# def crop_pdf(pdf_bytes, bounding_box):
#     # Open the PDF and create a PdfWriter to save the cropped PDF
#     reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
#     writer = PyPDF2.PdfWriter()

#     # Access the first page of the PDF
#     page = reader.pages[0]

#     # Get the crop box from the bounding box (left, bottom, right, top)
#     left, bottom, right, top = bounding_box

#     # Set the new mediabox to crop the PDF
#     page.mediabox.lower_left = (left, bottom)
#     page.mediabox.upper_right = (right, top)

#     # Add the modified page to the writer
#     writer.add_page(page)

#     # Save the cropped PDF to bytes
#     cropped_pdf_bytes = io.BytesIO()
#     writer.write(cropped_pdf_bytes)

#     return cropped_pdf_bytes.getvalue()

# Full function to detect bounding box and crop the PDF
def crop_pdf_based_on_image(pdf_bytes):
    # Convert the PDF to an image (PNG)
    img = convert_pdf_to_image(pdf_bytes)

    # Get the bounding box from the image
    # bounding_box = get_bounding_box_from_image(img)
    img2, bounding_box = crop_whitespace(img)

    # Crop the PDF using the bounding box
    cropped_pdf = crop_pdf(pdf_bytes, bounding_box)

    return cropped_pdf

def convert_pdf_to_image(pdf_bytes, dpi = 300):
    images = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi)  # Use a high DPI for quality
    return images[0]  # Return first page as a PIL Image

def get_bounding_box_from_image(img, white_threshold=245):
    img_array = np.array(img)  # Convert image to numpy array

    # Create a mask for non-white pixels (considering tolerance)
    non_white_mask = np.all(img_array[:, :, :3] < white_threshold, axis=-1)

    # Find the coordinates of the non-white pixels
    non_white_coords = np.argwhere(non_white_mask)

    if len(non_white_coords) == 0:
        return (0, 0, img_array.shape[1], img_array.shape[0])  # No non-white content, return full image

    # Get the min/max values for rows and columns (bounding box of non-white pixels)
    min_row, min_col = non_white_coords.min(axis=0)
    max_row, max_col = non_white_coords.max(axis=0)

    return (min_col, min_row, max_col, max_row)

def crop_whitespace(img, buffer=2):
    """
    Crops whitespace around an image, leaving a buffer of extra pixels.
    
    Parameters:
    - img (PIL.Image): The input image to crop.
    - buffer (int): The number of pixels to leave as a buffer around the cropped area.
    
    Returns:
    - PIL.Image: The cropped image with a buffer.
    """
    # Convert image to greyscale for better boundary detection
    gray_img = img.convert("L")  # Convert to grayscale
    bg = Image.new("L", img.size, 255)  # Create a white background
    diff = ImageChops.difference(gray_img, bg)
    bbox = diff.getbbox()  # Get the bounding box of the non-white areas
    
    if bbox:
        # Expand the bounding box by the buffer
        left = max(0, bbox[0] - buffer)
        upper = max(0, bbox[1] - buffer)
        right = min(img.size[0], bbox[2] + buffer)
        lower = min(img.size[1], bbox[3] + buffer)
        return img.crop((left, upper, right, lower))
    
    # Return the original image if no cropping is needed
    return img

def get_bbox_from_png(img, buffer=2):
    """
    Crops whitespace around an image, leaving a buffer of extra pixels.
    
    Parameters:
    - img (PIL.Image): The input image to crop.
    - buffer (int): The number of pixels to leave as a buffer around the cropped area.
    
    Returns:
    - PIL.Image: The cropped image with a buffer.
    """
    # Convert image to greyscale for better boundary detection
    gray_img = img.convert("L")  # Convert to grayscale
    bg = Image.new("L", img.size, 255)  # Create a white background
    diff = ImageChops.difference(gray_img, bg)
    original_bbox = img.getbbox()
    print("Original bbox of png figure: {}".format(original_bbox))
    bbox = diff.getbbox()  # Get the bounding box of the non-white areas
    print("bbox of difference of png figure: {}".format(bbox))
    if bbox:
        # Expand the bounding box by the buffer
        x_start = max(0, bbox[0] - buffer)
        y_start = max(0, bbox[1] - buffer)
        x_end = min(img.size[0], bbox[2] + buffer)
        y_end = min(img.size[1], bbox[3] + buffer)
        
        
        return (x_start, y_start, x_end, y_end)
    
    # # Return the original image if no cropping is needed
    # return img
    
def compare_directivities(pressure_dict , plt3Ddir, nrows = 1, ncols = 1, 
                          base_figsize = (5,5), font_size = 14,
                          pixels_to_crop = [35, 40, 40, 130],
                          fine_tune_subplt = [0.01, 0.01, 0.83, 0.99],
                          wspace = 0.03, hspace = 0.03):
    """ Compare directivities using matplotlib
    
    """
    # update font for the labels
    plt.rcParams["font.family"] = plt3Ddir.font_family
    # get dictionary data and labels
    subfig_titles = list(pressure_dict.keys())
    pressure_data = list(pressure_dict.values())
    
    # Instantiate figure
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, 
                           figsize = (ncols*base_figsize[0], nrows*base_figsize[1]),
                           squeeze = False)
    counter = 0
    for row in range(nrows):
        for col in range(ncols):
            plt3Ddir.pressure = pressure_data[counter]
            plt3Ddir.plot_3d_polar(use_matplotlib  = True)
            # # fig, ax = plt.subplots(figsize = (7,5))
            axs[row, col] = plt3Ddir.plotly2matplotlib(ax = axs[row, col] , 
                                                      pixels_to_crop = pixels_to_crop,
                                                      keep_frame = True)
            # Draw frames around each subplot
            for spine in axs[row, col].spines.values():
                spine.set_edgecolor('grey')
                spine.set_linestyle('--')     # Set dashed line style
                spine.set_linewidth(2)
            # axs[row, col].spines['top'].set_edgecolor('red')
            axs[row, col].set_title(subfig_titles[counter], fontsize = font_size,
                                    loc='right')
            counter += 1
    plt3Ddir.add_colorbar_to_mplfigure(fig, font_size = font_size, subplots_adjust = False)
    fig.subplots_adjust(left = fine_tune_subplt[0], bottom = fine_tune_subplt[1], 
                        right = fine_tune_subplt[2], top = fine_tune_subplt[3], 
                        wspace = wspace, hspace = hspace)
    return fig, axs
    
def get_random_direction(num_dim = 2):
    """ Compute a randum unitary direction vector
    """
    direction = np.random.uniform(low = -1, high = 1, size = num_dim)
    direction /= np.linalg.norm(direction)
    return direction

def move_point(point, direction, distance):
    """ move a point in a given direction
    """
    new_point = point + distance * direction
    return new_point

def get_max_dist(point, direction, lower_bounds, upper_bounds):
    """ Determines the max distance allowed
    """
    pass

def check_crop_bounding_box(point, lower_bounds, upper_bounds):
    """ Check if given point is inside the bounding box or not. If not crop it to bb.
    """
    # difference
    lb_check = point - lower_bounds
    ub_check = upper_bounds - point
    
    # if lb_check has negative values, crop to bounding box
    if (lb_check < 0).any():
        id_true = np.where(lb_check < 0)[0]
        # Petubation
        pertubation = np.random.uniform(low = 0, 
                                        high = upper_bounds-lower_bounds, 
                                        size = len(upper_bounds))
        # Perturbed crop
        point[id_true] = lower_bounds[id_true] + pertubation[id_true]
    # if ub_check has negative values, crop to bounding box
    if (ub_check < 0).any():
        id_true = np.where(ub_check < 0)[0]
        # Petubation
        pertubation = np.random.uniform(low = 0, 
                                        high = upper_bounds-lower_bounds, 
                                        size = len(upper_bounds))
        # Perturbed crop
        point[id_true] = upper_bounds[id_true] - pertubation[id_true]
    return point
      
def random_move(origin, lower_bounds, upper_bounds):
    """ Random move (1-time only)
    """
    # number of dimensions
    num_dim = len(origin)
    # Get random direction
    direction = get_random_direction(num_dim = num_dim)
    # Get random distance
    distance = np.random.uniform(low = 0, high = 1, size = 1)
    # Move point
    new_coord = move_point(point = origin, direction = direction, distance = distance)
    # Check if it is in Bounding box, then crop it to it if not
    new_coord = check_crop_bounding_box(point = new_coord,
                                        lower_bounds = lower_bounds, 
                                        upper_bounds = upper_bounds)
    return new_coord, direction

def radom_walk(origin, num_walks, lower_bounds, upper_bounds, seed = 0):
    """ Makes random walk
    """
    np.random.seed(seed)
    num_dim = len(origin)
    new_coords = np.zeros((num_walks+1, num_dim))
    new_coords[0,:] = origin
    directions = np.zeros((num_walks, num_dim))
    
    for i in range(num_walks):
        new_coords[i+1,:], directions[i,:] = random_move(origin = new_coords[i,:], 
                                                         lower_bounds = lower_bounds, 
                                                         upper_bounds = upper_bounds)
        # # Get random direction
        # directions[i,:] = get_random_direction(num_dim = num_dim)
        # # Get random distance
        # distance = np.random.uniform(low = 0, high = 1, size = 1)
        # # Move point
        # new_coords[i+1,:] = move_point(point = new_coords[i,:],
        #                                direction = directions[i,:], distance = distance)
        # # Check if it is in Bounding box, then crop it to it if not
        # new_coords[i+1,:] = check_crop_bounding_box(point = new_coords[i+1,:],
        #                                             lower_bounds = lower_bounds, 
        #                                             upper_bounds = upper_bounds)
    return new_coords, directions


    
def plot_pt_dir2D(point, direction, plot_arrows = True):
    """ plots the point and direction
    """
    # plt.figure(figsize = (6,3))
    plt.scatter(point[0], point[1], c = 'k', s = 10)
    if plot_arrows:
        plt.quiver(point[0], point[1], direction[0], direction[1])
    
def plot_random_walk_2D(points, directions, plot_arrows = True):
    """ plots random walk
    """
    num_pts = points.shape[0]
    plt.figure(figsize = (6,3))
    for i in range(num_pts-1):
        plot_pt_dir2D(points[i,:], direction = directions[i,:], plot_arrows = plot_arrows)
    plt.scatter(points[-1,0], points[-1,1], c = 'k', s = 10)
        
    plt.grid(linestyle = '--')
    minimum_range = 1.2*np.amin([np.amin(points[:,0]), np.amin(points[:,1])])
    maximum_range = 1.2*np.amax([np.amax(points[:,0]), np.amax(points[:,1])])    
    plt.xlim((minimum_range, maximum_range))
    plt.ylim((minimum_range, maximum_range))
    
def give_me_an_ax(figshape = (1, 1), figsize = (6,3)):
    """ return me a default matplotlib ax
    """
    fig, ax = plt.subplots(figsize = figsize,
                           nrows = figshape[0], ncols = figshape[1],
                           squeeze = False)
    return fig, ax


def plot_1d_curve(xdata, ydata, ax, xlims = None, ylims = None, 
                  color = 'tab:blue', linewidth = 1.5, marker = None, 
                  linestyle = '-', alpha = 1.0, label = None,
                  xlabel = None, ylabel = None,
                  linx = True, liny = True, xticks = None):
    """ plots 1d curve
    """
    # Set defaults
    if xlims is None:
        xlims = xdata.min(), xdata.max()
    if ylims is None:
        ylims = ydata.min(), ydata.max()
    if linx and liny:
        ax.plot(xdata, ydata, color = color, linewidth = linewidth, 
                linestyle = linestyle, alpha = alpha, label = label)
    elif ~linx and liny:
        ax.semilogx(xdata, ydata, color = color, linewidth = linewidth, 
                linestyle = linestyle, alpha = alpha, label = label)
    elif linx and ~liny:
        ax.semilogy(xdata, ydata, color = color, linewidth = linewidth, 
                linestyle = linestyle, alpha = alpha, label = label)
    else:
        ax.loglog(xdata, ydata, color = color, linewidth = linewidth, 
                linestyle = linestyle, alpha = alpha, label = label)
    ax.grid(linestyle = '--')
    if label is not None:
        ax.legend()
    if xticks is not None:
        xlabels = [str(num) for num in xticks]
        ax.set_xticks(xticks, xlabels)
        ax.minorticks_off()
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_time(fs, sig, ax = None, xlims = None, ylims = None, 
              color = 'tab:blue', linewidth = 1.5, linestyle = '-',
              alpha = 1.0, label = None):
    """ plot time signals (single channel)
    
    fs : int
        Sampling rate
    sig : numpy1dArray
        Signal in time dommain - single channel    
    """
    # Create axis if axis is None
    if ax is None:
        _, ax = give_me_an_ax()
        ax = ax[0,0]
    n_samples = len(sig)
    time = np.linspace(0, (n_samples-1)/fs, n_samples)
    plot_1d_curve(time, sig, ax, xlims = xlims, ylims = ylims,
                  color = color, linewidth = linewidth, 
                  linestyle = linestyle, alpha = alpha, 
                  label = label, xlabel = "Time [s]", ylabel = "Amplitude",
                  linx = True, liny = True)
    return ax

def plot_spk_mag(freq, spk, ax = None, xlims = None, ylims = None, 
              color = 'tab:blue', linewidth = 1.5, linestyle = '-',
              alpha = 1.0, label = None):
    """ plot magnitude spectrum of signals in dB (single channel)
    
    freq : numpy1dArray
        frequency vector
    spk_mag : numpy1dArray
        spectrum   
    """
    # Create axis if axis is None
    if ax is None:
        _, ax = give_me_an_ax()
        ax = ax[0,0]
    spk_mag_dB = 20*np.log10(np.abs(spk)) 
    plot_1d_curve(freq, spk_mag_dB, ax, xlims = xlims, ylims = ylims,
                  color = color, linewidth = linewidth, 
                  linestyle = linestyle, alpha = alpha, 
                  label = label, xlabel = "Frequency [Hz]", ylabel = "Magnitude [dB]",
                  linx = False, liny = True)
    return ax

def plot_spk_mag_pha(freq, spk, ax = None, xlims = None, ylims = None, 
              color = 'tab:blue', linewidth = 1.5, linestyle = '-',
              alpha = 1.0, label = None):
    """ plot spectrum of signals Magnitude and phase (single channel)
    
    freq : numpy1dArray
        frequency vector
    spk_mag : numpy1dArray
        spectrum   
    """
    # Create axis if axis is None
    if ax is None:
        _, ax = give_me_an_ax(figshape = (1, 2), figsize = (10,3))

    spk_mag_dB = 20*np.log10(np.abs(spk))
    spk_pha = np.rad2deg(np.angle(spk))
    # plot mag
    plot_1d_curve(freq, spk_mag_dB, ax[0,0], xlims = xlims, ylims = ylims,
                  color = color, linewidth = linewidth, 
                  linestyle = linestyle, alpha = alpha, 
                  label = label, xlabel = "Frequency [Hz]", ylabel = "Magnitude [dB]",
                  linx = False, liny = True)
    # plot phase
    plot_1d_curve(freq, spk_pha, ax[0,1], xlims = xlims, ylims = ylims,
                  color = color, linewidth = linewidth, 
                  linestyle = linestyle, alpha = alpha, 
                  label = label, xlabel = "Frequency [Hz]", ylabel = "Phase [deg.]",
                  linx = False, liny = True)    
    return ax

def plot_spk_re_imag(freq, spk, ax = None, xlims = None, ylims = None, 
              color = 'tab:blue', linewidth = 1.5, linestyle = '-',
              alpha = 1.0, label = None):
    """ plot spectrum of signals Real and Imag (single channel)
    
    freq : numpy1dArray
        frequency vector
    spk_mag : numpy1dArray
        spectrum   
    """
    # Create axis if axis is None
    if ax is None:
        _, ax = give_me_an_ax(figshape = (1, 2), figsize = (10,3))

    spk_re = np.real(spk)
    spk_im = np.imag(spk)
    # plot real
    plot_1d_curve(freq, spk_re, ax[0,0], xlims = xlims, ylims = ylims,
                  color = color, linewidth = linewidth, 
                  linestyle = linestyle, alpha = alpha, 
                  label = label, xlabel = "Frequency [Hz]", ylabel = "Real [-]",
                  linx = False, liny = True)
    # plot imag
    plot_1d_curve(freq, spk_im, ax[0,1], xlims = xlims, ylims = ylims,
                  color = color, linewidth = linewidth, 
                  linestyle = linestyle, alpha = alpha, 
                  label = label, xlabel = "Frequency [Hz]", ylabel = "Imag. [-]",
                  linx = False, liny = True)    
    return ax

def plot_absorption(freq, abs_coeff, ax = None, xlim = None, ylim = None, 
                    color = 'tab:blue', linewidth = 1.5, linestyle = '-',
                    alpha = 1.0, label = None):
    """ Plot absorption coefficient
    
    Parameters
    ----------
    ax : matplotlib axes or None
    """    
    # Create axis if axis is None
    if ax is None:
        _, ax = give_me_an_ax()
        ax = ax[0,0]
    
    plot_1d_curve(freq, abs_coeff, ax, xlims = xlim, ylims = ylim,
                  color = color, linewidth = linewidth, 
                  linestyle = linestyle, alpha = alpha, 
                  label = label, xlabel = "Frequency [Hz]", ylabel = r"$\alpha$  [-]",
                  linx = False, liny = True, 
                  xticks = [31.5, 63, 125, 250, 500, 1000, 2000, 4000])
    return ax

def plot_absorption_theta(theta, abs_coeff, ax = None, xlim = None, ylim = None,
                          color = 'tab:blue', linewidth = 1.5, linestyle = '-',
                          alpha = 1.0, label = None):
    """ Plot absorption coefficient as a function of the incidence angle
    
    Parameters
    ----------
    ax : matplotlib axes or None
    """    
    # Create axis if axis is None
    if ax is None:
        _, ax = give_me_an_ax()
        ax = ax[0,0]
    
    plot_1d_curve(theta, abs_coeff, ax, ylims = (-0.2, 1.2),
                  color = color, linewidth = linewidth, 
                  linestyle = linestyle, alpha = alpha, 
                  label = label, xlabel = r"$\theta$ [deg]", ylabel = r"$\alpha$  [-]",
                  linx = True, liny = True, xticks = np.arange(0, 105, 15))
    return ax
    
    
def MPM(samp, L, Ts, tol):
    """ Matrix Pencil Method (implemented by Martin Eser - JASA 2021)
    
    Parameters
    ----------
    samp : numpy1dArray
        Samples of the variable of interest
    L : int
        Pencil Parameter
    Ts : float
        Sampling period
    tol : float
        Tolerance for SVD truncation
    
    Parameters
    ----------
    An : numpy1dArray
        Amplitudes in transformed space
    Bn : numpy1dArray
        Locations in transformed space
    """
    # samp: samples of reflection coefficient; sampled at each K_z0 calculated from t
    # L: pencil parameter
    # Ts: sampling rate deltat in Eser, 2021
    # tol: SVD truncation threshold

    Y = scipy.linalg.hankel(samp)[: len(samp) - L, : L + 1]  # full Hankel matrix
    U, s, Vh = np.linalg.svd(Y)  # singular value decomposition (SVD) of Hankel matrix
    indx = np.where(s < s.max() * tol)[0][
        0
    ]  # truncation of SVD given prescribed threshold tol
    Vpr = Vh.conj().T[:, :indx]  # V^f in Eser, 2021

    V1pr = Vpr[:-1, :]  # V_1^fH in Eser, 2021
    V2pr = Vpr[1:, :]  # V_2^fH in Eser, 2021
    spr = np.zeros((np.shape(U)[0], np.shape(V1pr)[-1]))  # Sigma^f in Eser, 2021
    np.fill_diagonal(spr, s[:indx])
    Y1 = (U.dot(spr)).dot(V1pr.conj().T)
    Y2 = (U.dot(spr)).dot(V2pr.conj().T)

    poles = scipy.linalg.eigvals(
        np.dot(scipy.linalg.pinv(Y1), Y2)
    )  # Eq. (15) in Eser, 2021
    poles = poles[:indx]  # z_n in Eser, 2021

    # alp = np.log(np.abs(poles[:]))/(Ts)
    # freqs = np.arctan2(np.imag(poles[:]),np.real(poles[:]))/(2*np.pi*Ts)
    # Bn = (alp+1j*(2*np.pi*freqs))*Ts
    Bn = np.log(poles[:]) / Ts  # Eq. (16) in Eser, 2021

    Vrhs = np.array((samp))  # right hand side of Eq. (17) in Eser, 2021
    Van = np.vstack(
        [poles ** (i) for i in range(np.shape(samp)[0])]
    )  # Vandermonde matrix in Eq. (17) in Eser, 2021
    # calculate An from least-squares Vandermonde system
    An = np.linalg.lstsq(Van, Vrhs, rcond=None)[0]  # Solution of Eq. (17) in Eser, 2021

    return An, Bn  # output of sought coeffcients of the series of exponentials
    
   