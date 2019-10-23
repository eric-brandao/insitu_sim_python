import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import insitu_cpp

def vert_2d(normal, vertcoord):
    '''
    Function to transform the 3D plane to 2D.
    Input: normal of the plane
           3D vertex coordinates of the plane (array of 1x3 arrays)
    Output: vert_x: a numpy array of x-coordinates of vertexes
            vert_y: a numpy array of y-coordinates of vertexes 
    '''
    # Find biggest component of the normal - component to ignore
    normal_abs = np.absolute(normal)
    normal_id = np.where(normal_abs == normal_abs.max())
    
    # vert_x
    vert_x = []
    vert_y = []
    for row in vertcoord:
        v = np.delete(row, normal_id)
        vert_x.append(v[0])
        vert_y.append(v[1])
    
    # append the first vertex to the end of the list (circular)
    v = np.delete(vertcoord[0], normal_id)
    vert_x.append(v[0])
    vert_y.append(v[1])

    # Transform in numpy array
    vert_x = np.array(vert_x)
    vert_y = np.array(vert_y)
    return vert_x, vert_y

def rp_2d(normal, ref_point):
    '''
    Function to transform the 3D rp to 2D.
    Input: normal of the plane
    ref_point (array 1x3)
    Output: ref_point2d: a numpy array of 1x2 of 2D reflection point
    '''
    # Find biggest component of the normal - component to ignore
    normal_abs = np.absolute(normal)
    normal_id = np.where(normal_abs == normal_abs.max())
    print(normal_id)
    # ref_point2D
    ref_point2d = np.delete(ref_point, normal_id)
    return ref_point2d

def main():

    ray_origin = np.array([0.5, -0.232, 0.7])
    v_in = np.array([0.0, 0.0, -1.0])
    normal = np.array([0.0, 0.0, 1.0])
    vertcoord = np.array([[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0],
        [1.0, 0.5, 0.0], [-1.0, 0.5, 0.0]])

    # Test reflection point
    print(vertcoord[2])
    Rp = insitu_cpp._refpoint(ray_origin, v_in, normal, vertcoord[2])
    # print('The reflection point back to python is {}.'.format(Rp))
    # Test lambda and distance
    # Rp = np.array([0.57, -0.23, 0.0])
    lam = insitu_cpp._whichside(ray_origin, v_in, Rp)
    print('Lambda back to python is {}.'.format(lam))
    # print('dist back to python is {}.'.format(lam[0][1]))

    
    #####  Test point in polygon ######
    # 2D x and y lists
    vert_x, vert_y = vert_2d(normal, vertcoord)
    # print(vertcoord)
    # print(vert_x)
    # print(vert_y)
    # vert_x = np.array(v_x)

    Rp_2d = rp_2d(normal, Rp)
    print(Rp_2d)
    
    # point in polygon
    wn = insitu_cpp._ptinpol(vert_x, vert_y, Rp_2d)
    print('The winding number back to python is {}.'.format(wn))

  

if __name__ == '__main__':
    main()