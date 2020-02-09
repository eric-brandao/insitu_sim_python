
import numpy as np
from scipy import spatial as sp_spatial


class SphereTessellator():

    def __init__(self, nverts=12, depth=0):
        '''Tessellates a sphere with a number of vertices greater or equal than
        `nverts`.
        If `depth` is greater than 0 then `nverts` is ignored and the number of
        vertices is a consequence of `depth`.
        The returned object exposes the sphere vertices and indices through
        a property named `sphere`.
        '''
        self.vertices, self.indices = self.icosahedron()
        if depth > 0:
            self.depth = depth
        elif nverts >= 12:
            self.depth = self.nverts2depth(nverts)
        else:
            msg = (
                'SphereTessellator was passed nverts:{} and depth:{}, but '
                'expected nverts >= 12 or depth >=0'.format(nverts, depth)
            )
            raise SphereTessellatorBadArgsError(msg)

        self.iterate()

    def nverts2depth(self, nverts):
        '''Returns the minimum number of iterations, `depth`, to tessellate a
        sphere with `nverts` vertices'''
        nv = 12  # num of vertices of an icosahedron
        nf = 20  # num of faces of an icosahedron
        ne = 30  # num of edges of an icosahedron
        depth = 1  # iteration
        while nv < nverts:
            depth += 1
            nv += ne
            ne = 2*ne + 3*nf
            nf *= 4
        return depth

    def icosahedron(self,):
        x = .525731112119133606
        z = .850650808352039932
        vertices = np.array([
            [-x, 0.0, z], [x, 0.0, z], [-x, 0.0, -z], [x, 0.0, -z],
            [0.0, z, x], [0.0, z, -x], [0.0, -z, x], [0.0, -z, -x],
            [z, x, 0.0], [-z, x, 0.0], [z, -x, 0.0], [-z, -x, 0.0]
        ], dtype=np.float32)
        hull = sp_spatial.ConvexHull(vertices)
        indices = hull.simplices
        return vertices, indices

    def norm(self, arr):
        arr[:] = (arr.T / np.sqrt(np.sum(arr**2, axis=1))).T

    def iterate(self,):
        vs, ids = self.vertices, self.indices
        for _ in range(self.depth - 1):
            v0 = vs[ids[:, 0]]
            v1 = vs[ids[:, 1]]
            v2 = vs[ids[:, 2]]
            a = (v0 + v2) * 0.5
            b = (v0 + v1) * 0.5
            c = (v1 + v2) * 0.5
            self.norm(a)
            self.norm(b)
            self.norm(c)
            vs = np.unique(np.concatenate((vs, a, b, c)), axis=0)
            hull = sp_spatial.ConvexHull(vs)
            ids = hull.simplices
        self.vertices, self.indices = vs, ids

    @property
    def sphere(self,):
        return self.vertices, self.indices


class SphereTessellatorBadArgsError(Exception):
    '''Exception raised when the constructor is passed bad arguments.'''
    pass


if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as a3
    import scipy as sp

    tess = SphereTessellator(
        # nverts=12,
        depth=1
    )
    vertices, indices = tess.sphere
    # vertices *= 0.5

    hull = sp_spatial.ConvexHull(vertices)
    indices = hull.simplices
    faces = vertices[indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.dist = 30
    ax.azim = -140
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for f in faces:
        face = a3.art3d.Poly3DCollection([f])
        face.set_edgecolor('k')
        face.set_alpha(0.5)
        ax.add_collection3d(face)

    plt.show()
