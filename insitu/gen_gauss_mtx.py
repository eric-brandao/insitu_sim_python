import math
import numpy as np
import pickle

# Original gauss points and weigths (-1 to +1)
zeta = np.array([-0.93246951, -0.66120939, -0.23861918,
    0.23861918, 0.66120939, 0.93246951], dtype=np.float32)

weigths = np.array([0.17132449, 0.36076157, 0.46791393,
    0.46791393, 0.36076157, 0.17132449], dtype=np.float32)

# Create vectors of size 1 x 36 for the zetas
N1 = 0.25 * np.matmul(np.reshape(1-zeta, (zeta.size,1)),  np.reshape(1-zeta, (1,zeta.size)))
N2 = 0.25 * np.matmul(np.reshape(1+zeta, (zeta.size,1)),  np.reshape(1-zeta, (1,zeta.size)))
N3 = 0.25 * np.matmul(np.reshape(1+zeta, (zeta.size,1)),  np.reshape(1+zeta, (1,zeta.size)))
N4 = 0.25 * np.matmul(np.reshape(1-zeta, (zeta.size,1)),  np.reshape(1+zeta, (1,zeta.size)))

N1 = np.reshape(N1, (1,zeta.size**2))
N2 = np.reshape(N2, (1,zeta.size**2))
N3 = np.reshape(N3, (1,zeta.size**2))
N4 = np.reshape(N4, (1,zeta.size**2))

# Let each line of the following matrix be a N vector
Nzeta = np.zeros((4, zeta.size**2), dtype=np.float32)
Nzeta[0,:] = N1
Nzeta[1,:] = N2
Nzeta[2,:] = N3
Nzeta[3,:] = N4

# Create vector of size 1 x 36 for the weights
Nweigths = np.matmul(np.reshape(weigths, (zeta.size,1)),  np.reshape(weigths, (1,zeta.size)))
Nweigths = np.reshape(Nweigths, (1,zeta.size**2))

print(Nweigths)
print(Nweigths.shape)

# path_filename = '/home/eric/dev/insitu/data/' + 'gauss_data' + '.pkl'
# with open(path_filename, 'wb') as output:
#     pickle.dump(Nzeta, output, pickle.HIGHEST_PROTOCOL)
#     pickle.dump(Nweigths, output, pickle.HIGHEST_PROTOCOL)

# f = open(path_filename, 'wb')
# pickle.dump(self.__dict__, f, 2)
# f.close()

# N1 = 0.25 * (1-zeta.T) * (1-zeta)


# print(N1)
# N1 = np.reshape(N1, (1,zeta.size**2))
# print(N1)
# print(N1.shape)

# for value in wzeta:
#     Nzeta
