import numpy as np
import numpy.polynomial as nppoly


N = 4    # number of underlying assets
N_p = 4    # N'
S = np.array(np.ones(N))    # asset prices at time 0
r = 0    # risk-free interest rate
T = 1    # maturity of option
t = np.array(np.ones(N))    # time for calculating prices of underlying assets
q = np.array(np.zeros(N))    # dividend rates of underlying assets
sigma_0 = np.array(np.ones(N))    # standard deviations of underlying assets
corr = np.identity(N)    # correlation coefficients of underlying assets
w = np.array(np.ones(N))    # weights of underlying assets
e = np.array(np.zeros(N))    # unit basic vector for Householder reflection matrix
e[0] = 1
ee = np.array(np.ones(N))    # unit vector with all elements equal 1
M = np.array(np.ones(N) * 10)    # dimension vector, the numbers of node sides
M[0] = 4.
M[1] = 6.
M[2] = 8.
M[3] = 7.
M_max = np.max(M[1:N_p])    # maximum of dimension from j = 2 to j = N'
z = np.tile( 10., (int(N_p - 0.5),int(M_max + 0.5)))    # a matrix for values of node sides, a row is nodes of a dimension
h = np.tile( 10., (int(N_p - 0.5),int(M_max + 0.5)))    # a matrix for weights of node sides, a row is weights of a dimension

F = S[:,None] * np.exp(r * T - t[:,None] * q[:,None])    # forward prices of underlying assets
g = w[:,None] * F / (w[None,:] @ F)    # normalized forward-adjust weight factors
sigma = sigma_0[:,None] @ sigma_0[None,:] * corr    # covariance matrix of underlying assets
C = np.linalg.cholesky(sigma)    # Cholesky decomposation
Q_x1 = np.transpose(C) @ g / np.sqrt(np.transpose(g) @ sigma @ g)    # the first column of standrad orthogonal matrix Q
V_x1 = sigma @ g / np.sqrt(np.transpose(g) @ sigma @ g)    # the first column of matrix V, which is a square root of covariance matrix sigma
for p in range(N):
    if V_x1[p] < 0:
        V_x1[p] = 0.1 * np.sign(w[p]) * np.sqrt(sigma[p,p])
CV = np.linalg.inv(C) @ V_x1    # inverse C multiplise V_*1
mju = np.sqrt(np.transpose(Q_x1) @ Q_x1) / np.sqrt(np.transpose(CV) @ CV)
V_x1 = mju * V_x1    # adjusted V_*1
v = (Q_x1 - e[:,None]) / np.sqrt(np.transpose(Q_x1 - e[:,None]) @ (Q_x1 - e[:,None]))  # adjusted vector for Householder reflection matrix
R = np.identity(N) - 2 * v @ np.transpose(v)    # Householder reflection matrix
CR = C @ R[:,1:N]    # sub-matrix for SVD
CR_2 = np.transpose(CR) @ CR
A, Q_dot = np.linalg.eig(CR_2)
Sgr = np.array(np.sqrt(A))    # singular value
U = CR @ Q_dot / np.vstack((Sgr, Sgr, Sgr, Sgr))    # unitary matrix
D = np.diag(np.sqrt(A))    # singular value matrix
V = np.hstack((V_x1, U @ D))

for p in range(int(N_p - 0.5)):
    zz, hh = nppoly.hermite_e.hermegauss(deg = int(M[p + 1] + 0.5))   
    z[p,0:int(M[p + 1] + 0.5)] = np.array(zz)    # save the node sides values
    h[p,0:int(M[p + 1] + 0.5)] = np.array(hh)    # save the weights of node sides
    print(p)
    print(zz)
    print(hh)
    print(z[p,:])
    print(h[p,:])
    print( )

print(z)
print( )
print(h)

import numpy as np
a = np.array([1.,2.,3.,4.,5.])
b = np.tile(1., (int(2. + 0.5),int(2. + 0.5)))
print(b)


import numpy as np
import numpy.polynomial as nppoly
#zz, hh = nppoly.hermite_e.hermegauss(deg = 100)
#print(hh[0:10])
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
b = np.reshape(a,(4,3))
c = np.reshape(b,(12,1))
d = np.array(np.ones(12))
print(b)
print(c)
e = np.hstack((c,d[:,None]))
print(e)