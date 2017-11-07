import numpy as np
import numpy.polynomial as nppoly
import scipy.stats as ss

# parameters setting
n = 3 #number of assets
w = np.ones(n) #weight of assets
F = np.ones(n) #forward price of assets
K = 100

# create a test martrix
X = np.random.uniform(size = n**2).reshape(n, n)
X = np.triu(X)
cov_matrix = X.T*X

#Cholesky decomposition C of covariance matrix
C = np.linalg.cholesky(cov_matrix)


#normalized forward-adjusted weight vector
g = w*F/np.linalg.norm(w*F)

#equation 22，generate Q_1 and V_1
Q_1 = C.T@g/np.sqrt(g.T@cov_matrix@g)
V_1 = C@Q_1
V_1.shape = (n,1)

#obtain full V
e1 = np.zeros(n)
e1[0] = 1
Qe_1 = Q_1-e1
v = (Q_1-e1)/np.linalg.norm(Q_1-e1)
R = np.eye(n)-2*v@v.T

#singular value decomposition
U,D,Q = np.linalg.svd(C@R[:,1:],full_matrices=False)

V = np.hstack((V_1,U@np.diag(D)))

#generate equation 7, f_k
def coeff_func(z,k):
    return np.exp(-0.5*sum(V[k-1,1:]**2)+V[k-1,1:]@z.T[1:])  
	
# solve z1 from equation 20
def unitroot_func(z):
    g.shape = (n,1)
    gV = g*V
    summ = []
    for k in range(n):
        e = gV[k-1,:]@z
        summ.append(e)
    V1 = V[:,0]
    V1.shape = (3,1)
    z1 = (K-sum(summ[1:])-sum(g))/(g.T@V[:,0])
    #return print(k,sum(summ[1:]),sum(g),g.T@V[:,0])
    return z1


#formular 9, price of bs, with input of z_dot
def price_bs(z):
    summ2 = []
    dz_dot = unitroot_func(z)
    for k in range(1,n):
        e2 = g[k]*coeff_func(z,k)*ss.norm.cdf(dz_dot+V[k,0])-K*ss.norm.cdf(dz_dot)
        summ2.append(e2)
    price = sum(summ2)
    
    return(price)
	
def GHQ_n(func, n = 3, deg = 10):
    x, w = nppoly.hermite_e.hermegauss(deg=deg)
    w = w / np.sqrt(2.0*np.pi)
    weight_points = []
    for i in range(deg**n):
        index = []
        points = []
        weights = []
    
        for j in range(n):
            remain = i%deg
            exa_divi = i//deg
            index.append(remain)

        for k in range(n):
            points.append(x[index[k]])
            weights.append(w[index[k]])

        weight_points.append(func(points)*np.cumproduct(weights)[n-1])
    
    return sum(weight_points)
	
