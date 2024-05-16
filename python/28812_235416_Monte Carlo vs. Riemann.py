import numpy as np
import matplotlib.pyplot as plt

def grid(xl, xu, N, D):
    """
    Create a grid of N evenly spaced points in D-dimensional space
    xl: lower bound of x
    xu: upper bound of x
    N: number of points per dimension
    D: number of dimensions
    """
    xr = np.linspace(xl, xu, N)
    g = np.zeros((N**D, D))
    for n in range(N**D):
        index = np.unravel_index(n, tuple([N]*D))
        g[n] = [xr[i] for i in index]
        
    return g

def f(x):
    return (1 - (np.sum(np.square(x)) / x.size))

def riemann(N, D):
    # riemann integration
    x = grid(0.0, 1.0, N, D)
    dx = 1.0 / (N**D)
    F_r = np.sum(np.apply_along_axis(f, 1, x) * dx)
    return F_r

def monte_carlo(N, D):
    # monte carlo integration
    x = np.random.rand(N**D, D)
    F_mc = np.sum(np.apply_along_axis(f, 1, x)) / (N**D)
    return F_mc

D = 1
N = np.logspace(1, 3, num=10, dtype=int)

F_r = np.zeros(10)
for i,n in enumerate(N):
    F_r[i] = riemann(n, D)
    
# error in riemann estimate
e_r = np.abs(F_r - 2.0/3.0)
print(e_r)

plt.plot(N, e_r)
plt.plot(N, (e_r[0]*N[0])/N)
plt.legend(['Error in Riemann estimate', '1/N'])
plt.show()

repeats = 1000
e_mc = np.zeros(10)
for i,n in enumerate(N):
    for r in range(repeats):
        e_mc[i] += np.abs(monte_carlo(n, D) - 2.0/3.0)

e_mc /= repeats
print(e_mc)

plt.plot(N, e_mc)
plt.plot(N, (e_mc[0]*np.sqrt(N[0]))/np.sqrt(N))
plt.legend(['Error in Monte Carlo estimate', r"$\frac{1}{N^{-1/2}}$"])
plt.show()

D = 2
N = np.array([5, 10, 20 , 50, 100])

F_rd = np.zeros(5)
for i,n in enumerate(N):
    F_rd[i] = riemann(n, D)
    
e_rd = np.abs(F_rd - 2.0/3.0)

plt.plot(N, e_rd)
plt.plot(N, (e_rd[0]*N[0])/(N))
plt.plot(N, (e_rd[0]*(N[0]**D))/(N**D))
plt.legend(['Error in Riemann estimate', '1/N', '1/N^D'])
plt.show()

D = 3
N = np.array([3, 6, 12 , 25, 50])

F_rd = np.zeros(5)
for i,n in enumerate(N):
    F_rd[i] = riemann(n, D)
    
e_rd = np.abs(F_rd - 2.0/3.0)

plt.plot(N, e_rd)
plt.plot(N, (e_rd[0]*N[0])/(N))
plt.plot(N, (e_rd[0]*(N[0]**D))/(N**D))
plt.legend(['Error in Riemann estimate', '1/N', '1/N^D'])
plt.show()

D = [1,2,3,4,5]
N = 10

F_rn = np.zeros(5)
for i,d in enumerate(D):
    F_rn[i] = riemann(N, d)
    
e_rn = np.abs(F_rn - 2.0/3.0)

plt.plot(D, e_rn)
plt.show()

D = np.array([1, 2, 4, 8, 16])
# we need to keep the number of samples constant across dimensions
N = np.array([65536, 256, 16, 4, 2])
F_rc = np.zeros(5)
for i,(n,d) in enumerate(zip(N,D)):
    F_rc[i] = riemann(n, d)
    
# error in riemann estimate
e_rc = np.abs(F_rc - (2.0 / 3.0))
print(e_rc)

plt.plot(D, e_rc)
plt.plot(D, (e_rc[0]*N[0])/N)
plt.legend(['Error in Riemann estimate', '1/N'])
plt.show()

repeats = 100
D = 2
N = np.array([5, 10, 20 , 50, 100])
e_mcd2 = np.zeros(5)
for i,n in enumerate(N):
    for r in range(repeats):
        e_mcd2[i] += np.abs(monte_carlo(n, D) - (2.0/3.0))

e_mcd2 /= repeats
print(e_mcd2)

plt.plot(N, e_mcd2)
plt.plot(N, (e_mcd2[0]*(N[0]))/N)
plt.legend(['Error in Monte Carlo estimate', r"$\frac{1}{\sqrt{N^D}}$"])
plt.show()

repeats = 100
D = np.array([1, 2, 3, 4, 5])
# we need to keep the number of samples constant across dimensions
N = 10
e_mcd = np.zeros(5)
for i,d in enumerate(D):
    for r in range(repeats):
        e_mcd[i] += np.abs(monte_carlo(N, d) - (2.0/3.0))

e_mcd /= repeats
print(e_mcd)

plt.plot(D, e_mcd)
D = np.array([1, 2, 3, 4, 5])
plt.plot(D, (e_mcd[0]*np.sqrt(N))/np.sqrt(N**D))
plt.legend(['Error in Monte Carlo estimate', r"$\frac{1}{\sqrt{N^D}}$"])
plt.show()

