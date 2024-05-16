from scipy.integrate import odeint
from pylab import *
get_ipython().magic('matplotlib inline')

# decay rates are global so they can be seen inside the function 
global lam
lam = array([.04, .01, .000, .0])

# define a function to represent coupled ordinary differential eqns.
def dcdt(c,t):
    dfdt = np.zeros(4)
    dfdt[0] = c[0]* -lam[0] - c[0]*lam[3]
    dfdt[1] = c[1]* -lam[1] + c[0]*lam[0] - c[1]*lam[3] 
    dfdt[2] = c[2]* -lam[2] + c[1]*lam[1] - c[2]*lam[3]
    dfdt[3] =                 c[2]*lam[2] - c[3]*lam[3]
    return dfdt
    
# intial concentration for four constituents
C0 = array([.68, .23, .06, 0.])
# time array
t = linspace(0.0,100.,50)
# call 
C = odeint(dcdt,C0,t)

print "Shape of the final concentration matrix: ",shape(C)
fig = plt.figure()
plt.plot(t,C[:,0],label='$DDE$')
plt.plot(t,C[:,1],label='$DDMU$')
plt.plot(t,C[:,2],label='$DDNU$')
plt.plot(t,C[:,3],label='?')
plt.plot(t,np.sum(C,1),label='Total')
plt.xlabel('Time (years)')
plt.ylabel('Concentration ($\mu$mol / kg)')
plt.legend(loc='upper right')

whos

