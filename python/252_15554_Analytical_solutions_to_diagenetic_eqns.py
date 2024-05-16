import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

# C = Co exp(-kt )
# 0.5 = exp (-k thalf)
# 7Be, 53.3
thalf = 53.3*3600.*24.
k=-np.log(0.5)/thalf
print '7Be k =',k,' s-1'

# 234Th, 24.1 days
thalf = 24.1*3600.*24.
k=-np.log(0.5)/thalf
print '234Th k =',k,' s-1'

Co=1.
k = .1
thalf = -np.log(0.5)/k
tmax = 5.*thalf
dt = tmax / 10.
t = np.arange(0.,tmax+dt,dt)

C = Co*np.exp(-k*t)

fig = plt.figure(figsize=(6,4))
plt.plot(t,C)
plt.xlabel('Time')
plt.ylabel('C/Co')
ts = 'Half-life: {0:.2f}'.format(thalf)
plt.title(ts)

import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
zmax = 1.
dz = 0.12
zb = np.arange(0.,zmax+dz,dz)
Co=1
w = 0.01
k = .1

C = Co*np.exp(-k*zb/w)

fig = plt.figure(figsize=(6,4))
plt.plot(C,-zb)
plt.xlabel('C/Co')
plt.ylabel('z (m)')
ts = 'w = {0:.2f} m/s; k = {1:.2f} 1/s'.format(w,k)
plt.title(ts)

