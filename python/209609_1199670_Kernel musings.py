get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12,6))

x = np.linspace(0,4,200)
y = np.exp(-x*x/2)

ax.plot(x, y)
ax.plot(x, x*y)
None

np.sum(x*y) * (x[1]-x[0])

fig, ax = plt.subplots(ncols=2, figsize=(16,6))

x = np.linspace(0,10,1000)
omega = 1
y = omega * omega * np.exp(-x * omega)

ax[0].plot(x, x*y)
ax[1].plot(x, 1 - (1 + omega * x) * np.exp(-omega * x))
None

np.sum(x*y) * (x[1]-x[0])

fig, ax = plt.subplots(ncols=2, figsize=(16,6))

x = np.linspace(0,10,1000)
e = 1
y = (e+e) / ((1+x*x)**(1+e))

ax[0].plot(x, x*y)
ax[1].plot(x, (1-(1+x*x)**(-e)))
None

np.sum(x*y) * (x[1]-x[0])





