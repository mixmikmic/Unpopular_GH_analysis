import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

fn = r"C:\\crs\\proj\\gps\\results.csv"
csv = np.genfromtxt (fn, delimiter=",")
x = csv[:,1]
y = csv[:,2]
z = csv[:,3]

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='darkgray', marker='.')

ax.set_xlabel('Easting (m?)')
ax.set_ylabel('Northing (m?)')
ax.set_zlabel('Z (m?)')

plt.show()
plt.savefig('C:\\crs\\proj\\gps\\results.png',transparent=True,bbox_inches='tight')

