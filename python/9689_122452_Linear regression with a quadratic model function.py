get_ipython().magic('matplotlib notebook')
import matplotlib.pylab as plt
plt.style.use("ggplot")
import numpy as np

x = np.random.uniform(-1,2,20)

y = x**2 - 2*x + 1 + np.random.normal(0,0.2,20)

plt.scatter(x,y)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y");

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x,x**2,y)

from sklearn import linear_model
reg = linear_model.LinearRegression()
X = np.vstack((x,x**2)).transpose()
reg.fit(X,y)

reg.intercept_, reg.coef_[0], reg.coef_[1]

fig = plt.figure()
ax = fig.gca(projection='3d')
Xf = np.arange(-1.1, 2.2, 0.25)
Yf = np.arange(-0.2, 4.25, 0.25)
Xf, Yf = np.meshgrid(Xf, Yf)
Zf = reg.intercept_ + reg.coef_[0]*Xf + reg.coef_[1]*Yf
surf = ax.plot_surface(Xf, Yf, Zf, alpha=0.2)
ax.scatter(x,x**2,y,c="r")

yf = reg.predict(X)

fig = plt.figure()
 
ax = fig.gca(projection='3d')

Xf = np.arange(-1.1, 2.2, 0.25)
Yf = np.arange(-0.2, 4.25, 0.25)
Xf, Yf = np.meshgrid(Xf, Yf)
Zf = reg.intercept_ + reg.coef_[0]*Xf + reg.coef_[1]*Yf
surf = ax.plot_surface(Xf, Yf, Zf, alpha=0.2)

for i in range(len(X)):
    ax.plot([X[i,0], X[i,0]], [X[i,1],X[i,1]], [yf[i], y[i]], linewidth=2, color='r', alpha=.5)
ax.plot(X[:,0], X[:,1], y, 'o', markersize=8, 
        markerfacecolor='none', color='r')

plt.figure()
xx = np.linspace(-1,2)
ytrue = xx**2 - 2*xx + 1
yy = reg.coef_[1]*xx**2 + reg.coef_[0]*xx + reg.intercept_
plt.plot(xx, yy, "r")
#plt.plot(xx,ytrue, "g--")
plt.scatter(x,y)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y");



