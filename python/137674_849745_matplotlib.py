import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256)
C,S = np.cos(X), np.sin(X)

plt.plot(X,C)
plt.plot(X,S)

#plt.show() is a must to draw in our notebook
plt.show()

X = [1, 2, 3, 4, 5]
Y = [2, 3, 4, 5, 6]
plt.plot(X, Y, label = 'ratio: 1')
plt.plot(X, [i * 2 for i in Y], label = 'ratio: 2')
plt.title('test')
plt.xlabel('x axis')
plt.ylabel('y axis')
# draw our legend box
plt.legend()
plt.show()

x_axis_first = np.random.uniform(size = [10])
x_axis_second = np.random.uniform(size = [10])
y_axis_first = np.random.uniform(size = [10])
y_axis_second = np.random.uniform(size = [10])
plt.scatter(x_axis_first, y_axis_first, color = 'r', label = 'red scatter')
plt.scatter(x_axis_second, y_axis_second, color = 'b', label = 'blue scatter')
plt.title('test')
plt.xlabel('x axis')
plt.ylabel('y axis')
# draw our legend box
plt.legend()
plt.show()

n = 12
X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)
Y2 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x,y in zip(X,Y1):
    plt.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

plt.ylim(-1.25,+1.25)
plt.show()

def f(x,y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)

plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap='jet')
C = plt.contour(X, Y, f(X,Y), 8, colors='black', linewidth=.5)
plt.show()

plt.subplot(2,2,1)
plt.subplot(2,2,3)
plt.subplot(2,2,4)

plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

plt.show()



