import numpy as np

from sympy import *

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

f = x**2*cos(y) + exp(z)*sin(y)

J = np.array(
    [diff(f, x), diff(f, y), diff(f, z)])

print J

print np.array([
    diff(f, x).subs({x:pi, y:pi, z:1}),
    diff(f, y).subs({x:pi, y:pi, z:1}),
    diff(f, z).subs({x:pi, y:pi, z:1})])

u = x**2*y - cos(x)*sin(y)
v = exp(x+y)

J = np.array([
    [diff(u, x), diff(u, y)],
    [diff(v, x), diff(v, y)]
])
    
print J

print np.array([
    [diff(u, x).subs({x:0, y:pi}), diff(u, y).subs({x:0, y:pi})],
    [diff(v, x).subs({x:0, y:pi}), diff(v, y).subs({x:0, y:pi})]
])

f = x**3*cos(y) + x*sin(y)

H = np.array([
    [diff(diff(f, x), x), diff(diff(f, x), y)],
    [diff(diff(f, y), x), diff(diff(f, y), y)]
])

print H

f = x*y + sin(y)*sin(z) + (z**3)*exp(x)

H = np.array([
    [diff(diff(f, x), x), diff(diff(f, x), y), diff(diff(f, x), z)],
    [diff(diff(f, y), x), diff(diff(f, y), y), diff(diff(f, y), z)],
    [diff(diff(f, z), x), diff(diff(f, z), y), diff(diff(f, z), z)]
])

print H

f = x*y*cos(z) - sin(x)*exp(y)*(z**3)

H = np.array([
    [diff(diff(f, x), x), diff(diff(f, x), y), diff(diff(f, x), z)],
    [diff(diff(f, y), x), diff(diff(f, y), y), diff(diff(f, y), z)],
    [diff(diff(f, z), x), diff(diff(f, z), y), diff(diff(f, z), z)]
])

print H

print np.array([
    [diff(diff(f, x), x).subs({x:0, y:0, z:0}), diff(diff(f, x), y).subs({x:0, y:0, z:0}), diff(diff(f, x), z).subs({x:0, y:0, z:0})],
    [diff(diff(f, y), x).subs({x:0, y:0, z:0}), diff(diff(f, y), y).subs({x:0, y:0, z:0}), diff(diff(f, y), z).subs({x:0, y:0, z:0})],
    [diff(diff(f, z), x).subs({x:0, y:0, z:0}), diff(diff(f, z), y).subs({x:0, y:0, z:0}), diff(diff(f, z), z).subs({x:0, y:0, z:0})]
])

