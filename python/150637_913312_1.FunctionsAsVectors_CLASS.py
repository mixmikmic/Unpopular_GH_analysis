get_ipython().run_line_magic('pylab', 'inline')

import numpy as np
#import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual,widgets
import ipywidgets as widgets

print 'version of ipwidgets=',widgets.__version__

import sys
sys.path.append('lib')
from recon_plot import recon_plot
from Eigen_decomp import Eigen_decomp
from YearPlotter import YearPlotter

# We define a grid that extends from o to 2*pi
step=2*pi/365
x=arange(0,2*pi,step)
len(x)

c=sqrt(step/(pi))
v=[]
v.append(np.array(cos(0*x))*c/sqrt(2))
v.append(np.array(sin(x))*c)
v.append(np.array(cos(x))*c)
v.append(np.array(sin(2*x))*c)
v.append(np.array(cos(2*x))*c)
v.append(np.array(sin(3*x))*c)
v.append(np.array(cos(3*x))*c)
v.append(np.array(sin(4*x))*c)
v.append(np.array(cos(4*x))*c)

print"v contains %d vectors"%(len(v))

# plot some of the functions (plotting all of them results in a figure that is hard to read.
figure(figsize=(8,6))
for i in range(5):
    plot(x,v[i])
grid()
legend(['const','sin(x)','cos(x)','sin(2x)','cos(2x)'])

for i in range(len(v)): 
    print
    for j in range(len(v)):
        a=dot(v[i],v[j])
        a=round(1000*a+0.1)/1000
        print '%1.0f'%a,

U=vstack(v)
shape(U)

f1=abs(x-4)
plot(x,f1);
grid()

eigen_decomp=Eigen_decomp(x,f1,np.zeros(len(x)),v)
recon_plot(eigen_decomp,year_axis=False,Title='Best Reconstruction',interactive=False);

eigen_decomp=Eigen_decomp(x,f1,np.zeros(len(x)),v)
plotter=recon_plot(eigen_decomp,year_axis=False,interactive=True);
display(plotter.get_Interactive())

noise=np.random.normal(size=x.shape)
f2=2*v[1]-4*v[5] +1*noise
plot(x,f2);

eigen_decomp=Eigen_decomp(x,f2,np.zeros(len(x)),v)
plotter=recon_plot(eigen_decomp,year_axis=False,interactive=True);
display(plotter.get_Interactive())







