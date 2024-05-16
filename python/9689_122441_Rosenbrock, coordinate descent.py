get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np

def objfun(x,y):
    return 100*(y-x**2)**2 + (1-x)**2
def gradient(x,y):
    return np.array([-40*x*y + 40*x**3 -2 + 2*x, 20*(y-x**2)])
def hessian(x,y):
    return np.array([[120*x*x - 40*y+2, -40*x],[-40*x, 20]])

def contourplot(objfun, xmin, xmax, ymin, ymax, ncontours=50, fill=True):

    x = np.linspace(xmin, xmax, 300)
    y = np.linspace(ymin, ymax, 300)
    X, Y = np.meshgrid(x,y)
    Z = objfun(X,Y)
    if fill:
        plt.contourf(X,Y,Z,ncontours); # plot the contours
    else:
        plt.contour(X,Y,Z,ncontours); # plot the contours
    plt.scatter(1,1,marker="x",s=50,color="r");  # mark the minimum

conts = sorted(set([objfun(2,y) for y in np.arange(-2,5,0.25)]))
contourplot(objfun, -3,3, -2, 5, ncontours=conts,fill=False)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contours of $f(x,y)=100(y-x^2)^2 + (1-x)^2$");

def coordinatedescent(objfun, gradient, init, dim=2, tolerance=1e-6, maxiter=10000, steplength=0.01):
    p = np.array(init)
    iterno=0
    endflag = False
    parray = [p]
    fprev = objfun(p[0],p[1])
    farray = [fprev]
    eye = np.eye(dim)
    while iterno < maxiter: # main loop
        for d in range(dim): # loop over dimensions
            g = gradient(p[0],p[1])
            p = p - steplength*g[d]*eye[d]
            fcur = objfun(p[0], p[1])
            parray.append(p)
            farray.append(fcur)
            

        if abs(fcur-fprev)<tolerance:
            break
        fprev = fcur
        iterno += 1
    return np.array(parray), np.array(farray)

p, f = coordinatedescent(objfun, gradient, init=[2,4], steplength=0.005,maxiter=10000)

plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -1,3,0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");

plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");

p, f = coordinatedescent(objfun, gradient, init=[2,4], steplength=0.01,maxiter=500)

plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -2,3,0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");

plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");

p, f = coordinatedescent(objfun, gradient, init=[2,6], steplength=0.01)

f

p, f = coordinatedescent(objfun, gradient, init=[2,6], steplength=0.005)

plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -2,3,0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");

plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");

p, f = coordinatedescent(objfun, gradient, init=[2,5.1155], steplength=0.01)

plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -3,3,0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");

plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");



