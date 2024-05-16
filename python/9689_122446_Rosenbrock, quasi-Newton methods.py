get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np

def objfun(x,y):
    return 10*(y-x**2)**2 + (1-x)**2
def gradient(x,y):
    return np.array([-40*x*y + 40*x**3 -2 + 2*x, 20*(y-x**2)])
def hessian(x,y):
    return np.array([[120*x*x - 40*y+2, -40*x],[-40*x, 20]])

def contourplot(objfun, xmin, xmax, ymin, ymax, ncontours=50, fill=True):

    x = np.linspace(xmin, xmax, 200)
    y = np.linspace(ymin, ymax, 200)
    X, Y = np.meshgrid(x,y)
    Z = objfun(X,Y)
    if fill:
        plt.contourf(X,Y,Z,ncontours); # plot the contours
    else:
        plt.contour(X,Y,Z,ncontours); # plot the contours
    plt.scatter(1,1,marker="x",s=50,color="r");  # mark the minimum

contourplot(objfun, -7,7, -10, 40, fill=False)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contours of $f(x,y)=10(y-x^2)^2 + (1-x)^2$");

def sr1(objfun, gradient, init, tolerance=1e-6, maxiter=10000):
    x = np.array(init)
    iterno = 0
    B = np.identity(2)
    xarray = [x]
    fprev = objfun(x[0],x[1])
    farray = [fprev]
    gprev = gradient(x[0],x[1])
    xtmp = x - 0.01*gprev/np.sqrt(np.dot(gprev,gprev))
    gcur = gradient(xtmp[0],xtmp[1])
    s = xtmp-x
    y = gcur-gprev
    while iterno < maxiter:
        r = y-np.dot(B,s)
        B = B + np.outer(r,r)/np.dot(r,s)        
        x = x - np.linalg.solve(B,gcur)
        fcur = objfun(x[0], x[1])
        if np.isnan(fcur):
            break
        gprev = gcur
        gcur = gradient(x[0],x[1])
        xarray.append(x)
        farray.append(fcur)
        if abs(fcur-fprev)<tolerance:
            break
        fprev = fcur
        s = xarray[-1]-xarray[-2]
        y = gcur-gprev
        iterno += 1
    return np.array(xarray), np.array(farray)

p, f = sr1(objfun, gradient, init=[2,4])

f

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

p, f = sr1(objfun, gradient, init=[-1,9])

plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -3,3,-10,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");
plt.xlim(-3,3)
plt.ylim(-10,10)
        
plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");



