get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np
from time import sleep

N = 100 # count of data points
Xpos = np.random.multivariate_normal((1,1),[[1,0],[0,1]],int(N/2))
Xneg = np.random.multivariate_normal((-2,-2),[[1,0],[0,1]],int(N/2))

plt.scatter(Xpos[:,0],Xpos[:,1])
plt.scatter(Xneg[:,0],Xneg[:,1])
plt.legend(("Positive", "Negative"));

X = np.vstack((Xpos,Xneg))
y = np.hstack((np.ones(int(N/2)),np.zeros(int(N/2))))

ind = np.random.permutation(N)
X = np.hstack((np.ones((N,1)), X[ind,:])) # add the fixed column of ones for bias.
y = y[ind]

w = np.random.uniform(-1,1,3)
print(w)

xline = np.linspace(-4,3,100)
yline = -(w[1]*xline+w[0])/w[2]
plt.plot(xline, yline)
plt.scatter(Xpos[:,0],Xpos[:,1])
plt.scatter(Xneg[:,0],Xneg[:,1])
plt.legend(("wx=0","Positive", "Negative"));

xline = np.linspace(-4,3,100)

for repeat in range(50):
    predicted = [1 if i else 0 for i in np.dot(X,w)>0];
    misclassified = [i for i in range(N) if predicted[i]!=y[i]]
    if not misclassified:
        print("best solution found")
        break
    # Choose a random misclassified point
    i = np.random.choice(misclassified)
    # Update the weight vector
    w = w + 1*(y[i]-predicted[i])*X[i,:]
    #print(w)
    
    plt.figure()
    yline = -(w[1]*xline+w[0])/w[2]
    plt.plot(xline, yline)
    plt.scatter(Xpos[:,0],Xpos[:,1])
    plt.scatter(Xneg[:,0],Xneg[:,1])
    plt.legend(("wx=0","Positive", "Negative"))



