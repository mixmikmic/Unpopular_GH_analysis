import numpy as np

# data
x = np.array([2.5, 0.3, 2.8, 0.5])
y = np.array([1, -1, 1, 1])

# model
w0 = 0
w1 = 1
y_hat = w0 + w1*x

y_hat

# model
def f(x, w=np.array([w0, w1])):
    return w[0] + w[1]*x


# P[y = +1 | x,w] = sigmoid(f(x,w))
def sigmoid(x):
    f_x = f(x)
    return 1.0 / (1.0 + np.exp(-f(x)))
    

# P(y | x,w)
#     P[y = +1 | x,w] = sigmoid(f(x,w))
#     P[y = -1 | x,w] = 1 - sigmoid(f(x,w))
def P(y, x):
    f_x = f(x)
    if y == 1:
        return 1.0 / (1.0 + np.exp(-f_x))
    else:
        return np.exp(-f_x) / (1.0 + np.exp(-f_x))


# Unit function: 
#     1[y = c] = 1 if y == c
#     1[y = c] = 0 if y != c
def unit(y, c=1):
    if y == c:
        return 1
    else:
        return 0

likelihoods = np.array([P(y[i], x[i]) for i in range(4)])
likelihoods

data_likelihood = likelihoods.prod()
data_likelihood

derivative_log_likelihoods = np.array([ x[i] * (unit(y[i], 1) - sigmoid(x[i])) for i in range(4)])
derivative_log_likelihoods

data_derivative_log_likelihood = derivative_log_likelihoods.sum()
data_derivative_log_likelihood

