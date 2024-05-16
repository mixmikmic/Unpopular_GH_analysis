# generate data
import numpy as np
import scipy.stats as stat

np.random.seed(123)
m1 = [2., 2., 2., 2.]
m2 = [-2., -2., -2., -2.]
m3 = [-2., -2., 2., 2.]
m4 = [2., 2., -2., -2.]
x1 = stat.multivariate_normal.rvs(m1, size=150)
x2 = stat.multivariate_normal.rvs(m2, size=150)
x3 = stat.multivariate_normal.rvs(m3, size=150)
x4 = stat.multivariate_normal.rvs(m4, size=150)
tx = np.vstack([x1[0:100], x2[0:100], x3[0:100], x4[0:100]])
vx = np.vstack([x1[100:], x2[100:], x3[100:], x4[100:]])
ty = np.zeros(400, dtype=np.int8)
vy = np.zeros(200, dtype=np.int8)
ty[100:200] = 1
vy[50:100] = 1
ty[200:300] = 2
vy[100:150] = 2
ty[300:400] = 3
vy[150:200] = 3

# shuffle
rperm = np.random.permutation(400)
tx = tx[rperm]
ty = ty[rperm]

# randomly initialize parameters (we want all runs to start from the same initial parameters)
w_rec_init = np.random.randn(4, 2).T
b_rec_init = np.random.randn(2)
w_gen_init = np.random.randn(2, 4).T
b_gen_init = np.random.randn(4)

def calc_dl_dgen(xi, w_gen, b_gen, w_rec, b_rec, L=10):
    """Calculate derivative of L with respect to generative parameters
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
        L: Number of samples used to estimate derivatives
    Returns:
        ndarray: Derivative of L wrt to generative weight matrix
        ndarray: Derivative of L wrt to generative biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    dl_dw_gen = 0.0
    dl_db_gen = 0.0
    for l in range(L):
        # latent sample
        zi = mu_zi + np.random.randn(*mu_zi.shape)
        # generate x from latent sample
        mu_xi = np.dot(w_gen, zi) + b_gen
        # calculate derivatives
        dl_dw_gen += np.outer((xi - mu_xi), zi)
        dl_db_gen += (xi - mu_xi)
    dl_dw_gen /= L
    dl_db_gen /= L
    return dl_dw_gen, dl_db_gen

def calc_dl_drec(xi, w_gen, b_gen, w_rec, b_rec, L=10):
    """Calculate derivative of L with respect to recognition parameters
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
        L: Number of samples used to estimate derivatives
    Returns:
        ndarray: Derivative of L wrt to recognition weight matrix
        ndarray: Derivative of L wrt to recognition biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    dl_dw_rec = 0.0
    dl_db_rec = 0.0
    for l in range(L):
        # latent sample
        zi = mu_zi + np.random.randn(*mu_zi.shape)
        # run generation network
        mu_xi = np.dot(w_gen, zi) + b_gen
        # log p_{\theta}(x,z_l) - log q_{\phi}(z_l|x)
        d = np.sum(np.square(zi - mu_zi)) - np.sum(np.square(xi - mu_xi))
        dl_dw_rec += d*np.outer((zi - mu_zi), xi)
        dl_db_rec += d*(zi - mu_zi)
    dl_dw_rec /= L
    dl_db_rec /= L
    return dl_dw_rec, dl_db_rec

def calc_log_ll(x, w_gen, b_gen, w_rec, b_rec):
    log_ll = 0.0
    for i in range(x.shape[0]):
        xi = x[i]
        zp = np.dot(w_rec, xi) + b_rec
        xp = np.dot(w_gen, zp) + b_gen
        log_ll += -np.sum(np.square(xp - xi))
    return log_ll / x.shape[0]
    

# learn parameters using gradient ascent
w_gen_ga = w_gen_init.copy()  
b_gen_ga = b_gen_init.copy()  
w_rec_ga = w_rec_init.copy()  
b_rec_ga = b_rec_init.copy()  

np.random.seed(456)

epoch_count = 50
lr_gen = 1e-3
lr_rec = lr_gen * 1e-1

log_ll_ga = np.zeros(epoch_count+1)
log_ll_ga[0] = calc_log_ll(vx, w_gen_ga, b_gen_ga, w_rec_ga, b_rec_ga)
print "Initial val log ll: {0:f}".format(log_ll_ga[0])

dwrec_magnitude = np.zeros(tx.shape[0])
for e in range(epoch_count):
    for i in range(tx.shape[0]):
        xi = tx[i]
        dwgen, dbgen = calc_dl_dgen(xi, w_gen_ga, b_gen_ga, w_rec_ga, b_rec_ga, 1)
        w_gen_ga += lr_gen * dwgen
        b_gen_ga += lr_gen * dbgen
        dwrec, dbrec = calc_dl_drec(xi, w_gen_ga, b_gen_ga, w_rec_ga, b_rec_ga, 1)
        w_rec_ga += lr_rec * dwrec
        b_rec_ga += lr_rec * dbrec  
        
        dwrec_magnitude[i] = np.sum(np.square(dwrec))

    log_ll_ga[e+1] = calc_log_ll(vx, w_gen_ga, b_gen_ga, w_rec_ga, b_rec_ga)
    
print "Final val log ll: {0:f}, grad w_rec magnitude in last epoch: {1:f}+-{2:f}".format(log_ll_ga[-1], 
                                                                              np.mean(dwrec_magnitude), 
                                                                              np.std(dwrec_magnitude))

log_ll_ga

get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (10, 8)

def plot_z(x, y, w_rec, b_rec):
    z = np.zeros((x.shape[0], 2))
    for i in range(x.shape[0]):
        xi = x[i]
        zi = np.dot(w_rec, xi) + b_rec
        z[i] = zi
    plt.scatter(z[:,0], z[:,1], c=y)
  
# plot the 2D latent space
plot_z(tx, ty, w_rec_ga, b_rec_ga)

def calc_dl_drec_reparameterized(xi, w_gen, b_gen, w_rec, b_rec, L=10):
    """Calculate derivative of L with respect to recognition parameters using the reparameterization trick
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
        L: Number of samples used to estimate derivatives
    Returns:
        ndarray: Derivative of L wrt to recognition weight matrix
        ndarray: Derivative of L wrt to recognition biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    dl_dw_rec = 0.0
    dl_db_rec = 0.0
    for l in range(L):
        # latent sample
        zi = stat.multivariate_normal.rvs(mu_zi)
        # run generation network
        mu_xi = np.dot(w_gen, zi) + b_gen
        # calculate derivatives
        dl_dw_rec += np.dot(w_gen.T, np.outer((xi - mu_xi), xi))
        dl_db_rec += np.dot(w_gen.T, (xi - mu_xi))
    dl_dw_rec /= L
    dl_db_rec /= L
    return dl_dw_rec, dl_db_rec

w_gen_rt = w_gen_init.copy()  
b_gen_rt = b_gen_init.copy()  
w_rec_rt = w_rec_init.copy()  
b_rec_rt = b_rec_init.copy()  

np.random.seed(456)

epoch_count = 100
lr_gen = 1e-3
lr_rec = lr_gen * 1e-1

log_ll_rt = np.zeros(epoch_count+1)
log_ll_rt[0] = calc_log_ll(x, w_gen_rt, b_gen_rt, w_rec_rt, b_rec_rt)
print "Initial log ll: {0:f}".format(log_ll_rt[0])

dwrec_magnitude = np.zeros(x.shape[0])
for e in range(epoch_count):
    for i in range(x.shape[0]):
        xi = x[i]
        dwgen, dbgen = calc_dl_dgen(xi, w_gen_rt, b_gen_rt, w_rec_rt, b_rec_rt, 1)
        w_gen_rt += lr_gen * dwgen
        b_gen_rt += lr_gen * dbgen
        dwrec, dbrec = calc_dl_drec_reparameterized(xi, w_gen_rt, b_gen_rt, w_rec_rt, b_rec_rt, 1)
        w_rec_rt += lr_rec * dwrec
        b_rec_rt += lr_rec * dbrec    
        
        dwrec_magnitude[i] = np.sum(np.square(dwrec))

    log_ll_rt[e+1] = calc_log_ll(x, w_gen_rt, b_gen_rt, w_rec_rt, b_rec_rt)

print "Final log ll: {0:f}, grad w_rec magnitude in last epoch: {1:f}+-{2:f}".format(log_ll_rt[-1], 
                                                                              np.mean(dwrec_magnitude), 
                                                                              np.std(dwrec_magnitude))

# plot the 2D latent space
plot_z(x, y, w_rec_rt, b_rec_rt)

def wake_sleep(xi, w_gen, b_gen, w_rec, b_rec, L=10):
    """Apply one step of wake-sleep algorithm and get updates for generative and recognition parameters
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
        L: Number of samples used to estimate derivatives
    Returns:
        ndarray: Update for generative weight matrix
        ndarray: Update for generative biases
        ndarray: Update for recognition weight matrix
        ndarray: Update for recognition biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    dl_dw_gen = 0.0
    dl_db_gen = 0.0
    dl_dw_rec = 0.0
    dl_db_rec = 0.0
    for l in range(L):
        # latent sample
        zi = stat.multivariate_normal.rvs(mu_zi)
        # generate x from latent sample
        mu_xi = np.dot(w_gen, zi) + b_gen
        xi_pred = stat.multivariate_normal.rvs(mu_xi)
        # run recognition on predicted x
        mu_zi_pred = np.dot(w_rec, xi_pred) + b_rec
        zi_pred = stat.multivariate_normal.rvs(mu_zi)
        # calculate derivatives
        dl_dw_gen += np.outer((xi - xi_pred), zi)
        dl_db_gen += (xi - xi_pred)
        dl_dw_rec += np.outer((zi - zi_pred), xi_pred)
        dl_db_rec += (zi - zi_pred)
    dl_dw_gen /= L
    dl_db_gen /= L
    dl_dw_rec /= L
    dl_db_rec /= L
    return dl_dw_gen, dl_db_gen, dl_dw_rec, dl_db_rec

w_gen_ws = w_gen_init.copy()  
b_gen_ws = b_gen_init.copy()  
w_rec_ws = w_rec_init.copy()  
b_rec_ws = b_rec_init.copy()  

np.random.seed(456)

epoch_count = 100
lr_gen = 1e-4
lr_rec = lr_gen * 1e-1

log_ll_ws = np.zeros(epoch_count+1)
log_ll_ws[0] = calc_log_ll(x, w_gen_ws, b_gen_ws, w_rec_ws, b_rec_ws)
print "Initial log ll: {0:f}".format(log_ll_ws[0])

dwrec_magnitude = np.zeros(x.shape[0])
for e in range(epoch_count):
    for i in range(x.shape[0]):
        xi = x[i]
        dwgen, dbgen, dwrec, dbrec = wake_sleep(xi, w_gen_ws, b_gen_ws, w_rec_ws, b_rec_ws, 1)
        w_gen_ws += lr_gen * dwgen
        b_gen_ws += lr_gen * dbgen
        w_rec_ws += lr_rec * dwrec
        b_rec_ws += lr_rec * dbrec    
        
        dwrec_magnitude[i] = np.sum(np.square(dwrec))

    log_ll_ws[e+1] = calc_log_ll(x, w_gen_ws, b_gen_ws, w_rec_ws, b_rec_ws)

print "Final log ll: {0:f}, grad w_rec magnitude in last epoch: {1:f}+-{2:f}".format(log_ll_ws[-1], 
                                                                              np.mean(dwrec_magnitude), 
                                                                              np.std(dwrec_magnitude))

plot_z(x, y, w_rec_ws, b_rec_ws)

def calc_dl_dgen_ae(xi, w_gen, b_gen, w_rec, b_rec):
    """Calculate derivative of R (reconstruction error for classical autoencoder) 
    with respect to generative parameters
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
    Returns:
        ndarray: Derivative of R wrt to generative weight matrix
        ndarray: Derivative of R wrt to generative biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    # generate x from latent z
    mu_xi = np.dot(w_gen, mu_zi) + b_gen
    # calculate derivatives
    dl_dw_gen = np.outer((mu_xi - xi), mu_zi)
    dl_db_gen = (mu_xi - xi)
    return dl_dw_gen, dl_db_gen

def calc_dl_drec_ae(xi, w_gen, b_gen, w_rec, b_rec):
    """Calculate derivative of R (reconstruction error for classical autoencoder) 
    with respect to recognition parameters
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
    Returns:
        ndarray: Derivative of R wrt to recognition weight matrix
        ndarray: Derivative of R wrt to recognition biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    # run generation network
    mu_xi = np.dot(w_gen, mu_zi) + b_gen
    # calculate derivatives
    dl_dw_rec = np.dot(w_gen.T, np.outer((mu_xi - xi), xi))
    dl_db_rec = np.dot(w_gen.T, (mu_xi - xi))
    return dl_dw_rec, dl_db_rec

w_gen_ae = w_gen_init.copy()  
b_gen_ae = b_gen_init.copy()  
w_rec_ae = w_rec_init.copy()  
b_rec_ae = b_rec_init.copy()  

np.random.seed(456)

epoch_count = 100
lr_gen = 1e-3
lr_rec = lr_gen * 1e-1

log_ll_ae = np.zeros(epoch_count+1)
log_ll_ae[0] = calc_log_ll(x, w_gen_ae, b_gen_ae, w_rec_ae, b_rec_ae)
print "Initial log ll: {0:f}".format(log_ll_ae[0])

dwrec_magnitude = np.zeros(x.shape[0])
for e in range(epoch_count):
    for i in range(x.shape[0]):
        xi = x[i]
        dwgen, dbgen = calc_dl_dgen_ae(xi, w_gen_ae, b_gen_ae, w_rec_ae, b_rec_ae)
        w_gen_ae -= lr_gen * dwgen
        b_gen_ae -= lr_gen * dbgen
        dwrec, dbrec = calc_dl_drec_ae(xi, w_gen_ae, b_gen_ae, w_rec_ae, b_rec_ae)
        w_rec_ae -= lr_rec * dwrec
        b_rec_ae -= lr_rec * dbrec    
        
        dwrec_magnitude[i] = np.sum(np.square(dwrec))

    log_ll_ae[e+1] = calc_log_ll(x, w_gen_ae, b_gen_ae, w_rec_ae, b_rec_ae)
    
print "Final log ll: {0:f}, grad w_rec magnitude in last epoch: {1:f}+-{2:f}".format(log_ll_ae[-1], 
                                                                              np.mean(dwrec_magnitude), 
                                                                              np.std(dwrec_magnitude))

# plot the 2D latent space
plot_z(x, y, w_rec_ae, b_rec_ae)

# Let us compare the log likelihoods for each algorithm
plt.plot(range(81), log_ll_ga[20:])
plt.plot(range(81), log_ll_rt[20:])
plt.plot(range(81), log_ll_ws[20:])
plt.plot(range(81), log_ll_ae[20:])
plt.legend(['Gradient ascent', 'Variational autoencoder', 'Wake-sleep algorithm', 'Classical autoencoder'], loc='best')

