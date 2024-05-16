import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Latex, display

import sys

get_ipython().magic('matplotlib inline')
np.random.seed(1)

K = 100  # The dimensionality of the ambient space (can be up to 2^16 for FEM solutions)
n = K    # The truncation dimension of the PCA / embedding dimension of the manifold 
m = 6    # The dimension off the measurement space

# First make two random orthonormal vector bases
Phi = sp.stats.ortho_group.rvs(dim=K) # The "PCA" space
Psi = sp.stats.ortho_group.rvs(dim=K) # The "measurement" space

sigma = np.sort(np.random.random(n))[::-1]
sigma[n:] = 0
Sigma = np.pad(np.diag(sigma), ((0,K-n),(0,K-n)), 'constant')
Sigma_inv = np.pad(np.diag(1.0/sigma), ((0,K-n),(0,K-n)), 'constant')
Sigma_n = np.diag(sigma)
Sigma_n_inv = np.diag(1.0/sigma)

V = Phi[:,:n]
W = Psi[:,:m]
W_p = Psi[:,m:]

T = Psi.T @ Phi @ Sigma_inv @ Sigma_inv @ Phi.T @ Psi
S = Psi.T @ Phi @ Sigma @ Sigma @ Phi.T @ Psi

T21 = T[m:, :m]
T22 = T[m:, m:]
S11 = S[:m, :m]
S21 = S[m:, :m]
S22 = S[m:, m:]

T22_alt = W_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W_p
T21_alt = W_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W
S11_alt = W.T @ V @ Sigma_n @ Sigma_n @ V.T @ W
S22_alt = W_p.T @ V @ Sigma_n @ Sigma_n @ V.T @ W_p
S21_alt = W_p.T @ V @ Sigma_n @ Sigma_n @ V.T @ W

print('T_21 of shape {0}, rank {1}, condition {2}'.format(T21.shape, np.linalg.matrix_rank(T21), np.linalg.cond(T21)))
print('T_22 of shape {0}, rank {1}, condition {2}\n'.format(T22.shape, np.linalg.matrix_rank(T22), np.linalg.cond(T22)))
print('S_11 of shape {0}, rank {1}, condition {2}'.format(S11.shape, np.linalg.matrix_rank(S11), np.linalg.cond(S11)))
print('S_21 of shape {0}, rank {1}, condition {2}'.format(S21.shape, np.linalg.matrix_rank(S21), np.linalg.cond(S21)))
print('S_22 of shape {0}, rank {1}, condition {2}\n'.format(S22.shape, np.linalg.matrix_rank(S22), np.linalg.cond(S22)))

# Just to check
display(Latex(r'$\left\| \bT_{{2,1}} - \bW_\perp^T \bV \bSigma^{{-2}} \bV^T \bW \right\|_F =$ {0}'.format(np.linalg.norm(T21 - T21_alt))))
display(Latex(r'$\left\| \bS_{{1,1}} - \bW^T \bV \bSigma^{{2}} \bV^T \bW \right\|_F =$ {0}'.format(np.linalg.norm(S11 - S11_alt))))
display(Latex(r'$\left\| \bS_{{2,1}} - \bW_\perp^T \bV \bSigma^{{2}} \bV^T \bW \right\|_F =$ {0}'.format(np.linalg.norm(S21 - S21_alt))))

P, Q, RT = sp.linalg.svd(W.T @ V @ Sigma_n)
P_p, Q_p, R_pT = sp.linalg.svd(W_p.T @ V @ Sigma_n_inv)

rm = min(m,n)
rk = min(K-m, n)

T22_pinv = P_p[:,:rk] @ np.diag(1.0/(Q_p*Q_p)) @ P_p[:,:rk].T
S11_pinv = P[:,:rm] @ np.diag(1/(Q*Q)) @ P[:,:rm].T

display(Latex(r'$\left \| \bS_{{1,1}}^\dagger - \bP_{{1:n}} \bQ^{{-2}} (\bP_{{1:n}})^T \right\|_F =$  {0}'.format(np.linalg.norm(S11_pinv - np.linalg.pinv(S11)))))
display(Latex(r'$\left \| \bT_{{2,2}}^\dagger - \bP_{{\perp,1:n}} \bQ_\perp^{{-2}} (\bP_{{\perp,1:n}})^T \right\|_F =$  {0}'.format(np.linalg.norm(T22_pinv - np.linalg.pinv(T22)))))

ra = min(K-m, n)
rm = min(m, n)

P, Q, RT = sp.linalg.svd(W.T @ V @ Sigma_n)
P_p, Q_p, R_pT = sp.linalg.svd(W_p.T @ V @ Sigma_n)
Ssolver_alt = P_p[:,:ra] @ np.diag(Q_p) @ R_pT[:ra] @ RT[:rm].T @ np.diag(1.0/Q) @ P[:,:rm].T
Ssolver = S21 @ np.linalg.pinv(S11)

print('(S21 * S11_inv) shape {0} condition {1}'.format(Ssolver.shape, np.linalg.cond(Ssolver)))
display(Latex(r'$\left\| \bS_{{2,1}} \bS_{{1,1}}^\dagger - (\bP_\perp)_{{1:n}} \bQ_\perp \bR_\perp^T \bR_W \bQ_W^{{-1}} (\bP_W)_{{1:n}}^T \right\|_F =${0}'.format(np.linalg.norm(Ssolver - Ssolver_alt))))

P, Q, RT = sp.linalg.svd(W.T @ V @ Sigma_n_inv)
P_p, Q_p, R_pT = sp.linalg.svd(W_p.T @ V @ Sigma_n_inv)
Tsolver_alt = P_p[:,:ra] @ np.diag(1.0/Q_p) @ R_pT[:ra] @ RT[:rm].T @ np.diag(Q) @ P[:,:rm].T
Tsolver = np.linalg.pinv(T22) @ T21

print('(T22_inv * T21) shape {0} condition {1}'.format(Tsolver.shape, np.linalg.cond(Tsolver)))
display(Latex(r'$\left\| \bT_{{2,2}}^\dagger \bT_{{2,1}} - (\bP_\perp)_{{1:n}} \bQ_\perp^{{-1}} \bR_\perp^T \bR_W \bQ_W (\bP_W)_{{1:n}}^T \right\|_F =${0}'.format(np.linalg.norm(Tsolver - Tsolver_alt))))

display(Latex(r'$\left\| \bS_{{2,1}} \bS_{{1,1}}^\dagger + \bT_{{2,2}}^\dagger \bT_{{2,1}} \right\|_F =$ {0}'.format(np.linalg.norm(Tsolver + Ssolver))))

print('That other thing Ive been semi hopeful for : {0}'.format(np.linalg.norm(Ssolver - Q.T @ P[:,:n].T)))

W2_p = Psi[:,m:2*m]

S11_approx = W.T @ V @ Sigma_n @ Sigma_n @ V.T @ W
S21_approx = W2_p.T @ V @ Sigma_n @ Sigma_n @ V.T @ W
print('S_11_approx of shape {0}, rank {1}, condition {2}'.format(S11_approx.shape, np.linalg.matrix_rank(S11_approx), np.linalg.cond(S11_approx)))
print('S_21_approx of shape {0}, rank {1}, condition {2}'.format(S21_approx.shape, np.linalg.matrix_rank(S21_approx), np.linalg.cond(S21_approx)))
Ssolver_approx = S21_approx @ np.linalg.pinv(S11_approx)

T22_approx = W2_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W2_p
T21_approx = W2_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W
print('T_21_approx of shape {0}, rank {1}, condition {2}'.format(T21_approx.shape, np.linalg.matrix_rank(T21_approx), np.linalg.cond(T21_approx)))
print('T_22_approx of shape {0}, rank {1}, condition {2}'.format(T22_approx.shape, np.linalg.matrix_rank(T22_approx), np.linalg.cond(T22_approx)))
Tsolver_approx = np.linalg.pinv(T22_approx) @ T21_approx

print('')
display(Latex(r'$\left\| \bS_{{2,1}} \bS_{{1,1}}^\dagger - \bS_{{2,1}}^{{(\mathrm{{app}})}} (\bS_{{1,1}}^{{(\mathrm{{app}})}})^\dagger \right\|_2 =$ {0}'.format(np.linalg.norm(W2_p @ Ssolver_approx - W_p @ Ssolver, ord=2))))
display(Latex(r'$\left\| \bT_{{2,2}}^\dagger \bT_{{2,1}} - (\bT_{{2,2}}^{{(\mathrm{{app}})}})^\dagger \bT_{{2,1}}^{{(\mathrm{{app}})}} \right\|_2 =$ {0}'.format(np.linalg.norm(W2_p @ Tsolver_approx - W_p @ Tsolver, ord=2))))

Ssolver_acc = np.zeros(K-m)
Tsolver_acc = np.zeros(K-m)

for M in range(m+1,K):
    
    W2_p = Psi[:,m:M]
    S11_approx = W.T @ V @ Sigma_n @ Sigma_n @ V.T @ W
    S21_approx = W2_p.T @ V @ Sigma_n @ Sigma_n @ V.T @ W
    T22_approx = W2_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W2_p
    T21_approx = W2_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W

    Ssolver_approx = S21_approx @ np.linalg.pinv(S11_approx)
    Tsolver_approx = np.linalg.pinv(T22_approx) @ T21_approx

    Ssolver_acc[M-(m+1)] = np.linalg.norm(W2_p @ Ssolver_approx - W_p @ Ssolver, ord=2)
    Tsolver_acc[M-(m+1)] = np.linalg.norm(W2_p @ Tsolver_approx - W_p @ Tsolver, ord=2)

plt.figure(figsize=(10, 7))
plt.plot(range(m+1, K+1), Ssolver_acc, label=r'$S_{2,1} S_{1,1}^{-1}$')
plt.plot(range(m+1, K+1), Tsolver_acc, label=r'$T_{2,2}^{-1} T_{2,1}$')
plt.legend(loc=1)
plt.xlabel(r'Dim of $W + W_\perp$')
plt.ylabel(r'$||$ Solver - Approx Solver $||_2$')
plt.title(r'Meas space $W$ of dim $m=${0}, PCA space of dim $n=${1}'.format(m,n))
plt.show()



