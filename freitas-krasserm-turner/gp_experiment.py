# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 23:07:09 2021

@author: tanni
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

""" This is code for simple GP regression. It assumes a zero mean GP Prior """


# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()
# f = lambda x: (0.25*(x**2)).flatten()

# Define the kernel
def kernel(a, b):
    """
    GP squared exponential kernel 
    Assumes an isotropic Gaussian
    """
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T) # Eq. to distance.cdist('sqeuclidean')
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

def rq_kernel(a, b):
    pass

N = 10         # number of training points.
n = 50         # number of test points.
s = 0.00005    # noise variance.

# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + s*np.random.randn(N)

# Mesons
# X = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, -1],
#      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
#      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, -1],
#      [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1],
#      [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1],
#      [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, -1],
#      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1],
#      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, -1],
#      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1],
#      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, -1],
#      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, -1],
#      [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0.5, 0, -1],
#      [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, -1],
#      [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.5, 0, -1],
#      [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0.5, 0, -1],
#      [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0.5, 0, -1],
#      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 1, -1],
#      [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0.5, 1, -1],
#      [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.5, 1, -1],
#      [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.5, 1, -1]])

# y = np.array([139.57039, 139.57039, 134.9768, 547.862, 957.78, 775.4, 775.4, 775.49, 782.65, 1019.461,
#      493.677, 493.677, 497.611, 497.611, 497.614, 497.614, 891.66, 891.66, 895.55, 895.55])

#K
K = kernel(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
#K* = kernel(X, Xtest);
#L*   = cholskey (K*)
#L* = np.linalg.cholesky(K*)
#mu* = 0 + K*T  X  K^(-1) X y
#mu* = 0 + K*' * inv(K)  *  y
Lk = np.linalg.solve(L, kernel(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
#K** , sigma*
K_ = kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)

# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])

# draw samples from the prior at our test points.
#L** from K** or sigma*
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,5)))
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.plot(X, y, 'r+', ms=20)
pl.title('Five samples from the GP prior')
pl.axis([-5, 5, -3, 3])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.
# sigma* ------   K** - K*T K(-1) K* .. ----- L_new  = cholsky(sigma*)
#sigma* = 
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
#L_new
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,5)))
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.plot(X, y, 'r+', ms=20)
pl.title('Five samples from the GP posterior')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')

pl.show()