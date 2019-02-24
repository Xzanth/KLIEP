#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.set_printoptions(linewidth=175)

nnu = 100
nde = 100

xnu = np.random.normal(1, 0.125, nnu)
xde = np.random.normal(1, 0.5, nde)

b = min(100, nnu)
xb = np.random.permutation(xnu)[:b]

def kernel(x, xprime, sigma):
    return np.exp(-((np.linalg.norm(x - xprime)**2)/(2 * sigma**2)))

def psi(x, bs, sigma):
    return [kernel(x, xprime, sigma) for xprime in bs]

def constraint(theta, nu, mean_de):
    mean_de_T = np.transpose(mean_de)
    de2 = np.dot(mean_de_T, mean_de)
    theta += np.divide(np.dot((1 - np.dot(mean_de_T, theta)), mean_de), de2)
    theta = np.amax([theta, np.zeros(len(theta))], axis=0)
    theta = np.divide(theta, np.dot(mean_de_T, theta))
    return theta

def grad_asc(nu, mean_de, epsilon, max_iter):
    old_score = 0
    theta = np.ones(len(mean_de))
    theta = constraint(theta, nu, mean_de)
    old_theta = theta.copy()
    for e in epsilon:
        for i in range(max_iter):
            inv_nt = np.divide(1, np.dot(nu, theta))
            theta += np.dot(e, np.dot(np.transpose(nu), inv_nt))
            theta = constraint(theta, nu, mean_de)
            score = np.mean(np.log(np.dot(nu, theta)), axis=0)
            if score <= old_score:
                theta = old_theta.copy()
                score = old_score
                break
            old_score = score
            old_theta = theta.copy()
    return (theta, score)

sigma = 0.1

psi_nu = [psi(x, xb, sigma) for x in xnu]
psi_de = [psi(x, xb, sigma) for x in xde]

m_psi_de = np.mean(psi_de, axis=0)

epsilons = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
max_iter = 100

theta, score = grad_asc(psi_nu, m_psi_de, epsilons, max_iter)

print(score)

what = np.dot(psi_de, theta)


# TAKEN CODE BELOW
xdisp = np.linspace(-0.5, 3, 100)
p_nu_xdisp = norm.pdf(xdisp, 1, 0.125)
p_de_xdisp = norm.pdf(xdisp, 1, 0.5)
w_xdisp = p_nu_xdisp / p_de_xdisp
x_re = [psi(x, xb, sigma) for x in xdisp]
wh_xdisp = np.dot(x_re, theta)
plt.figure(1)
plt.plot(xdisp, p_de_xdisp, 'b-')
plt.plot(xdisp, p_nu_xdisp, 'k-')
plt.xlabel('x')
plt.figure(2)
plt.plot(xdisp, w_xdisp, 'r-')
plt.plot(xdisp, wh_xdisp, 'g-')
plt.plot(xde, what, 'bo')
plt.show()
