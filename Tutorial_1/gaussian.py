import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import random


def gaussian(x, mean, variance):
    const = 1/(((2*math.pi)**0.5)*sigma)
    y = math.e**((-0.5*(sigma**2))*((x - mean)**2))
    return y*const

def likelihood(x, mean):
    p = 1
    for i in x:
        p *= gaussian(i, mean, 1)
    return p

mean = 5
variance = 1
sigma = math.sqrt(variance)
const = 1/(((2*math.pi)**0.5)*sigma)
x = random.normal(mean, sigma, 10)
y = [gaussian(i, mean, sigma) for i in x]
plt.plot(x, y, 'bo')
plt.title("Distribution")
plt.savefig("Gaussian_dist")
plt.show()

means = [0]*11
likelihoods = [0]*11
for i in range(0, 11):
    likelihoods[i] = likelihood(x, i)
    means[i] = i

plt.plot(means, likelihoods, 'red')
plt.xlabel("Mean")
plt.ylabel("Likelihood")
plt.title("Likelihood curve")
plt.savefig("likelihood_curve")
plt.show()

log_likelihood = [math.log(i) for i in likelihoods]
plt.plot(means, log_likelihood)
plt.xlabel("Mean")
plt.ylabel("Log Likelihood")
plt.title("Log Likelihood Curve")
plt.savefig("log_likelihood")
plt.show()
