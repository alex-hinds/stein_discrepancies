import torch
from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt
import seaborn as sns

from KSDs import imq_ksd

from torch.autograd.functional import jacobian
from torch.distributions import normal, multivariate_normal

import numpy as np
import time

# 

torch.manual_seed(0)

# First generate R D-dimensional data points from p(x|alpha) when alpha = 1

alpha = 1
D = 2
R = 10

gamma = 1.5

x_mean = torch.zeros(D)
x_mean[0] = alpha

x_samples = multivariate_normal.MultivariateNormal(x_mean, torch.eye(D)).sample([R])

# exact target posterior p(alpha|x_samples)

mu = torch.mean(x_samples[:,0])
sigma = 1/R**0.5

target = normal.Normal(mu, sigma)

# score function of p(alpha|x_samples)

score_p = lambda alpha_: jacobian(target.log_prob, alpha_)

sequence_len = 50

# log of target density (up to additive constant)

log_p = lambda alpha_ : - (alpha_ - mu) ** 2 * R / 2

# score function

score_p = lambda alpha_ : jacobian(log_p, alpha_)[0]

# score function for each individual likelihood contribution

def make_score_p(alpha_, r):  
    def log_p_i(alpha_):
        return - (alpha_**2 - 2 * alpha_ * x_samples[r,0]) / 2
    def score_p_(alpha_):
        return jacobian(log_p_i, alpha_)[0]
    return score_p_(alpha_)


c = 1
beta = -1/2

KSD = imq_ksd(c,beta)


n_list = [int(10 ** r) for r in torch.arange(0.5,3,0.25)]
n_no = len(n_list)

# initialise
    
IMQ = np.zeros(n_no)
Stoch_IMQ_01 = np.zeros(n_no)
Stoch_IMQ_05 = np.zeros(n_no)
wass_dist = np.zeros(n_no)

for i in range(n_no):
    
    n = n_list[i]
            
    sequence = normal.Normal(gamma * mu, sigma).sample([n, 1])
    
    sequence.requires_grad_()
     
    # start_time = time.time()
    IMQ[i] = KSD.ksd(sequence, score_p_1)
    # print(IMQ[i], (time.time() - start_time))

    # start_time = time.time()
    Stoch_IMQ_01[i] = KSD.stoch_ksd(sequence, make_score_p, round(0.1 * R), R)
    # print(Stoch_IMQ_01[i], (time.time() - start_time))
    
    # start_time = time.time()
    Stoch_IMQ_05[i] = KSD.stoch_ksd(sequence, make_score_p, round(0.5 * R), R)
    # print(Stoch_IMQ_05[i], (time.time() - start_time))

    # generate independent sample from target to approximate Wasserstein-1 distance
    # start_time = time.time()
    target_sample = target.sample([n]).detach().numpy()
    sequence_np = sequence.detach().numpy().reshape(n)
    
    wass_dist[i] = wasserstein_distance(sequence_np, target_sample)
    # print(wass_dist[i], (time.time() - start_time))
    

sns.set_style('whitegrid')
plt.plot(n_list, wass_dist, '-bo', label = 'Wasserstein-1')
plt.plot(n_list, IMQ, '-r^', label = 'IMQ KSD')
plt.plot(n_list, Stoch_IMQ_01, '--g^', label = 'StochKSD: 0.1R')
plt.plot(n_list, Stoch_IMQ_05, '--m^', label = 'StochKSD: 0.5R')
plt.ylim(1e-3,1e2)
plt.xlabel('Number of samples $n$')
plt.ylabel('Discrepancy value')
plt.yscale("log")
plt.xscale("log") 
plt.legend()
plt.title("$\gamma$ = {}".format(gamma))
# plt.savefig('1D_gamma_4.png', dpi = 1200)
plt.show()  
