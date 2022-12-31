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

gamma = 1.2

x_mean = torch.zeros(D)
x_mean[0] = alpha

x_samples = multivariate_normal.MultivariateNormal(x_mean, torch.eye(D)).sample([R])

# exact target posterior p(alpha|x_samples)

mu = torch.mean(x_samples[:, 0])
sigma = 1 / R**0.5

target = normal.Normal(mu, sigma)

# score function of p(alpha|x_samples)

score_p = lambda alpha_: jacobian(target.log_prob, alpha_)[0]

sequence_len = 50

# score function for each individual likelihood contribution


def make_score_p(alpha_, r):
    def log_p_i(alpha_):
        return -(alpha_**2 - 2 * alpha_ * x_samples[r, 0]) / 2

    def score_p_(alpha_):
        return jacobian(log_p_i, alpha_)[0]

    return score_p_(alpha_)


c = 1
beta = -1 / 2

KSD = imq_ksd(c, beta)


n_list = [int(10**r) for r in torch.arange(0.5, 3.25, 0.25)]
n_no = len(n_list)

wass_dist = np.zeros(n_no)

for i in range(n_no):

    n = n_list[i]

    if i == 0:
        sequence = normal.Normal(gamma * mu, sigma).sample([n, 1])
    else:
        sequence = torch.cat(
            [sequence, normal.Normal(gamma * mu, sigma).sample([n - n_list[i - 1], 1])]
        )

    sequence.requires_grad_()

    # generate independent sample from target to approximate Wasserstein-1 distance
    target_sample = target.sample([n]).detach().numpy()
    sequence_np = sequence.detach().numpy().reshape(n)

    wass_dist[i] = wasserstein_distance(sequence_np, target_sample)


start_time = time.time()
IMQ_list = KSD.ksd(sequence, score_p, n_list=n_list)
print([IMQ_list, time.time() - start_time])


start_time = time.time()
Stoch_IMQ_01_list = KSD.stoch_ksd(
    sequence, make_score_p, round(0.1 * R), R, n_list=n_list
)
print([Stoch_IMQ_01_list, time.time() - start_time])


start_time = time.time()
Stoch_IMQ_02_list = KSD.stoch_ksd(
    sequence, make_score_p, round(0.2 * R), R, n_list=n_list
)
print([Stoch_IMQ_02_list, time.time() - start_time])


start_time = time.time()
RF_L1_list = []
for j in range(20):
    L1_SD = torch.stack(KSD.L1_RF_sd(sequence, score_p, M=100, n_list=n_list))
    RF_L1_list.append(L1_SD)


sns.set_style("whitegrid")

plt.plot(n_list, wass_dist, "-bo", label="Wasserstein-1")

RF_L1_list = torch.stack(RF_L1_list)
RF_L1_medians = torch.median(RF_L1_list, axis=0).values.detach().numpy()

plt.plot(n_list, RF_L1_medians, "-c.", label="L1IMQ")
plt.plot(n_list, IMQ_list, "-r^", label="IMQ KSD")
plt.plot(n_list, Stoch_IMQ_01_list, "--g^", label="StochKSD: 0.1R")
plt.plot(n_list, Stoch_IMQ_02_list, "--m^", label="StochKSD: 0.2R")

plt.ylim(1e-3, 1e3)

plt.xlabel("Number of samples $n$")
plt.ylabel("Discrepancy value")
plt.yscale("log")
plt.xscale("log")

plt.legend()
plt.title("$\gamma$ = {}".format(gamma))
# plt.savefig('1D_gamma_3.png', dpi = 1200)
plt.show()
