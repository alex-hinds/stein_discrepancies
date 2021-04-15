import torch

from GMMposterior import GMM_posterior
from SVGD import SVGD
from KSDs import imq_ksd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time


n_steps = 20  # number of SVGD steps taken for each iteration
n_iter = 12  # number of iterations

theta_1 = 0
theta_2 = 1
R = 100

GMM_post = GMM_posterior(theta_1, theta_2, R)

score_p = GMM_post.posterior_score
log_posterior_density = GMM_post.log_posterior_density

c = 1
beta = - 1/2

KSD = imq_ksd(c,beta)
GMM_SVGD = SVGD(c,beta)

KSD_list = []


n_particles = 30  # number of particles

# generate random initial particles from N([1,1],I)

init_samples = torch.distributions.MultivariateNormal(torch.ones(2), torch.eye(2)).sample([n_particles])
theta_new = init_samples.requires_grad_()

for iter in range(n_iter+1):

    start_time = time.time()    
    
    if iter % 4 == 0:  # plot at 0 steps and then every 4 * n_steps after
        
        sns.set_style('whitegrid')
        x = np.linspace(-3., 3., 80)
        y = np.linspace(-3.,3., 80)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = np.array([log_posterior_density(torch.tensor(xx, dtype = torch.float32)).detach().numpy() for xx in XX])
        Z = Z.reshape(X.shape)

        # levels for contours
        Z_max = np.max(Z)
        min_max_gap = 0.01 * abs(Z_max)
        levels = np.linspace(Z_max - min_max_gap, Z_max, 6)
        
        # plot contours and particles
        plt.contour(X,Y,Z, levels = levels)
        plt.scatter(theta_new[:,0].detach().numpy(), theta_new[:,1].detach().numpy(), marker = 'x', c='black')
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_2$')
        plt.savefig('GMM_contour_{}.png'.format(iter+1), dpi = 1200)
        plt.title("{} steps".format(iter * n_steps))
        plt.show()
        
    if iter == n_iter:
        break

    KSD_list.append(KSD.ksd(theta_new, score_p))
        
    theta_old = theta_new.clone()
    
    theta_new = GMM_SVGD.update(theta_old, score_p, n_steps)
    
    print('iter {} out of {} done, time = {}'.format(iter + 1, n_iter, time.time() - start_time))



# plot the KSD

steps_list = torch.arange(0, n_iter * n_steps, n_steps)

plt.plot(steps_list, KSD_list, '-ro')
plt.ylim(bottom = 0)
plt.xlabel('Number of steps')
plt.ylabel('IMQ KSD')    
plt.savefig('GMM_SVGD_KSDs.png', dpi = 1200)
plt.show()




