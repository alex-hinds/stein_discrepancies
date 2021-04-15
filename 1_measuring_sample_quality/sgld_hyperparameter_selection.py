'''
Recreates Section 5.1 of https://proceedings.neurips.cc/paper/2020/file/d03a857a23b5285736c4d55e0bb067c8-Paper.pdf
'''

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from KSDs import imq_ksd
from GMMposterior import GMM_posterior

import time

class SGLD():
    
    def __init__(self, R, prior_score, make_lik_grad):
        # prior_score = score function of the prior
        # make_lik_grad = factory for the scores of the likelihood terms
        # R = number of likelihood terms
        
        self.R = R
        self.prior_score = prior_score
        self.make_lik_grad = make_lik_grad
      
    def update(self, init_sample, batch_size, epsilon, n_iter, burn_in = 0):
        # init_sample = starting point for algorithm
        # batch_size = size of subsets sampled
        # epsilon = constant step size
        # n_iter = number of iterations over entire dataset
        # burn_in = number of burn-in steps
        
        batch_no = int(n_iter / batch_size)
        
        D = len(init_sample)
        
        noise_dist = torch.distributions.MultivariateNormal(torch.zeros(D), epsilon * torch.eye(D))
                
        x = init_sample
        
        x_list = []  # list of generated samples
        
        for n in range(n_iter):
            
            perm = torch.randperm(R)
                        
            for i in range(batch_no):
                                
                x.requires_grad_()
                
                batch = perm[batch_size * i : batch_size * (i+1)]
                
                lik_score = 0
                
                for j in batch:
                    lik_score += self.make_lik_grad(x, j)
                    
                pri_score = self.prior_score(x)
                
                x.detach() 
                
                x = x + 0.5 * epsilon * (pri_score + R/batch_size * lik_score)
                
                noise = noise_dist.sample()
                
                x += noise
                
                x_list.append(x)
                        
        return torch.stack(x_list[burn_in:])
        
    
'''
SGLD for the GMM Posterior with a range of learning rates epsilon
'''

if __name__ == "__main__":
    
    theta_1 = 0
    theta_2 = 1 
    
    R = 100
    
    GMM_post = GMM_posterior(theta_1, theta_2, R) 
    
    log_posterior_density = GMM_post.log_posterior_density     
    score_p = GMM_post.posterior_score
    make_score_p = GMM_post.make_posterior_score    
    
    
    opt = SGLD(R, GMM_post.prior_score, GMM_post.make_lik_grad)
    
    batch_size = int(0.1 * R)
    
    c = 1
    beta = -1/2
    
    KSD = imq_ksd(c, beta)        

    epsilon_list = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    
    n_iter = 200  # number of SGLD iterations through whole dataset
    
    samples_list = []  # list of samples for each epsilon
    KSD_list = []
    Stoch_KSD_list01 = []
    Stoch_KSD_list001 = []
        
    for epsilon in epsilon_list:
        
        torch.manual_seed(0)
                
        # generate initial sample from N([1,1],I)

        init_sample = torch.distributions.MultivariateNormal(torch.ones(2), torch.eye(2)).sample()
        init_sample.requires_grad_()
        
        samples = opt.update(init_sample, batch_size, epsilon, n_iter, burn_in = 2000)
        samples_list.append(samples)

        
        KSD_list.append(KSD.ksd(samples, score_p))
        Stoch_KSD_list01.append(KSD.stoch_ksd(samples, make_score_p, int(0.1 * R), R))
        # Stoch_KSD_list001.append(KSD.stoch_ksd(samples, make_score_p, int(0.01 * R), R))

        
        # plot samples and contours
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
            
        plt.contour(X,Y,Z, levels = levels)
        
        C = samples.detach().numpy()
        plt.scatter(C[:,0],C[:,1], marker = 'x', color = 'black')
    
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_2$') 

        # plt.savefig('SGLD_GMM_{}.png'.format(), dpi = 1200)
        plt.title("$\epsilon = ${}".format(epsilon))
        plt.show()

    
    KSD_list = torch.stack(KSD_list).detach()
    Stoch_KSD_list01 = torch.stack(Stoch_KSD_list01).detach()
    Stoch_KSD_list001 = torch.stack(Stoch_KSD_list001).detach()
    
    plt.scatter(epsilon_list, KSD_list, marker = 'o', label='IMQ KSD')
    plt.scatter(epsilon_list, Stoch_KSD_list01, marker = '^', label = 'StochKSD: 0.1R')
    # plt.scatter(epsilon_list, Stoch_KSD_list001, marker = '^', label = 'StochKSD: 0.01R')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(bottom = 1e-6, top = 1e-1)
    plt.legend()
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('KSDs')    
    # plt.savefig('GMM_SGLD_KSDs.png', dpi = 1200)
    plt.show()
