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
        
        batch_no = int(R / batch_size)
        
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
    
    n_iter = 400  # number of SGLD iterations through whole dataset
    
    samples_list = []  # list of samples for each epsilon
    
    KSD_list = []
    Stoch_KSD_list01 = []
    Stoch_KSD_list001 = []
    RF_L1_SD_list = []
    
    itr = 0
        
    for epsilon in epsilon_list:
        
        torch.manual_seed(0)
                
        # generate initial sample from N([1,1],I)

        init_sample = torch.distributions.MultivariateNormal(torch.ones(2), torch.eye(2)).sample()
        init_sample.requires_grad_()
        
        samples = opt.update(init_sample, batch_size, epsilon, n_iter, burn_in = 2000)
        samples_list.append(samples)

        # separate out samples to reduce correlation
        
        samples_red = torch.stack([samples[j] for j in np.arange(0, len(samples), 40)])
        
        KSD_list.append(KSD.ksd(samples_red, score_p))
        
        list_01 = []
        for jj in range(5):
            list_01.append(KSD.stoch_ksd(samples_red, make_score_p, int(0.1 * R), R))
            
        list_01 = torch.stack(list_01)
        Stoch_KSD_list01.append(torch.mean(list_01))
        print(Stoch_KSD_list01)
        
        list_001 = []
        for kk in range(5):
            list_001.append(KSD.stoch_ksd(samples_red, make_score_p, int(0.01 * R), R))
        
        list_001 = torch.stack(list_001)
        Stoch_KSD_list001.append(KSD.stoch_ksd(samples_red, make_score_p, int(0.01 * R), R))
        print(Stoch_KSD_list001)
        
        
        RF_iter_list = []
        
        for j in range(5):
            L1_SD = KSD.L1_RF_sd(samples_red, score_p, M = 10)
            RF_iter_list.append(L1_SD)
            
        RF_iter_tensor = torch.stack(RF_iter_list)
        RF_L1_SD_list.append(torch.median(RF_iter_tensor, axis = 0).values.detach())
                
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
        min_max_gap = 0.006 * abs(Z_max)
        levels = np.linspace(Z_max - min_max_gap, Z_max, 5)
            
        plt.contour(X,Y,Z, levels = levels)
        
        C = samples.detach().numpy()
        plt.scatter(C[:,0],C[:,1], marker = 'x', color = 'black')
    
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_2$') 

        itr += 1
        #plt.savefig('SGLD_GMM_{}.png'.format(itr), dpi = 1200)
        plt.title("$\epsilon = ${}".format(epsilon))
        plt.show()

    
    KSD_list = torch.stack(KSD_list).detach()
    Stoch_KSD_list01 = torch.stack(Stoch_KSD_list01).detach()
    Stoch_KSD_list001 = torch.stack(Stoch_KSD_list001).detach()
    RF_L1_SD_list = torch.stack(RF_L1_SD_list).detach()
    
    plt.plot(epsilon_list, KSD_list, '-bo', label='IMQ KSD')
    plt.plot(epsilon_list, Stoch_KSD_list01, '--m^', label = 'StochKSD: 0.1R')
    plt.plot(epsilon_list, Stoch_KSD_list001, '--g^', label = 'StochKSD: 0.01R')
    plt.plot(epsilon_list, RF_L1_SD_list, '-c.', label = 'L1IMQ')
    plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(bottom = 1e-6, top = 1e-1)
    plt.legend()
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('KSDs')    
    #plt.savefig('GMM_SGLD_KSDs_ALT.png', dpi = 1200)
    plt.show()

    
    
    
    


