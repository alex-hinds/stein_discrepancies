import torch
from torch.autograd.functional import hessian, jacobian
from scipy.special import gammaln
import math
import numpy as np
import time

class multivariate_t_dist:  # multivariate t distribution with isotropic scale matrix 
    
    # scale parameter is a scalar such that the scale matrix is scale * torch.eye(p)
    
    def __init__(self, loc, scale, df):
        self.loc = loc
        self.scale = scale
        self.df = df
        
        self.p = len(loc)
        p = self.p
        
        self.Gaussian = torch.distributions.MultivariateNormal(torch.zeros(p), scale * torch.eye(p))
        self.chi2 = torch.distributions.chi2.Chi2(df)
    
    def log_density(self, x):
        p = self.p 
        A1 = gammaln((self.df + p)/2) - gammaln(self.df/2) - (p/2) * (np.log(self.df) + np.log(math.pi) + np.log(self.scale))
        A2 = torch.log(1 + 1/(self.df * self.scale) * torch.norm(x - self.loc)**2)
        
        return A1 - (self.df + p)/2 * A2
        
    def density(self, x):
        return torch.exp(self.log_density(x))
    
    def sample(self, M):

        # returns M samples from multivariate t distribution
            
        a1 = self.Gaussian.sample([M])
            
        a2 = self.chi2.sample([M])

        return torch.stack([a1[j] / torch.sqrt(a2[j]/self.df) + self.loc for j in range(M)])
        

class imq_ksd:
        
    def __init__(self, c, beta):
        self.c = c
        self.beta = beta
        assert c > 0 and beta > -1 and beta < 0
        

    def kernel(self, x, y):
        c = self.c
        beta = self.beta
        return (c**2 + ((x-y)**2).sum())**beta
    
    def grad_x(self, x, y):
        return jacobian(self.kernel, (x,y))[0]
    
    def grad_y(self, x, y):
        return jacobian(self.kernel, (x,y))[1]

    def grad_xy(self, x, y):    
        return torch.trace(hessian(self.kernel, (x,y))[0][1])
    
        
    # IMQ KSD (q_samples is an n x d matrix of n d-dimensional samples)
    
    def k_p(self, x, y, score_p):
        kernel = self.kernel
        grad_x = self.grad_x
        grad_y = self.grad_y
        grad_xy = self.grad_xy
        
        t1 = torch.dot(score_p(x), score_p(y)) * kernel(x,y)
        t2 = torch.dot(score_p(x), grad_y(x,y))
        t3 = torch.dot(score_p(y), grad_x(x,y))
        t4 = grad_xy(x,y)
        
        return t1 + t2 + t3 + t4
             
    def ksd(self, q_samples, score_p, n_list = None): 
        
        # if n_list is specified, returns list of KSDs for the first n samples in q_samples, for each n in n_list
        
        if n_list is None:
            
            n_list = [len(q_samples)]
            
        n_max = max(n_list)
        out = []
        A = torch.empty(n_max,n_max)
        
        start_time = time.time()
        for i in range(n_max):
            
            for j in range(n_max):
                A[i,j] = self.k_p(q_samples[i,:], q_samples[j,:], score_p)
                
        
        for n_ in n_list:
            out.append(torch.mean(A[:n_,:n_]) ** (1/2))                
            
        return out if len(out) >= 2 else out[0]
 
    # Stochastic IMQ KSD
    
         # R is number of likelihood terms
         # batch_size is size of randomly sampled subsets
        
         # make_score_p is a factory taking inputs (z,r) and returning the 
         # score for the r^th likelihood term evaluated at z  (r = 1,...,R)

    def stoch_k_p(self, x, y, make_score_p, perm_1, perm_2, R):
        kernel = self.kernel
        grad_x = self.grad_x
        grad_y = self.grad_y
        grad_xy = self.grad_xy
        
        m = len(perm_1)            # batch size
        
        score_x = 0; score_y = 0
        
        for j in perm_1:
            score_x += make_score_p(x,j)
        
        for k in perm_2:
            score_y += make_score_p(y,k)
        
        t1 = (R/m)**2 * torch.dot(score_x, score_y) * kernel(x,y)
        t2 = (R/m) * torch.dot(score_x, grad_y(x,y))
        t3 = (R/m) * torch.dot(score_y, grad_x(x,y))
        t4 = grad_xy(x,y)
        
        return t1 + t2 + t3 + t4
        
    def stoch_ksd(self, q_samples, make_score_p, batch_size, R, n_list = None):
        
        if n_list is None:   
            
            n_list = [len(q_samples)]
            
        n_max = max(n_list)
        out = []
        A = torch.empty(n_max,n_max)

        sigma = []
        start_time = time.time()

        
        for i in range(n_max):
            sigma.append(torch.randperm(R)[:batch_size])
 
        for i in range(n_max):     
            for j in range(n_max):
                A[i,j] = self.stoch_k_p(q_samples[i,:], q_samples[j,:], make_score_p, sigma[i], sigma[j], R)
                            
                
        for n_ in n_list:
            out.append(torch.mean(A[:n_,:n_]) ** (1/2)) 
                
        return out if len(out) >= 2 else out[0]

    # Random Feature L1IMQ SD (not a KSD)

    def L1_RF_sd(self, q_samples, score_p, M, gamma=.25, target_df=3, n_list = None):
        
        if n_list is None:
            
            n_list = [len(q_samples)]
        
        d = q_samples.shape[1]
                
        alpha = gamma / 3
        
        xi = 4 * alpha / (2 + alpha)
        xi_min = d / (d + target_df) * xi
        
        lambda_max = 1 - alpha / 2
        
        beta_prime = -d / (2 * xi_min)
        c_prime = lambda_max * self.c / 2.

        nu_df = -2 * xi * beta_prime - d
        nu_scale = c_prime**2 / nu_df
        
        def stein_operator_d(x, y, j, score_p_):  # Stein operator on j^th dimension
            return score_p_(x)[j] * self.kernel(x, y) + self.grad_x(x, y)[j]
                    
        m_n = torch.mean(q_samples, axis = 0)
        importance_sampler = multivariate_t_dist(m_n, nu_scale, nu_df)
        
        
        Z_vec = importance_sampler.sample(M)
        
        out = []
        n_max = max(n_list)        
        
        A = torch.empty(n_max, d, M)
        
        for k in range(n_max):
            for j in range(d):
                for m in range(M):
                    A[k,j,m] = stein_operator_d(q_samples[k,:], Z_vec[m], j, score_p)
            
                    
        for n_ in n_list:
            
            dim_vec = []
            
            for j in range(d):
            
                feat_vec = []
            
                for m in range(M):
                    B1 = torch.mean(torch.tensor(A[:n_,j,m]))
                    B2 = (importance_sampler.density(Z_vec[m]) ** -1) * abs(B1)
                    feat_vec.append(B2)
            
                feat_vec = torch.stack(feat_vec)
                dim_vec.append((torch.mean(feat_vec))**2)
        
            dim_vec = torch.stack(dim_vec)
            out.append(torch.sum(dim_vec))
        
        return out if len(out) >= 2 else out[0]
    
    

#if __name__ == "__main__":
        
    


    

