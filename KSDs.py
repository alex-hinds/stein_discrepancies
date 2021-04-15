import torch
from torch.autograd.functional import hessian, jacobian

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
             
    def ksd(self, q_samples, score_p):       
        n = len(q_samples)
        A = torch.empty(n,n)
        
        for i in range(n):
            for j in range(n):
                A[i,j] = self.k_p(q_samples[i,:], q_samples[j,:], score_p)
        
        return torch.mean(A)**(1/2)       
 
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
        
    def stoch_ksd(self, q_samples, make_score_p, batch_size, R):
                
        n = len(q_samples)   # number of samples
        A = torch.empty(n,n)
                 
        sigma = []
        
        for i in range(n):
            sigma.append(torch.randperm(R)[:batch_size])   # n different random subsets of {1,...,R} of size batch_size
            
        for i in range(n):     
            for j in range(n):
                A[i,j] = self.stoch_k_p(q_samples[i,:], q_samples[j,:], make_score_p, sigma[i], sigma[j], R)

        return torch.mean(A)**(1/2)        

#if __name__ == "__main__":
        
    

