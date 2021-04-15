import torch
from torch.autograd.functional import jacobian


class SVGD():
    
    def __init__(self, c, beta):
        self.c = c
        self.beta = beta
        assert c > 0 and beta < 0 and beta > -1
    
    def kernel(self, x, y):
        c = self.c
        beta = self.beta
        return (c**2 + ((x-y)**2).sum())**beta
    
    def grad_x(self, x, y):
        return jacobian(self.kernel, (x,y))[0]
    
    
    # optimal descent direction phi for both algorithms
    
    def imq_phi(self, x, y, score_p):
        return self.kernel(x, y) * score_p(x) + self.grad_x(x, y)

    def stoch_imq_phi(self, x, y, make_score_p, perm, R):
        m = len(perm)
        
        score_x = 0
        
        for j in perm:
            score_x += make_score_p(x, j)
        
        return (R/m) * self.kernel(x,y) * score_x + self.grad_x(x,y)
    
    # update
    def update(self, init_samples, score_p, n_iter, lr = 1e-2):
        
        # standard SVGD update with constant learning rate lr
        # init_samples is a n x d matrix of n d-dimensional initial particles
        
        phi = self.imq_phi
        
        x = init_samples.requires_grad_()
        n = len(init_samples)

        for iter in range(n_iter):
            x_old = torch.clone(x)
            
            for i in range(n):
                phi_sum = 0
                
                for j in range(n):
                    phi_sum = phi_sum + phi(x_old[j,:], x_old[i,:], score_p)
                    
                x[i,:] = x_old[i,:] + lr * phi_sum/n
            
        return x
    
    def stoch_update(self, init_samples, n_iter, make_score_p, batch_size, R, lr = 1e-2):
        # stochastic SVGD update with constant learning rate lr
        phi = self.stoch_imq_phi

        x = init_samples.requires_grad_()
        n = len(init_samples)

        for iter in range(n_iter):
            x_old = torch.clone(x)
                        
            for i in range(n):
                phi_sum = 0
                sigma = torch.randperm(R)[:batch_size]

                for j in range(n):
                    phi_sum = phi_sum + phi(x_old[j,:], x_old[i,:], make_score_p, sigma, R)
                    
                x[i,:] = x_old[i,:] + lr * phi_sum/n

        return x

