"""
Prior is p(theta1, theta2) ~ N(0, diag(sigma2_1, sigma2_2))

R i.i.d. likelihoods x_r ~ 0.5 * N(theta1, sigma2_x) + 0.5 * N(theta1 + theta2, sigma2_x)

As used in http://www.columbia.edu/~jwp2128/Teaching/E9801/papers/WellingTeh2011.pdf
"""

import torch
import math

from torch.autograd.functional import jacobian


class GMM_posterior:
    def __init__(self, theta_1, theta_2, R, sigma2_1=1.0, sigma2_2=10.0, sigma2_x=2.0):

        # theta_1 and theta_2 specify the "true" values

        torch.manual_seed(0)

        # generate the R data points x_r

        mix_inds = torch.rand(R)
        x = []

        mix_1 = torch.distributions.normal.Normal(theta_1, sigma2_x**0.5)
        mix_2 = torch.distributions.normal.Normal(theta_1 + theta_2, sigma2_x**0.5)

        for r in range(R):
            if mix_inds[r] < 0.5:
                x.append(mix_1.sample())
            else:
                x.append(mix_2.sample())

        self.R_samples = x  # R samples x_r
        self.sigma2_1 = sigma2_1
        self.sigma2_2 = sigma2_2
        self.R = R
        self.sigma2_x = sigma2_x
        self.theta_1 = theta_1
        self.theta_2 = theta_2

    def prior_score(self, theta_vec):
        sigma2_1 = self.sigma2_1
        sigma2_2 = self.sigma2_2

        sigmas = torch.tensor([sigma2_1, sigma2_2])

        prior_dist = torch.distributions.MultivariateNormal(
            torch.zeros([2]), torch.diag(sigmas)
        )

        return jacobian(prior_dist.log_prob, theta_vec)

    def make_GMM_density(self, theta_vector, r):
        # returns gradient of log of r^th likelihood term
        assert r % 1 == 0 and r >= 0 and r <= self.R - 1

        x = self.R_samples[r]
        sigma2_x = self.sigma2_x

        def GMM_density(theta_vec):
            mixture_1 = (
                1
                / (2 * math.pi * sigma2_x) ** 0.5
                * torch.exp(-((x - theta_vec[0]) ** 2) / (2 * sigma2_x))
            )
            mixture_2 = (
                1
                / (2 * math.pi * sigma2_x) ** 0.5
                * torch.exp(-((x - theta_vec[0] - theta_vec[1]) ** 2) / (2 * sigma2_x))
            )

            return 0.5 * mixture_1 + 0.5 * mixture_2

        return GMM_density(theta_vector)

    def make_lik_grad(self, theta_vector, r):
        def log_density(theta_vec):
            return torch.log(self.make_GMM_density(theta_vec, r))

        return jacobian(log_density, theta_vector)

    def log_posterior_density(self, theta_vec):

        sigma2_1 = self.sigma2_1
        sigma2_2 = self.sigma2_2

        sigmas = torch.tensor([sigma2_1, sigma2_2])

        log_prior = torch.distributions.MultivariateNormal(
            torch.zeros([2]), torch.diag(sigmas)
        ).log_prob(theta_vec)

        lik_list = [
            torch.log(self.make_GMM_density(theta_vec, r)) for r in range(self.R)
        ]
        log_lik_sum = 0

        for r in range(self.R):
            log_lik_sum += lik_list[r]

        return log_prior + log_lik_sum

    def make_posterior_score(
        self, theta_vec, r
    ):  # decompose score into r terms (for Stoch KSD)
        return 1 / self.R * self.prior_score(theta_vec) + self.make_lik_grad(
            theta_vec, r
        )

    def posterior_score(self, theta_vec):
        return self.prior_score(theta_vec) + sum(
            [self.make_lik_grad(theta_vec, r) for r in range(self.R)]
        )

    def stoch_posterior_score(self, theta_vec, batch):
        # returns prior_score plus R/batch_size * sum of batch of scores from likelihood
        batch_size = len(batch)

        lik_score_sum = sum([self.make_lik_grad(theta_vec, i) for i in batch])

        return self.prior_score(theta_vec) + self.R / batch_size * lik_score_sum
