import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import pdb

from util import *

class GaussianKLD(nn.Module):
    def forward(self, q, p):
        (mu_q, sigma_q) = q
        (mu_p, sigma_p) = p
        mu_q = batch_flatten(mu_q)
        sigma_q = batch_flatten(sigma_q)
        mu_p = batch_flatten(mu_p)
        sigma_p = batch_flatten(sigma_p)

        log_sigma_p = torch.log(sigma_p)
        log_sigma_q = torch.log(sigma_q)
        sum_log_sigma_p = torch.sum(log_sigma_p, 1, keepdim=True)
        sum_log_sigma_q = torch.sum(log_sigma_q, 1, keepdim=True)
        a = sum_log_sigma_p - sum_log_sigma_q
        b = torch.sum(sigma_q / sigma_p, 1, keepdim=True)

        mu_diff = mu_p - mu_q
        c = torch.sum(torch.pow(mu_diff, 2) / sigma_p, 1, keepdim=True)

        D = mu_q.size(1)
        divergences = torch.mul(a + b + c - D, 0.5)
        return divergences.mean()

class FixedGaussianKLD(nn.Module):
    def forward(self, q, p):
        (mu_q, sigma_q) = q
        (mu_p, sigma_p) = p
        mu_q = batch_flatten(mu_q)
        sigma_q = batch_flatten(sigma_q)
        sigma_q2 = torch.pow(sigma_q, 2)
        mu_p = batch_flatten(mu_p)
        sigma_p = batch_flatten(sigma_p)
        sigma_p2 = torch.pow(sigma_p, 2)

        # a = torch.dot(1 / sigma_p, sigma_q)
        a = torch.sum((1 / sigma_p2) * sigma_q2, 1, keepdim=True)
        
        diff = mu_p - mu_q
        # b = torch.dot(diff, (1 / sigma_p) * diff)
        b = torch.sum(diff * ((1 / sigma_p2) * diff), 1, keepdim=True)

        c = - mu_q.size(1)

        # d = torch.log(torch.prod(sigma_p) / torch.prod(sigma_q))
        # d = torch.log(torch.prod(sigma_p, 1, keepdim=True) / 
                      # torch.prod(sigma_q, 1, keepdim=True))
        d = torch.log(torch.prod(sigma_p2, 1, keepdim=True)) - \
            torch.log(torch.prod(sigma_q2, 1, keepdim=True))

        divergences = 0.5 * (a + b + c + d)
        return divergences.mean()

class LogSquaredGaussianKLD(nn.Module):
    def forward(self, q, p):
        (mu_q, log_sigma_q2) = q
        (mu_p, log_sigma_p2) = p
        mu_q = batch_flatten(mu_q)
        log_sigma_q2 = batch_flatten(log_sigma_q2)
        mu_p = batch_flatten(mu_p)
        log_sigma_p2 = batch_flatten(log_sigma_p2)

        # a = torch.dot(1 / sigma_p, sigma_q)
        # a = torch.sum((1 / sigma_p2) * sigma_q2, 1, keepdim=True)
        a = torch.sum(torch.exp(log_sigma_q2 - log_sigma_p2), 1, keepdim=True)
        
        diff = mu_p - mu_q
        # b = torch.dot(diff, (1 / sigma_p) * diff)
        # b = torch.sum(diff * ((1 / sigma_p2) * diff), 1, keepdim=True)
        b = torch.sum(diff * diff * torch.exp(- log_sigma_p2), 1, keepdim=True)

        c = - mu_q.size(1)

        # d = torch.log(torch.prod(sigma_p) / torch.prod(sigma_q))
        # d = torch.log(torch.prod(sigma_p, 1, keepdim=True) / 
                      # torch.prod(sigma_q, 1, keepdim=True))
        d = torch.log(torch.exp(torch.sum(log_sigma_p2, 1, keepdim=True))) - \
            torch.log(torch.exp(torch.sum(log_sigma_q2, 1, keepdim=True)))

        divergences = 0.5 * (a + b + c + d)
        if math.isnan(divergences.data.sum()):
            pdb.set_trace()

        # pdb.set_trace()
        return divergences.mean()

class GaussianLL(nn.Module):
    def forward(self, p, target):
        # print(p[0].size())
        # print(target.size())
        (mu, sigma) = p
        mu = batch_flatten(mu)
        sigma = batch_flatten(sigma)
        target = batch_flatten(target)

        # sigma = Variable(torch.ones(sigma.size()).type_as(sigma.data) / 10)

        a = torch.sum(torch.log(sigma), 1, keepdim=True)
        diff = (target - mu)
        b = torch.sum(torch.pow(diff, 2) / sigma, 1, keepdim=True)
        c = mu.size(1) * math.log(2*math.pi)
        log_likelihoods = -0.5 * (a + b + c)
        if math.isnan(log_likelihoods.data.sum()):
            pdb.set_trace()
        return log_likelihoods.mean()

class MotionGaussianLL(nn.Module):
    def forward(self, p, target, mask):
        # print(p[0].size())
        # print(target.size())
        (mu, sigma) = p
        mu = batch_flatten(mu)
        sigma = batch_flatten(sigma)
        target = batch_flatten(target)
        # show(mask[0].data.cpu() / 4)
        mask = batch_flatten(mask)
        #
        # sigma = Variable(torch.ones(sigma.size()).type_as(sigma.data) / 10)
        #
        a = torch.sum(torch.log(sigma), 1, keepdim=True)
        diff = (target - mu) * mask
        b = torch.sum(torch.pow(diff, 2) / sigma, 1, keepdim=True)
        c = mu.size(1) * math.log(2*math.pi)
        log_likelihoods = -0.5 * (a + b + c)
        return log_likelihoods.mean()

class LogSquaredGaussianLL(nn.Module):
    def forward(self, p, target):
        # print(p[0].size())
        # print(target.size())
        (mu, log_sigma2) = p
        mu = batch_flatten(mu)
        log_sigma2 = batch_flatten(log_sigma2)
        target = batch_flatten(target)

        a = torch.sum(0.5 * log_sigma2, 1, keepdim=True)
        diff = (target - mu)
        b = torch.sum(
            torch.pow(diff, 2) * torch.exp(-0.5 * log_sigma2), 1, keepdim=True)

        c = mu.size(1) * math.log(2*math.pi)
        log_likelihoods = -0.5 * (a + b + c)
        return log_likelihoods.mean()