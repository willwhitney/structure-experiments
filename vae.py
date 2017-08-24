# taken from https://github.com/pytorch/examples/blob/master/vae/main.py

from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import pdb
import logging
from util import *
from setup import make_result_folder, write_options
# from modules import GaussianKLD

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--name', default='vae_default')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.save = 'results/' + args.name

make_result_folder(args, args.save)
write_options(args, args.save)

logging.basicConfig(filename = args.save + "/results.csv",
                    level = logging.DEBUG,
                    format = "%(message)s")
logging.debug(("step,train_loss,train_prior_loss,train_likelihood,"
               "test_loss,test_prior_loss,test_likelihood"))

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc41 = nn.Linear(400, 784)
        self.fc42 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc41(h3)), self.fc42(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

class GaussianKLD(nn.Module):
    def forward(self, q, p):
        (mu_q, log_sigma_q) = q
        (mu_p, log_sigma_p) = p
        mu_q = batch_flatten(mu_q)
        log_sigma_q = batch_flatten(log_sigma_q)
        mu_p = batch_flatten(mu_p)
        log_sigma_p = batch_flatten(log_sigma_p)

        # log_sigma_p = torch.log(sigma_p)
        # log_sigma_q = torch.log(sigma_q)
        sum_log_sigma_p = torch.sum(log_sigma_p, 1, keepdim=True)
        sum_log_sigma_q = torch.sum(log_sigma_q, 1, keepdim=True)
        a = sum_log_sigma_p - sum_log_sigma_q
        b = torch.sum(torch.exp(log_sigma_q - log_sigma_p), 1, keepdim=True)

        mu_diff = mu_p - mu_q
        c = torch.sum(
            torch.pow(mu_diff, 2) / torch.exp(log_sigma_p), 1, keepdim=True)

        D = mu_q.size(1)
        divergences = torch.mul(a + b + c - D, 0.5)
        return divergences.mean()

class GaussianLL(nn.Module):
    def forward(self, p, target):
        # print(p[0].size())
        # print(target.size())
        (mu, log_sigma) = p
        mu = batch_flatten(mu)
        log_sigma = batch_flatten(log_sigma)
        target = batch_flatten(target)

        # sigma = Variable(torch.ones(sigma.size()).type_as(sigma.data) / 10)

        a = torch.sum(log_sigma, 1, keepdim=True)
        diff = (target - mu)
        b = torch.sum(
            torch.pow(diff, 2) / torch.exp(log_sigma), 1, keepdim=True)
        c = mu.size(1) * math.log(2*math.pi)
        log_likelihoods = -0.5 * (a + b + c)
        return log_likelihoods.mean()

model = VAE()
if args.cuda:
    model.cuda()

gaussianKL = GaussianKLD()
gaussianLL = GaussianLL()
reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD, KLD, BCE

def gaussianKL_loss_function(xhat, x, mu, logvar):
    # -2.3 is log(0.1)
    # pdb.set_trace()
    target_logvar = Variable(torch.Tensor(x.size()).fill_(-2.3)).type_as(x)
    output_KLD = gaussianKL(xhat, (x, target_logvar))

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return output_KLD + KLD, KLD, output_KLD

def gaussianLL_loss_function(xhat, x, mu, logvar):
    # pdb.set_trace()
    output_NLL = - gaussianLL(xhat, x)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return output_NLL + KLD, KLD, output_NLL

loss_function = gaussianLL_loss_function

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss, train_prior_loss, train_likelihood = 0, 0, 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, prior, likelihood = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        train_prior_loss += prior.data[0]
        train_likelihood += likelihood.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    train_loss /= len(train_loader.dataset)
    train_prior_loss /= len(train_loader.dataset)
    train_likelihood /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))
    return train_loss, train_prior_loss, train_likelihood


def test(epoch):
    model.eval()
    test_loss, test_prior_loss, test_likelihood = 0, 0, 0
    for data, _ in test_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        loss, prior, likelihood = loss_function(recon_batch, data, mu, logvar)
        test_loss += loss.data[0]
        test_prior_loss += prior.data[0]
        test_likelihood += likelihood.data[0]


    test_loss /= len(test_loader.dataset)
    test_prior_loss /= len(test_loader.dataset)
    test_likelihood /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, test_prior_loss, test_likelihood


for epoch in range(1, args.epochs + 1):
    train_loss, train_prior_loss, train_likelihood = train(epoch)
    test_loss, test_prior_loss, test_likelihood = test(epoch)

    log_values = (epoch, train_loss, train_prior_loss, train_likelihood, test_loss, test_prior_loss, test_likelihood)
    format_string = ",".join(["{:.8e}"]*len(log_values))
    logging.debug(format_string.format(*log_values))

    z = Variable(torch.randn(args.batch_size, 20), volatile=True)
    if args.cuda:
        z = z.cuda()
    # pdb.set_trace()
    generations = model.decode(z)[0]
    generations = generations.resize(128, 28, 28)

    save_tensors_image("{}/generations{}.png".format(args.save, epoch),
                       [g.expand(3, 28, 28) for g in generations])
