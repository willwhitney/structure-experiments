# taken from https://github.com/pytorch/examples/blob/master/vae/main.py

from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import pdb
import math
import logging
from util import *
from setup import make_result_folder, write_options
from moving_mnist.dataset import MovingMNIST
# from modules import GaussianKLD

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--name', default='vae_default')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--hidden-dim', type=int, default=20, metavar='N',
                    help='size of the hidden dim (default: 20)')
parser.add_argument('--mlp', action="store_true",
                    help='use mlp instead of tinydcgan')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
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
logging.debug(("step,loss,divergence,prior divergence,nll,"
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

train_loader = torch.utils.data.DataLoader(
    MovingMNIST(train=True, 
                seq_len=1,
                image_size=28,
                colored=False),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    MovingMNIST(train=False, 
                seq_len=1,
                image_size=28,
                colored=False),
    batch_size=args.batch_size, shuffle=True, **kwargs)



activation = F.relu
dtype = torch.cuda.FloatTensor
class TinyDCGANFirstInference(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(TinyDCGANFirstInference, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        ndf = 64
        self.planes = [ndf,
                       ndf * 2,
                       ndf * 4,
                       ndf * 4]
        self.kernels = [3, 3, 3, 3]
        self.strides = [2, 2, 2, 1]
        if input_dims[-1] >= 16:
            self.pads = [1, 1, 0, 0]
        else:
            self.pads = [1, 1, 1, 1]

        self.out_dims = [input_dims]
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[-1]
            self.out_dims.append(tuple(conv_out_dim(in_planes,
                                               self.planes[l],
                                               in_height,
                                               in_width,
                                               self.kernels[l],
                                               padding=self.pads[l],
                                               stride=self.strides[l])))
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[l]
            self.convs.append(nn.Conv2d(in_planes,
                                        int(self.planes[l]),
                                        self.kernels[l],
                                        padding=self.pads[l],
                                        stride=self.strides[l]))
            if l < len(self.planes) - 1:
                self.bns.append(nn.BatchNorm2d(int(self.planes[l])))

        print(self.out_dims)
        # self.conv1 = nn.Conv2d(input_dims[0], 32, 3)
        self.input_lin = nn.Linear(prod(self.out_dims[-1]), hidden_dim)
        # self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
        #                              for _ in range(2)])


        self.lin_mu = nn.Linear(hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(hidden_dim, self.hidden_dim)


    def forward(self, x_t):
        current = x_t
        for i, conv in enumerate(self.convs):
            current = conv(current)
            if i < len(self.convs) - 1:
                current = self.bns[i](current)
            current = activation(current)
        current = current.resize(current.size(0), prod(self.out_dims[-1]))
        current = self.input_lin(current)
        new_hidden = activation(current)

        # joined = torch.cat([current, z_prev], 1)
        # new_hidden = F.tanh(self.joint_lin(joined))

        # mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        mu = self.lin_mu(new_hidden)
        sigma = self.lin_sigma(new_hidden)
        return (mu, sigma)

class TinyDCGANInference(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(TinyDCGANInference, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        ndf = 64
        self.planes = [ndf,
                       ndf * 2,
                       ndf * 4,
                       ndf * 4]
        self.kernels = [3, 3, 3, 3]
        self.strides = [2, 2, 2, 1]
        if input_dims[-1] >= 16:
            self.pads = [1, 1, 0, 0]
        else:
            self.pads = [1, 1, 1, 1]

        self.out_dims = [input_dims]
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[-1]
            self.out_dims.append(tuple(conv_out_dim(
                in_planes,
                self.planes[l],
                in_height,
                in_width,
                self.kernels[l],
                padding=self.pads[l],
                stride=self.strides[l])))
        self.convs = nn.ModuleList()
        for l in range(len(self.planes)):
            # confusingly this is actually offset by 1
            in_planes, in_height, in_width = self.out_dims[l]

            self.convs.append(nn.Conv2d(in_planes,
                                        self.planes[l],
                                        self.kernels[l],
                                        padding=self.pads[l],
                                        stride=self.strides[l]))

        # self.conv1 = nn.Conv2d(input_dims[0], 32, 3)

        self.input_lin = nn.Linear(prod(self.out_dims[-1]), hidden_dim)
        self.joint_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
        #                              for _ in range(2)])


        self.lin_mu = nn.Linear(hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(hidden_dim, self.hidden_dim)


    def forward(self, x_t, z_prev):
        current = x_t
        for conv in self.convs:
            current = conv(current)
            current = activation(current)
        current = current.resize(current.size(0), prod(self.out_dims[-1]))
        current = self.input_lin(current)
        current = activation(current)

        joined = torch.cat([current, z_prev], 1)
        new_hidden = activation(self.joint_lin(joined))

        # mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        mu = self.lin_mu(new_hidden)
        sigma = self.lin_sigma(new_hidden)
        return (mu, sigma)

class TinyDCGANGenerator(nn.Module):
    def __init__(self, hidden_dim, output_dims):
        super(TinyDCGANGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims

        # self.planes[0] will be set automatically so that it is
        # similar to but larger than hidden_dim
        # self.planes = [None, 64, 64, 128, output_dims[0] * 2]
        # self.kernels = [None, 3, 3, 3, 3]

        # DCGAN numbers
        ngf = 64
        self.planes = [None,
                       ngf * 4,
                       ngf * 2,
                       ngf,
                       output_dims[0] * 2]
        self.kernels = [None, 3, 3, 3, 3]
        if output_dims[-1] >= 16:
            self.pads = [None, 0, 1, 1, 1]
        else:
            self.pads = [None, 1, 1, 1, 1]
        self.strides = [None, 1, 1, 1, 1]


        self.in_dims = [output_dims[1:]]
        # self.in_dims = [(7,7)]
        for l in range(len(self.planes) - 1, 0, -1):
            in_dim = conv_transpose_in_dim(*self.in_dims[0],
                                 self.kernels[l],
                                 stride=self.strides[l],
                                 padding=self.pads[l])
            # in_dim = list(int(d / 2) for d in in_dim)
            self.in_dims = [in_dim] + self.in_dims

        self.planes[0] = math.ceil(hidden_dim / prod(self.in_dims[0]))
        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, self.planes[0] * prod(self.in_dims[0]))])
        # import pdb; pdb.set_trace()

        # print("Generator: ", self.in_dims)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(1, len(self.planes)):
            in_height, in_width = self.in_dims[l]
            self.convs.append(nn.ConvTranspose2d(self.planes[l-1],
                                        self.planes[l],
                                        self.kernels[l],
                                        stride=self.strides[l],
                                        padding=self.pads[l]))
            if l < len(self.planes) - 1:
                # pdb.set_trace()
                self.bns.append(nn.BatchNorm2d(self.planes[l]))

        self.convs[-1].bias.data.add_(0.5)

    def forward(self, input):
        current = input
        for lin in self.lins:
            current = lin(current)
            current = activation(current)

        current = current.resize(current.size(0),
                                 self.planes[0],
                                 *self.in_dims[0])
        for i, conv in enumerate(self.convs):
            current = activation(current)

            # current = self.upsample(current)
            current = conv(current)
            if i < len(self.convs) - 1:
                current = self.bns[i](current)

        mu = current[:, : int(current.size(1) / 2)]
        sigma = Variable(torch.ones(mu.size()).type(dtype) * 0.5)
        return (mu, sigma)

if args.mlp:
    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()

            self.fc1 = nn.Linear(784, 400)
            self.fc21 = nn.Linear(400, args.hidden_dim)
            self.fc22 = nn.Linear(400, args.hidden_dim)
            self.fc3 = nn.Linear(args.hidden_dim, 400)
            self.fc41 = nn.Linear(400, 784)
            self.fc42 = nn.Linear(400, 784)

            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def encode(self, x):
            h1 = self.relu(self.fc1(x))
            # return self.fc21(h1), self.sigmoid(self.fc22(h1)) * 4
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
            # z = self.reparametrize(mu, logvar)
            z = sample_log2((mu, logvar))
            xhat = self.decode(z)
            return xhat, mu, logvar
else:
    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()

            self.encoder = TinyDCGANFirstInference([1, 28, 28], 
                                                   args.hidden_dim)
            self.decoder = TinyDCGANGenerator(args.hidden_dim, [1, 28, 28])

            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def encode(self, x):
            mu, sigma = self.encoder(x)
            return mu, sigma

        def reparametrize(self, mu, logvar):
            std = logvar.mul(0.5).exp_()
            if args.cuda:
                eps = torch.cuda.FloatTensor(std.size()).normal_()
            else:
                eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps)
            return eps.mul(std).add_(mu)

        def decode(self, z):
            mu, sigma = self.decoder(z)
            return self.sigmoid(mu), sigma

        def forward(self, x):
            mu, logvar = self.encode(x)
            # z = self.reparametrize(mu, logvar)
            z = sample_log2((mu, logvar))
            xhat = self.decode(z)
            return xhat, mu, logvar



from loss_modules import LogSquaredGaussianKLD, GaussianLL, LogSquaredGaussianLL

model = VAE()
if args.cuda:
    model.cuda()

# gaussianKL = GaussianKLD()
# squaredgaussianKL = SquaredGaussianKLD()
# fixedgaussianKL = FixedGaussianKLD()
logsquaredgaussianKL = LogSquaredGaussianKLD()

gaussianLL = GaussianLL()
logsquaredLL = LogSquaredGaussianLL()
# othergaussianLL = OtherGaussianLL()
# originalgaussianLL = OriginalGaussianLL()
# logsquaredLL = LogSquaredGaussianLL()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

def original_loss_function(xhat, x, mu, logvar):
    BCE = reconstruction_function(xhat[0], x)

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
    target_logvar = Variable(torch.Tensor(x.size()).fill_(-5.298317367)).type_as(x)
    output_KLD = gaussianKL(xhat, (x, target_logvar))

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return output_KLD + KLD, KLD, output_KLD

def gaussianLL_loss_function(xhat, x, mu, logvar):
    # 0.005
    # log_outvar = Variable(torch.Tensor(xhat[0].size()).fill_(-5.298317367), 
                          # requires_grad=False).type_as(x)
    # 0.05
    # log_outvar = Variable(torch.Tensor(xhat[0].size()).fill_(-3), 
    #                       requires_grad=False).type_as(x)
    # 0.2
    # log_outvar = Variable(torch.Tensor(xhat[0].size()).fill_(-1.609437912), 
    #                       requires_grad=False).type_as(x)
    # 0.5
    log_outvar = Variable(torch.Tensor(xhat[0].size()).fill_(-.693147181), 
                          requires_grad=False).type_as(x)
    outvar = Variable(torch.Tensor(xhat[0].size()).fill_(0.5), 
                      requires_grad=False).type_as(x)
    # pdb.set_trace()
    log_squared_outvar = Variable(
        torch.Tensor(xhat[0].size()).fill_(math.log(0.1**2)), 
        requires_grad=False).type_as(x)

    # output_NLL = - logsquaredLL(xhat, x)

    # pdb.set_trace()
    output_NLL = reconstruction_function(xhat[0], x) / args.batch_size

    # output_NLL = - logsquaredLL((xhat[0], log_squared_outvar), x)
    # output_NLL = - gaussianLL((xhat[0], outvar), x)

    # output_NLL = - gaussianLL((xhat[0], log_outvar), x)
    # output_NLL = - gaussianLL(xhat, x)
    # output_NLL = - othergaussianLL((xhat[0], outvar), x)
    # output_NLL = - originalgaussianLL((xhat[0], outvar), x)
    # output_NLL = - motiongaussianLL(
    #     (xhat[0], outvar), x,
    #     Variable(torch.ones(x.size())).type_as(x))
    # output_NLL = - othergaussianLL(xhat, x)

    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_element).mul_(-0.5) / args.batch_size

    zeros = Variable(torch.zeros(mu.size())).type_as(mu)
    zeros2 = Variable(torch.zeros(mu.size())).type_as(mu)
    ones = Variable(torch.ones(mu.size())).type_as(mu)
    # KLD = gaussianKL((mu, torch.exp(logvar)), (zeros, ones))
    # KLD = squaredgaussianKL((mu, torch.exp(logvar)), (zeros, ones))
    # KLD = fixedgaussianKL((mu, torch.exp(0.5 * logvar)), (zeros, ones))
    KLD = logsquaredgaussianKL((mu, logvar), (zeros, zeros2))
    # KLD = gaussianKL((mu, logvar), (zeros, ones))

    return output_NLL + KLD, KLD, output_NLL
    # return output_NLL, KLD, output_NLL

# loss_function = gaussianKL_loss_function
loss_function = gaussianLL_loss_function

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# for i in range(10):
#     q = (Variable(torch.randn(50, 20)),
#          Variable((torch.rand(50, 20) - 0.5) * 10))
#     KLD_element = q[0].pow(2).add_(q[1].exp()).mul_(-1).add_(1).add_(q[1])
#     KLD = torch.sum(KLD_element).mul_(-0.5)

#     zeros = Variable(torch.zeros(50, 20))
#     ones = Variable(torch.ones(50, 20))
#     my_KLD = fixedgaussianKL((q[0], torch.exp(0.5 * q[1])), (zeros, ones))
#     print(KLD.data[0] - my_KLD.data[0], KLD.data[0], my_KLD.data[0])



def train(epoch):
    model.train()
    train_loss, train_prior_loss, train_likelihood = 0, 0, 0
    # for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, data in enumerate(train_loader):
        data.transpose_(0, 1)
        data.unsqueeze_(2)
        data = Variable(data[0])
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
                loss.data[0]))

    train_loss /= (len(train_loader.dataset) / args.batch_size)
    train_prior_loss /= (len(train_loader.dataset) / args.batch_size)
    train_likelihood /= (len(train_loader.dataset) / args.batch_size)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))
    return train_loss, train_prior_loss, train_likelihood


def test(epoch):
    model.eval()
    test_loss, test_prior_loss, test_likelihood = 0, 0, 0
    # for data, _ in test_loader:
    for data in test_loader:
        data.transpose_(0, 1)
        data.unsqueeze_(2)
        if args.cuda:
            data = data.cuda()
        data = Variable(data[0], volatile=True)
        recon_batch, mu, logvar = model(data)
        loss, prior, likelihood = loss_function(recon_batch, data, mu, logvar)
        test_loss += loss.data[0]
        test_prior_loss += prior.data[0]
        test_likelihood += likelihood.data[0]


    test_loss /= (len(test_loader.dataset) / args.batch_size)
    test_prior_loss /= (len(test_loader.dataset) / args.batch_size)
    test_likelihood /= (len(test_loader.dataset) / args.batch_size)

    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, test_prior_loss, test_likelihood


for epoch in range(1, args.epochs + 1):
    train_loss, train_prior_loss, train_likelihood = train(epoch)
    test_loss, test_prior_loss, test_likelihood = test(epoch)

    log_values = (epoch, train_loss, train_prior_loss, train_prior_loss, train_likelihood, test_loss, test_prior_loss, test_likelihood)
    format_string = ",".join(["{:.8e}"]*len(log_values))
    logging.debug(format_string.format(*log_values))

    z = Variable(torch.randn(args.batch_size, args.hidden_dim), volatile=True)
    if args.cuda:
        z = z.cuda()
    # pdb.set_trace()
    generations = model.decode(z)[0]
    generations = generations.resize(generations.size(0), 28, 28)

    for batch_idx, data in enumerate(train_loader):
        data.transpose_(0, 1)
        data.unsqueeze_(2)
        data = Variable(data[0], volatile=True)
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        recon_batch = recon_batch[0].data
        recon_batch.resize_(recon_batch.size(0), 28, 28)
        save_tensors_image("{}/recons{}.png".format(args.save, epoch),
                       [g.expand(3, 28, 28) for g in recon_batch])  
        break      

    save_tensors_image("{}/generations{}.png".format(args.save, epoch),
                       [g.expand(3, 28, 28) for g in generations])
