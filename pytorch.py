import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import scipy.stats
import env
import math
from PIL import Image

import socket
# %matplotlib inline

if socket.gethostname() == 'zaan':
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

eps = 1e-2
class Transition(nn.Module):
    def __init__(self, hidden_dim):
        super(Transition, self).__init__()
        self.dim = hidden_dim
        self.l1 = nn.Linear(self.dim, self.dim)
        self.lin_mu = nn.Linear(self.dim, self.dim)
        self.lin_sigma = nn.Linear(self.dim, self.dim)


    def forward(self, input):
        hidden = F.tanh(self.l1(input))
        mu = F.tanh(self.lin_mu(hidden))
        # sigma = Variable(torch.ones(mu.size()).type(dtype) / 2)
        sigma = F.sigmoid(self.lin_sigma(hidden)) + eps
        # print(sigma.mean().data[0])
        return (mu, sigma)

class Generator(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim)
                                     for _ in range(2)])
        self.lin_mu = nn.Linear(self.hidden_dim, self.output_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        current = input
        for layer in self.layers:
            current = layer(current)
            current = F.relu(current)

        mu = F.sigmoid(self.lin_mu(current))
        sigma = F.sigmoid(self.lin_sigma(current)) + eps
        return (mu, sigma)

class Inference(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Inference, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.joint_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                     for _ in range(2)])

        self.lin_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, x_t, z_prev):
        embedded = F.tanh(self.input_lin(x_t))
        joined = torch.cat([embedded, z_prev], 1)
        new_hidden = F.tanh(self.joint_lin(joined))
        for layer in self.layers:
            new_hidden = layer(new_hidden)
            new_hidden = F.tanh(new_hidden)

        mu = F.tanh(self.lin_mu(new_hidden))
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)

class FirstInference(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FirstInference, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                     for _ in range(2)])

        self.lin_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, x_t):
        new_hidden = F.tanh(self.input_lin(x_t))
        for layer in self.layers:
            new_hidden = layer(new_hidden)
            new_hidden = F.tanh(new_hidden)

        mu = F.tanh(self.lin_mu(new_hidden))
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)

# TODO: test this
class GaussianKLD(nn.Module):
    def forward(self, q, p):
        (mu_q, sigma_q) = q
        (mu_p, sigma_p) = p

        a = torch.sum(torch.log(sigma_p), 1) - torch.sum(torch.log(sigma_q), 1)
        b = torch.sum(sigma_q / sigma_p, 1)

        mu_diff = mu_p - mu_q
        c = torch.sum(torch.pow(mu_diff, 2) / sigma_p, 1)

        D = mu_q.size(1)
        divergences = torch.mul(a + b + c - D, 0.5)
        return divergences.mean()

# class GaussianKLD1(nn.Module):
#     def forward(self, q, p):
#         (mu_q, sigma_q) = q
#         (mu_p, sigma_p) = p
#         a1 = torch.sum(torch.pow(sigma_p, -1) * sigma_q, 1)
#         b1 = torch.sum((mu_p - mu_q) * torch.pow(sigma_p, -1) * (mu_p - mu_q), 1)
#         c1 = - mu_q.size(1)
#         d1 = torch.log(torch.prod(sigma_p, 1) / torch.prod(sigma_q, 1))
#         return 0.5 * (a1 + b1 + c1 + d1)

class GaussianLL(nn.Module):
    def forward(self, p, target):
        (mu, sigma) = p

        a = torch.sum(torch.log(sigma), 1)
        diff = (target - mu)
        b = torch.sum(torch.pow(diff, 2) / sigma, 1)
        c = mu.size(1) * math.log(2*math.pi)
        log_likelihoods = -0.5 * (a + b + c)
        return log_likelihoods.mean()

def sample(p):
    (mu, sigma) = p
    noise = torch.normal(torch.zeros(mu.size()), torch.ones(sigma.size())).type(dtype)
    noise = Variable(noise)
    return mu + sigma * noise

gen = env.DataGenerator()
batch_size = 32
data_dim = gen.start().render().nelement()
hidden_dim = 200
T = 3

KL = GaussianKLD().type(dtype)
# KL1 = GaussianKLD1().type(dtype)
LL = GaussianLL().type(dtype)
mse = nn.MSELoss().type(dtype)
l1 = nn.L1Loss().type(dtype)

def mv_logpdf(p, target):
    return scipy.stats.multivariate_normal.logpdf(target.data.numpy(),
                                                  mean=p[0].data[0].numpy(),
                                                  cov=p[1].data[0].numpy())

def test_KL(q, p):
    n_samples = 10000
    results = torch.zeros(n_samples)
    for i in range(n_samples):
        thing = sample(q)
        log_q_sample = mv_logpdf(q, thing)
        log_p_sample = mv_logpdf(p, thing)
        results[i] = log_q_sample - log_p_sample

    return results.mean()


z1_prior = (Variable(torch.zeros(batch_size, hidden_dim).type(dtype)),
            Variable(torch.ones(batch_size, hidden_dim).type(dtype)))


class WholeModel(nn.Module):
    def __init__(self):
        super(WholeModel, self).__init__()
        self.z1_prior = (
            Variable(torch.zeros(batch_size, hidden_dim).type(dtype)),
            Variable(torch.ones(batch_size, hidden_dim).type(dtype))
        )

        self.transition = Transition(hidden_dim)
        self.first_inference = FirstInference(data_dim, hidden_dim)
        self.inference = Inference(data_dim, hidden_dim)
        self.generator = Generator(hidden_dim, data_dim)

    def forward(self, sequence):
        loss = Variable(torch.zeros(1).type(dtype))
        generations = []

        inferred_z_post = self.first_inference(sequence[0])
        z_prior = z1_prior
        for t in range(len(sequence)):
            divergence = KL(inferred_z_post, z_prior)
            loss = loss + divergence

            z_sample = sample(inferred_z_post)
            # z_sample = inferred_z_post[0]

            gen_dist = self.generator(z_sample)
            # log_likelihood = - mse(gen_dist[0], sequence[t])
            log_likelihood = LL(gen_dist, sequence[t])
            loss = loss - log_likelihood

            generations.append(gen_dist)

            if t < len(sequence) - 1:
                z_prior = self.transition(z_sample)
                inferred_z_post = self.inference(sequence[t+1], z_sample)
                # inferred_z_post = self.first_inference(sequence[t+1])

        return generations, loss / len(sequence)


def make_real_seq(length):
    sequence = [torch.zeros(batch_size, gen.size**2 * 3) for _ in range(length)]
    for batch in range(batch_size):
        sequence[0][batch] = gen.start().render().view(gen.size**2 * 3)
        for i in range(1, length):
            sequence[i][batch] = gen.step().render().view(gen.size**2 * 3)

    return [Variable(x.type(dtype)) for x in sequence]

def make_seq(length, dim):
    sequence = [torch.zeros(batch_size, dim).normal_(0.5, 0.1)]
    for i in range(1, length):
        noise = torch.zeros(batch_size, dim) # torch.normal([0.0, 0.0], [0.1, 0.1])
        sequence.append(sequence[i-1] + noise)
    sequence = [Variable(x.type(dtype)) for x in sequence]
    return sequence


model = WholeModel().type(dtype)
# params = list(model.parameters())
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3)

print(model)

mean_loss = 0
n_steps = 10000
for i in range(n_steps):
    # sequence = make_seq(1, data_dim)
    sequence = make_real_seq(T)

    generations, loss = model(sequence)
    mean_loss += loss.data[0]

    model.zero_grad()
    loss.backward()

    optimizer.step()

    k = 10
    if i % k == 0:
        print("Step: ", i, "\tLoss: ", mean_loss / k)
        mean_loss = 0

    if i % 1000 == 0 or i == n_steps - 1:
        gen_data = [(gen[0].data, gen[1].data) for gen in generations]

        seq_data = [x.data for x in sequence]

        # %matplotlib inline
        # env.show(seq_data[0].view(4,4,3))
        for j in range(5):
            timesteps = len(seq_data)
            result = torch.zeros(2 * gen.size, timesteps * gen.size, 3)
            for t in range(timesteps):
                mu, sigma = gen_data[t]
                mu = mu[j]
                # import ipdb; ipdb.set_trace()
                result[:gen.size, gen.size*t:gen.size*(t+1)] = seq_data[t][j].view(gen.size,gen.size,3)
                result[gen.size:, gen.size*t:gen.size*(t+1)] = mu.view(gen.size,gen.size,3)
                # scipy.misc.imsave(str(j) + '_input.png', seq_data[t][j].view(4,4,3).numpy())
                # scipy.misc.imsave(str(j) + '_output.png', mu.view(4,4,3).numpy())
                # env.show(mu.view(4,4,3))
            scipy.misc.imsave(str(j) + '_result.png', result.numpy())

# params = Variable(torch.rand(1,5), requires_grad=True)
# x = Variable(torch.rand(5, 1))
# y_hat = torch.mm(params, x)

# y = Variable(torch.Tensor([1e4]))

# loss = (y_hat - y) ** 2
# loss.backward()
# print(y_hat.grad.data)
# print(params.grad.data)


# class Dummy(nn.Module):
#     def __init__(self):
#         super(Dummy, self).__init__()
#         self.l1 = nn.Linear(2,2)
#         self.l2 = nn.Linear(2,2)
#
#     def forward(self, input):
#         h = F.tanh(self.l1(input))
#         return F.tanh(self.l2(h))
#
#
# model = Dummy()
# input = Variable(torch.rand(5, 2))
#
# output = model(input)
# loss = mse(output, input)
#
# loss.backward()

# other = (Variable(torch.zeros(batch_size, hidden_dim) + 1.3),
#          Variable(torch.ones(batch_size, hidden_dim) * 2))
#
# print(KL(z1_prior, other)) # , KL1(z1_prior, other))
# print(test_KL(z1_prior, other))


# p = (Variable(torch.zeros(batch_size, data_dim)),
#      Variable(torch.ones(batch_size, data_dim)))
# target = Variable(torch.ones(batch_size, data_dim))
# for i in range(100):
#     p = (Variable(torch.rand(batch_size, data_dim)),
#          Variable(torch.rand(batch_size, data_dim)))
#     target = Variable(torch.rand(batch_size, data_dim))
#     print((LL(p, target).data[0] - mv_logpdf(p, target)) / mv_logpdf(p, target))
