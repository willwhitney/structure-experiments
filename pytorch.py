import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


import env
import math
# %matplotlib inline

# gen = env.DataGenerator()
# gen.start()
# env.show(gen.render())

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
        sigma = F.tanh(self.lin_sigma(hidden))
        return (mu, sigma)

class Generator(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = [nn.Linear(self.hidden_dim, self.hidden_dim)
                       for _ in range(2)]
        self.lin_mu = nn.Linear(self.hidden_dim, self.output_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        current = input
        for layer in self.layers:
            current = layer(current)
            current = F.relu(current)

        mu = F.sigmoid(self.lin_mu(current))
        sigma = F.sigmoid(self.lin_sigma(current))
        return (mu, sigma)

class Inference(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Inference, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.joint_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim)
                       for _ in range(1)]
        
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
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + 1e-3
        return (mu, sigma)

class FirstInference(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FirstInference, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim)
                       for _ in range(1)]
        
        self.lin_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, x_t):
        new_hidden = F.tanh(self.input_lin(x_t))
        for layer in self.layers:
            new_hidden = layer(new_hidden)
            new_hidden = F.tanh(new_hidden)
        
        mu = F.tanh(self.lin_mu(new_hidden))
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + 1e-3
        return (mu, sigma)

# TODO: test this
class GaussianKLD(nn.Module):
    def forward(self, q, p):
        (mu_q, sigma_q) = q
        (mu_p, sigma_p) = p
        
        a = torch.log(torch.prod(sigma_p, 1) / torch.prod(sigma_q, 1))
        b = torch.sum(sigma_q / sigma_p, 1)
        
        mu_diff = mu_p - mu_q
        c = torch.sum(torch.pow(mu_diff, 2) / sigma_p, 1)

        D = mu_q.size(1)
        result = torch.mul(a + b + c - D, 0.5)
        # result = 0.5 * 
        return result

# TODO: test this
class GaussianLL(nn.Module):
    def forward(self, p, target):
        (mu, sigma) = p
        a = torch.log(torch.prod(sigma, 1))
        diff = target - mu
        b = torch.sum(torch.pow(diff, 2) / sigma, 1)
        c = mu.size(1) * math.log(math.pi)

        return -0.5 * (a + b + c)

def sample(p):
    (mu, sigma) = p
    noise = torch.normal(torch.zeros(mu.size()), torch.ones(sigma.size()))
    noise = Variable(noise)
    return mu + sigma * noise

batch_size = 1
data_dim = 2
hidden_dim = 10

KL = GaussianKLD()
LL = GaussianLL()


# model = [
#     transition,
#     first_inference,
#     inference,
#     generator,
# ]

z1_prior = (Variable(torch.zeros(batch_size, hidden_dim)),
         Variable(torch.ones(batch_size, hidden_dim)))

class WholeModel(nn.Module):
    def __init__(self):
        super(WholeModel, self).__init__()
        self.z1_prior = (
            Variable(torch.zeros(batch_size, hidden_dim)),
            Variable(torch.ones(batch_size, hidden_dim))
        )

        self.transition = Transition(hidden_dim)
        self.first_inference = FirstInference(data_dim, hidden_dim)
        self.inference = Inference(data_dim, hidden_dim)
        self.generator = Generator(hidden_dim, data_dim)


    def forward(self, sequence):
        loss = Variable(torch.zeros(1))
        generations = []
        
        inferred_z1_post = self.first_inference(sequence[0])
        loss = loss + KL(inferred_z1_post, z1_prior)

        z1_sample = sample(inferred_z1_post)

        gen1_dist = self.generator(z1_sample)
        loss = loss + LL(gen1_dist, sequence[0])
        generations.append(gen1_dist[0])

        prev_z_sample = z1_sample
        for t in range(1, len(sequence)):
            z_prior = self.transition(prev_z_sample)
            inferred_z_post = self.inference(sequence[t], prev_z_sample)
            loss = loss + KL(inferred_z_post, z_prior)

            z_sample = sample(inferred_z_post)

            gen_dist = self.generator(z_sample)
            loss = loss + LL(gen_dist, sequence[t])
            generations.append(gen_dist[0])

        return generations, - loss / len(sequence)

def make_seq(length, dim):
    sequence = [torch.zeros(1, 2).normal_(0.5, 0.1)]
    for i in range(1, length):
        noise = torch.zeros(1, 2) # torch.normal([0.0, 0.0], [0.1, 0.1])
        sequence.append(sequence[i-1] + noise)
    sequence = [Variable(x) for x in sequence]
    return sequence

# def zero_params():
#     for module in model:
#         module.zero_grad()

model = WholeModel()
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-2)

for i in range(10000):
    sequence = make_seq(1, 2)
    generations, loss = model(sequence)
    # print(loss)

    model.zero_grad()
    # zero_params()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        seq_data = [x.data for x in sequence]
        gen_data = [gen.data for gen in generations]
        print("Loss: ", loss.data[0])
        for j in range(len(sequence)):
            print(seq_data[j][0][0], gen_data[j][0][0])
        

    




