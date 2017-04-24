import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util import *
from modules import *
from params import *

KL = GaussianKLD().type(dtype)
LL = GaussianLL().type(dtype)
mse = nn.MSELoss().type(dtype)

class VAEModel(nn.Module):
    def __init__(self, hidden_dim, g_size):
        super(VAEModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.img_size = img_size

        self.z1_prior = (
            Variable(torch.zeros(batch_size, self.hidden_dim).type(dtype)),
            Variable(torch.ones(batch_size, self.hidden_dim).type(dtype))
        )

        image_dim = [3, self.img_size, self.img_size]
        self.transition = Transition(self.hidden_dim)
        # self.first_inference = FirstInference(prod(image_dim), self.hidden_dim)
        # self.inference = Inference(prod(image_dim), self.hidden_dim)
        # self.generator = Generator(self.hidden_dim, prod(image_dim))

        self.first_inference = ConvFirstInference(image_dim, self.hidden_dim)
        self.inference = ConvInference(image_dim, self.hidden_dim)
        self.generator = ConvGenerator(self.hidden_dim, image_dim)

    def forward(self, sequence):
        loss = Variable(torch.zeros(1).type(dtype))
        seq_divergence = Variable(torch.zeros(1).type(dtype),
                                  requires_grad=False)
        batch_size = sequence[0].size(0)
        generations = []

        reshaped_sequence = sequence
        if isinstance(self.inference, ConvInference):
            reshaped_sequence = [x.resize(x.size(0), 3, self.img_size, self.img_size)
                                 for x in sequence]
        # reshaped_sequence = [x.resize(x.size(0), self.img_size, self.img_size, 3)
        #                      for x in sequence]
        # reshaped_sequence = [x.transpose(2, 3).transpose(1, 2)
        #                      for x in reshaped_sequence]

        inferred_z_post = self.first_inference(reshaped_sequence[0])
        z_prior = self.z1_prior

        z_var_mean = 0
        z_var_min = 1e6
        z_var_max = -1
        for t in range(len(sequence)):
            divergence = KL(inferred_z_post, z_prior)
            seq_divergence = seq_divergence + divergence
            loss = loss + divergence

            z_sample = sample(inferred_z_post)
            # z_sample = inferred_z_post[0]

            gen_dist = self.generator(z_sample)
            log_likelihood = LL(gen_dist, reshaped_sequence[t])
            loss = loss - log_likelihood

            generations.append(gen_dist)

            if t < len(sequence) - 1:
                z_prior = self.transition(z_sample)
                # z_prior = (z_prior[0],
                #            Variable(torch.ones(z_prior[1].size()) * 5e-2).type(dtype))
                inferred_z_post = self.inference(reshaped_sequence[t+1], z_sample)

                z_var_mean += z_prior[1].mean().data[0]
                z_var_min = min(z_var_min, z_prior[1].data.min())
                z_var_max = max(z_var_max, z_prior[1].data.max())

        z_var_mean = z_var_mean / (len(sequence) - 1) if len(sequence) > 1 else -1
        return (generations,
                loss / len(sequence),
                seq_divergence.data[0] / len(sequence),
                (z_var_min, z_var_mean, z_var_max))

    def generate(self, priming, steps, sampling=True):
        priming_steps = len(priming)
        generations = []

        if isinstance(self.inference, ConvInference):
            priming = [x.resize(x.size(0),  3, self.img_size, self.img_size)
                       for x in priming]

        latent = self.first_inference(priming[0])[0]
        generations.append(self.generator(latent)[0])
        for t in range(1, priming_steps):
            latent = self.inference(priming[t], latent)[0]
            generations.append(self.generator(latent)[0])

        for t in range(steps - priming_steps):
            latent_dist = self.transition(latent)
            if sampling:
                latent = sample(latent_dist)
            else:
                latent = sample((latent_dist[0], latent_dist[1] / 100))
            generations.append(self.generator(latent)[0])
        return generations

class IndependentModel(nn.Module):
    def __init__(self, n_latents, hidden_dim, img_size):
        super(IndependentModel, self).__init__()
        self.n_latents = n_latents
        self.img_size = img_size
        self.hidden_dim = hidden_dim

        single_z1 = (
            Variable(torch.zeros(batch_size, self.hidden_dim).type(dtype)),
            Variable(torch.ones(batch_size, self.hidden_dim).type(dtype))
        )
        self.z1_prior = (torch.cat([single_z1[0] for _ in range(n_latents)], 1),
                         torch.cat([single_z1[1] for _ in range(n_latents)], 1))

        total_z_dim = n_latents * self.hidden_dim

        image_dim = [3, self.img_size, self.img_size]

        # trans = Transition(self.hidden_dim)
        self.transitions = nn.ModuleList([Transition(self.hidden_dim)
                                          for _ in range(n_latents)])
        # self.first_inference = FirstInference(prod(image_dim), total_z_dim)
        # self.inference = Inference(prod(image_dim), total_z_dim)
        # self.generator = Generator(total_z_dim, prod(image_dim))

        self.first_inference = ConvFirstInference(image_dim, total_z_dim)
        self.inference = ConvInference(image_dim, total_z_dim)
        self.generator = ConvGenerator(total_z_dim, image_dim)

    def predict_latent(self, latent):
        z_prior = []
        for i, trans in enumerate(self.transitions):
            previous = latent[:, i*self.hidden_dim : (i+1)*self.hidden_dim]
            z_prior.append(trans(previous))

        cat_prior = (torch.cat([prior[0] for prior in z_prior], 1),
                     torch.cat([prior[1] for prior in z_prior], 1))
        return cat_prior

    def forward(self, sequence):
        # loss = Variable(torch.zeros(1).type(dtype))
        seq_divergence = Variable(torch.zeros(1).type(dtype))
        seq_prior_div = Variable(torch.zeros(1).type(dtype))
        seq_trans_div = Variable(torch.zeros(1).type(dtype))

        seq_nll = Variable(torch.zeros(1).type(dtype))

        batch_size = sequence[0].size(0)
        generations = []

        reshaped_sequence = sequence
        if isinstance(self.inference, ConvInference):
            reshaped_sequence = [x.resize(x.size(0),
                                          3,
                                          self.img_size,
                                          self.img_size)
                                 for x in sequence]

        inferred_z_post = self.first_inference(reshaped_sequence[0])
        cat_prior = self.z1_prior

        z_var_mean = 0
        z_var_min = 1e6
        z_var_max = -1

        for t in range(len(sequence)):
            divergence = KL(inferred_z_post, cat_prior)
            seq_divergence = seq_divergence + divergence
            if t == 0:
                seq_prior_div = seq_prior_div + divergence
            else:
                seq_trans_div = seq_trans_div + divergence

            # loss = loss + divergence

            z_sample = sample(inferred_z_post)
            # z_sample = inferred_z_post[0]

            gen_dist = self.generator(z_sample)
            log_likelihood = LL(gen_dist, reshaped_sequence[t])
            seq_nll = seq_nll - log_likelihood

            generations.append(gen_dist)

            if t < len(sequence) - 1:
                cat_prior = self.predict_latent(z_sample)

                # give it the sample from z_{t-1}
                # inferred_z_post = self.inference(reshaped_sequence[t+1], z_sample)

                # give it the mean of the prior p(z_t | z_{t-1})
                inferred_z_post = self.inference(reshaped_sequence[t+1],
                                                 cat_prior[0])
                z_var_mean += cat_prior[1].mean().data[0]
                z_var_min = min(z_var_min, cat_prior[1].data.min())
                z_var_max = max(z_var_max, cat_prior[1].data.max())

        seq_divergence = seq_divergence / len(sequence)
        seq_prior_div = seq_prior_div
        seq_trans_div = seq_trans_div / (len(sequence) - 1)
        if len(sequence) > 1:
            z_var_mean = z_var_mean / (len(sequence) - 1)
        else:
            z_var_mean = -1
        return (generations,
                # loss / len(sequence),
                seq_nll / len(sequence),
                (seq_divergence, seq_prior_div, seq_trans_div))
                # (z_var_min, z_var_mean, z_var_max))

    def generate(self, priming, steps, sampling=True):
        priming_steps = len(priming)
        # generations = []

        if isinstance(self.inference, ConvInference) and priming[0].dim() != 4:
            priming = [x.resize(x.size(0), 3, self.img_size, self.img_size)
                       for x in priming]

        generations = [p.cpu() for p in priming]
        latent = self.first_inference(priming[0])[0]
        # generations.append(self.generator(latent)[0])
        for t in range(1, priming_steps):
            latent = self.inference(priming[t],
                                    self.predict_latent(latent)[0])[0]
            # generations.append(self.generator(latent)[0])

        for t in range(steps - priming_steps):
            # make a transition
            latent_dist = self.predict_latent(latent)

            if sampling:
                latent = sample(latent_dist)
            else:
                latent = sample((latent_dist[0], latent_dist[1] / 100))
            generations.append(self.generator(latent)[0].cpu())
        return generations

    def generate_independent(self, priming, steps, sampling=True):
        priming_steps = len(priming)

        if isinstance(self.inference, ConvInference) and priming[0].dim() != 4:
            priming = [x.resize(x.size(0), 3, self.img_size, self.img_size)
                       for x in priming]

        generations = [[p.cpu() for p in priming]
                       for _ in range(self.n_latents)]

        latent = self.first_inference(priming[0])[0]
        # generation = self.generator(latent)[0]
        # [generations[i].append(generation) for i in range(len(generations))]
        for t in range(1, priming_steps):
            latent = self.inference(priming[t],
                                    self.predict_latent(latent)[0])[0]
            # generation = self.generator(latent)[0]
            # [generations[i].append(generation) for i in range(len(generations))]

        starting_latent = latent.clone()
        for z_i in range(self.n_latents):
            latent = starting_latent.clone()
            trans = self.transitions[z_i]
            for t in range(steps - priming_steps):
                previous = latent[:, z_i*self.hidden_dim : (z_i+1)*self.hidden_dim].clone()
                predicted_z = trans(previous)

                if sampling:
                    new_z = sample(predicted_z)
                else:
                    new_z = sample((predicted_z[0], predicted_z[1] / 100))
                latent[:, z_i*self.hidden_dim : (z_i+1)*self.hidden_dim] = new_z

                generations[z_i].append(self.generator(latent)[0].cpu())
        return generations

    def generate_variations(self, priming, steps):
        priming_steps = len(priming)
        # generations = [[] for _ in range(self.n_latents)]

        if isinstance(self.inference, ConvInference) and priming[0].dim() != 4:
            priming = [x.resize(x.size(0), 3, self.img_size, self.img_size)
                       for x in priming]

        generations = [[p.cpu() for p in priming]
                       for _ in range(self.n_latents)]
        latent = self.first_inference(priming[0])[0]
        # generation = self.generator(latent)[0]
        # [generations[i].append(generation) for i in range(len(generations))]
        for t in range(1, priming_steps):
            latent = self.inference(priming[t],
                                    self.predict_latent(latent)[0])[0]
            # generation = self.generator(latent)[0]
            # [generations[i].append(generation) for i in range(len(generations))]

        starting_latent = latent.clone()
        for z_i in range(self.n_latents):
            latent = starting_latent.clone()
            for t in range(steps - priming_steps):
                latent[:, z_i*self.hidden_dim : (z_i+1)*self.hidden_dim].data.normal_(0, 1)

                generations[z_i].append(self.generator(latent)[0].cpu())
        return generations

    def generate_interpolations(self, priming, steps):
        priming_steps = len(priming)
        # generations = [[] for _ in range(self.n_latents)]
        if isinstance(self.inference, ConvInference) and priming[0].dim() != 4:
            priming = [x.resize(x.size(0), 3, self.img_size, self.img_size)
                       for x in priming]

        generations = [[p.cpu() for p in priming]
                       for _ in range(self.n_latents)]
        latent = self.first_inference(priming[0])[0]
        # generation = self.generator(latent)[0]
        # [generations[i].append(generation) for i in range(len(generations))]
        for t in range(1, priming_steps):
            latent = self.inference(priming[t],
                                    self.predict_latent(latent)[0])[0]
            # generation = self.generator(latent)[0]
            # [generations[i].append(generation) for i in range(len(generations))]

        z_const = latent.clone()
        noise = Variable(torch.zeros(
            latent.size(0), self.hidden_dim).normal_(0, 1).type(dtype))
        for z_i in range(self.n_latents):
            latent = z_const.clone()
            single_z_const = z_const[:, z_i*self.hidden_dim : (z_i+1)*self.hidden_dim]

            # single_z = latent[:, z_i*self.hidden_dim : (z_i+1)*self.hidden_dim]

            for alpha in torch.linspace(-1, 1, steps - priming_steps):
                latent[:, z_i*self.hidden_dim : (z_i+1)*self.hidden_dim] = single_z_const + alpha * noise
                generations[z_i].append(self.generator(latent)[0].cpu())
        return generations


class MSEModel(nn.Module):
    def __init__(self, img_size):
        super(MSEModel, self).__init__()
        self.img_size = img_size
        self.hidden_dim = 100

        image_dim = [3, self.img_size, self.img_size]
        self.transition = Transition(self.hidden_dim)
        # self.first_inference = FirstInference(prod(image_dim), self.hidden_dim)
        # self.inference = Inference(prod(image_dim), self.hidden_dim)
        # self.generator = Generator(self.hidden_dim, prod(image_dim))

        self.first_inference = ConvFirstInference(image_dim, self.hidden_dim)
        self.inference = ConvInference(image_dim, self.hidden_dim)
        self.generator = ConvGenerator(self.hidden_dim, image_dim)

    def forward(self, sequence):
        loss = Variable(torch.zeros(1).type(dtype))
        seq_divergence = Variable(torch.zeros(1).type(dtype),
                                  requires_grad=False)
        batch_size = sequence[0].size(0)
        generations = [(sequence[0], None)]

        reshaped_sequence = sequence
        if isinstance(self.inference, ConvInference):
            reshaped_sequence = [x.resize(x.size(0), 3, self.img_size, self.img_size)
                                 for x in sequence]
        # reshaped_sequence = [x.transpose(2, 3).transpose(1, 2)
        #                      for x in reshaped_sequence]

        # print(reshaped_sequence[0])
        inferred_z0 = self.first_inference(reshaped_sequence[0])[0]
        inferred_z1 = self.inference(reshaped_sequence[1], inferred_z0)[0]

        current_z = inferred_z1
        for t in range(1, len(sequence)):
            # divergence = KL(inferred_z_post, z_prior)
            # seq_divergence = seq_divergence + divergence
            # loss = loss + divergence

            # z_sample = sample(inferred_z_post)
            # z_sample = inferred_z_post[0]

            gen_dist = self.generator(current_z)
            # print(gen_dist[0][0])
            # generation = gen_dist[0].transpose(1, 2).transpose(2, 3)
            generation = gen_dist[0].resize(gen_dist[0].size(0),
                                            prod(gen_dist[0].size()[1:]))
            loss = loss + mse(generation, sequence[t])

            generations.append((generation, gen_dist[1]))
            # save_tensors_image(name + '/gen_'+str(j)+'.png', mu)

            if t < len(sequence) - 1:
                current_z = self.transition(current_z)[0]
                # z_prior = (z_prior[0],
                        #    Variable(torch.ones(z_prior[1].size()) * 5e-2).type(dtype))
                # inferred_z_post = self.inference(reshaped_sequence[t+1], z_sample)

                # z_var_mean += z_prior[1].mean().data[0]

        # z_var_mean = z_var_mean / (len(sequence) - 1) if len(sequence) > 1 else -1
        return (generations,
                loss / (len(sequence) - 1),
                999,
                999)

    def generate(self, priming, steps, sampling=True):
        generations = [priming[0]]

        if isinstance(self.inference, ConvInference):
            priming = [x.resize(x.size(0),  3, self.img_size, self.img_size)
                       for x in priming]
        # priming = [x.transpose(2, 3).transpose(1, 2)
        #                      for x in priming]

        inferred_z0 = self.first_inference(priming[0])[0]
        inferred_z1 = self.inference(priming[1], inferred_z0)[0]

        current_z = inferred_z1
        for t in range(2, steps):
            gen_dist = self.generator(current_z)
            generation = gen_dist[0].resize(gen_dist[0].size(0),
                                            prod(gen_dist[0].size()[1:]))
            generations.append(generation)

            current_z = self.transition(current_z)[0]

        return generations
