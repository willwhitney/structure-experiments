import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import traceback
import ipdb

from util import *
from modules import *
from params import *

KL = LogSquaredGaussianKLD().type(dtype)
LL = GaussianLL().type(dtype)
# LL = MotionGaussianLL().type(dtype)
# LL = LogSquaredGaussianLL().type(dtype)
bce = nn.BCELoss().type(dtype)
bce.size_average = False

mse_loss = nn.MSELoss().type(dtype)
bce_loss = nn.BCELoss().type(dtype)


def mse(a, b):
    return sum([(a[i] - b[i]).norm()**2 for i in range(len(a))]) / len(a)


def motion_diffs(sequence):
    if len(sequence) == 1:
        return Variable(torch.zeros(sequence[0].size()))
    forward_diffs = []
    for t in range(len(sequence) - 1):
        diff = torch.gt(torch.abs(sequence[t] - sequence[t + 1]), 0.05)
        forward_diffs.append(diff)

    bidirectional_diffs = [forward_diffs[0]]
    for t in range(1, len(sequence) - 1):
        sum_diff = torch.max(forward_diffs[t - 1], forward_diffs[t])
        bidirectional_diffs.append(sum_diff)
    bidirectional_diffs.append(forward_diffs[-1])
    float_diffs = [diff.float() for diff in bidirectional_diffs]
    return float_diffs


def batch_normalize(a):
    """ Requires `a` to be 2-dimensional, with dim 0 the batch dim """
    norms = a.pow(2).sum(1).sqrt()
    return a / (norms.unsqueeze(1) + 1e-8)


class Autoencoder(nn.Module):
    def __init__(self, hidden_dim, img_size,
                 inference=DCGANFirstInference, generator=DCGANGenerator):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_dim = [opt.channels, img_size, img_size]

        self.inference = inference(self.image_dim, self.hidden_dim)
        self.generator = generator(self.hidden_dim, self.image_dim)

    def forward(self, x):
        latents = self.inference(x)
        reconstruction = self.generator(latents)
        reconstruction_loss = mse_loss(reconstruction, x)

        return reconstruction_loss, latents, reconstruction


def batch_select(x, latent_dim, start=None, end=None):
    start_loc = None if start is None else start * latent_dim
    end_loc = None if end is None else (end+1) * latent_dim

    try:
        result = x[:, start_loc: end_loc]
    except ValueError as e:
        return Variable().type(dtype)

    return result

def batch_select_all_but(x, latent_dim, index_to_skip):
    selected = [batch_select(x, latent_dim, end=index_to_skip - 1),
                batch_select(x, latent_dim, start=index_to_skip + 1)]
    selected = [s for s in selected if len(s) > 0]
    return torch.cat(selected, 1)

def batch_replace(latents, latent, latent_dim, index_to_replace):
    selected = [
        batch_select(latents, latent_dim, end=index_to_replace - 1),
        latent,
        batch_select(latents, latent_dim, start=index_to_replace + 1),
    ]
    selected = [s for s in selected if len(s) > 0]
    return torch.cat(selected, 1)


class IndependenceAdversary(nn.Module):
    def __init__(self, factors, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.factors = factors

        self.adversary = Adversary(latent_dim * factors, latent_dim, n_layers=4)

    def forward(self, latents_a, latents_b):
        outputs = {
            'true_loss': None,
            'true_outputs': None,
            'false_loss': 0,
            'false_outputs': 0,
        }

        true_output = self.infer(latents_a)
        true_target = Variable(torch.ones(true_output.size()))
        true_target = true_target.type(dtype)
        outputs['true_loss'] = bce_loss(true_output, true_target)
        outputs['true_outputs'] = true_output

        false_target = Variable(torch.zeros(true_target.size())).type(dtype)
        for i in range(self.factors):
            latent_b = batch_select(latents_b, self.latent_dim, start=i, end=i)
            false_example = batch_replace(latents_a, latent_b, self.latent_dim, i)
            false_output = self.infer(false_example)
            outputs['false_loss'] += (1 / self.factors) * bce_loss(
                false_output, false_target)
            outputs['false_outputs'] += false_output

        
        outputs['false_outputs'] /= self.factors
        return outputs

    def infer(self, latents):
        return self.adversary(latents)

            
class IndependentAutoencoder(nn.Module):
    def __init__(self, n_latents, latent_dim, img_size,
                 inference=DCGANFirstInference, generator=DCGANGenerator):
        super().__init__()
        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.image_dim = [opt.channels, img_size, img_size]

        total_z_dim = n_latents * latent_dim

        self.inference = inference(self.image_dim, total_z_dim)
        self.generator = generator(total_z_dim, self.image_dim)

    def forward(self, adversary, x):
        output = {
            'reconstruction_loss': None,
            'adversarial_loss': None,
            'latents': None,
            'reconstruction': None,
        }

        latents = self.infer(x)
        reconstruction = self.generate(latents)
        output['reconstruction_loss'] = mse_loss(reconstruction, x)

        adversary_outputs = adversary.infer(latents)
        stacked_adversary_outputs = torch.cat(adversary_outputs, 0)
        adversary_targets = torch.Tensor(stacked_adversary_outputs.size())
        adversary_targets = Variable(adversary_targets.fill_(0.5).type(dtype))

        output['adversarial_loss'] = bce_loss(stacked_adversary_outputs,
                                              adversary_targets)
        output['latents'] = latents.detach()
        output['reconstruction'] = reconstruction

        return output

    def infer(self, x, noise=opt.latent_noise):
        latents = self.inference(x)

        normed_latents = []
        for i in range(self.n_latents):
            single_latent = batch_select(
                latents, self.latent_dim, start=i, end=i)
            normed_latents.append(batch_normalize(single_latent))
        latents = torch.cat(normed_latents, 1)

        if noise > 0:
            latent_noise = torch.normal(means=torch.zeros(latents.size()),
                                        std=torch.ones(latents.size()))
            latents += noise * Variable(latent_noise).type(dtype)
        return latents

    def generate(self, z):
        return self.generator(z)

    def generate_independent_posterior(self, sequence):
        latents = [self.infer(s, noise=0) for s in sequence]
        l0 = latents[0]

        posterior_generations = [self.generate(latent) for latent in latents]
        all_generations = [posterior_generations]
        for l in range(self.n_latents):
            generations_l = [posterior_generations[0]]
            for i in range(1, len(sequence)):
                new_latent_l = batch_select(
                    latents[i], self.latent_dim, start=l, end=l)
                new_latents = batch_replace(
                    l0, new_latent_l, self.latent_dim, l)
                generations_l.append(self.generate(new_latents))
            all_generations.append(generations_l)

        return all_generations
        





class IndependentModel(nn.Module):
    def __init__(self, n_latents, hidden_dim, img_size,
                 transition=Transition, first_inference=DCGANFirstInference,
                 inference=DCGANInference, generator=DCGANGenerator):
        super(IndependentModel, self).__init__()
        self.n_latents = n_latents
        self.hidden_dim = hidden_dim
        self.image_dim = [opt.channels, img_size, img_size]

        # the second term is zeros because we're using log(sigma^2)
        # log(1^2) == 0
        single_z1 = (
            Variable(torch.zeros(opt.batch_size, self.hidden_dim).type(dtype)),
            Variable(torch.zeros(opt.batch_size, self.hidden_dim).type(dtype))
        )
        self.z1_prior = (
            torch.cat([single_z1[0] for _ in range(n_latents)], 1),
            torch.cat([single_z1[1] for _ in range(n_latents)], 1)
        )

        total_z_dim = n_latents * self.hidden_dim

        self.transitions = nn.ModuleList([transition(self.hidden_dim,
                                                     layers=opt.trans_layers)
                                          for _ in range(n_latents)])

        self.inference = inference(self.image_dim, total_z_dim)
        self.generator = generator(total_z_dim, self.image_dim)

    def predict_latent(self, latents):
        (latent1, latent2) = latents
        z_prior = []
        for i, trans in enumerate(self.transitions):
            previous1 = latent1[:, i *
                                self.hidden_dim: (i + 1) * self.hidden_dim]
            previous2 = latent2[:, i *
                                self.hidden_dim: (i + 1) * self.hidden_dim]
            z_prior.append(trans((previous1, previous2)))

        cat_prior = (torch.cat([prior[0] for prior in z_prior], 1),
                     torch.cat([prior[1] for prior in z_prior], 1))
        return cat_prior

    def forward(self, sequence, motion_weight=0):
        reshaped_sequence = [x.resize(x.size(0), *self.image_dim)
                             for x in sequence]

        output = {
            'generations': [],
            'prior_variances': [],
            'posterior_variances': [],
            'seq_divergence': Variable(torch.zeros(1).type(dtype)),
            'seq_prior_div': Variable(torch.zeros(1).type(dtype)),
            'seq_trans_div': Variable(torch.zeros(1).type(dtype)),
            'seq_nll': Variable(torch.zeros(1).type(dtype)),
        }

        # if len(sequence) > 1:
        #     diffs = motion_diffs(sequence)
        #     pixel_weights = [diff * motion_weight + 1 for diff in diffs]
        # else:
        #     ones = Variable(torch.ones(sequence[0].size()).type(dtype))
        #     pixel_weights = [ones]

        cat_prior = self.z1_prior
        latents = [cat_prior[0]]
        for t in range(len(sequence)):
            start_div = 2
            # give it the sample from z_{t-1}
            # inferred_z_post = self.inference(reshaped_sequence[t+1], z_sample)

            # give it the mean and logvar2 of the prior p(z_t | z_{t-1})
            inferred_z_post = self.inference(reshaped_sequence[t],
                                             (cat_prior[0].detach(),
                                              cat_prior[1].detach()))

            divergence = KL(inferred_z_post, cat_prior)
            output['seq_divergence'] += divergence
            if math.isnan(output['seq_divergence'].data.sum()):
                pdb.set_trace()
            if t == 0:
                output['seq_prior_div'] += divergence
            elif t >= start_div:
                output['seq_trans_div'] += divergence

            output['posterior_variances'].append(inferred_z_post[1])

            z_sample = sample_log2(inferred_z_post)
            latents.append(inferred_z_post[0])
            gen_dist = self.generator(z_sample)

            if opt.loss == 'normal':
                log_likelihood = LL(gen_dist,
                                    reshaped_sequence[t])
            elif opt.loss == 'bce':
                log_likelihood = -bce(gen_dist[0], reshaped_sequence[t]) / \
                    sequence[0].size(0)
            else:
                raise Exception('Invalid loss function.')

            output['seq_nll'] += - log_likelihood
            if math.isnan(output['seq_nll'].data.sum()):
                pdb.set_trace()

            output['generations'].append(gen_dist)

            if t < len(sequence) - 1:
                if t >= start_div - 1:
                    cat_prior = self.predict_latent((latents[-2], latents[-1]))
                    output['prior_variances'].append(cat_prior[1])
                else:
                    cat_prior = self.z1_prior

        output['seq_nll'] /= len(sequence)
        output['seq_divergence'] /= len(sequence)
        if len(sequence) > 1:
            output['seq_trans_div'] /= (len(sequence) - start_div)

        return output

    def generate(self, priming, steps, sampling=True):
        priming_steps = len(priming)

        generations = torch.Tensor(steps, opt.batch_size, *self.image_dim)
        generations[:priming_steps].copy_(torch.stack(priming).data)

        latents = [self.z1_prior[0]]
        latent = self.inference(priming[0], self.z1_prior)[0]
        for t in range(1, priming_steps):
            latents.append(latent)
            latent = self.inference(priming[t],
                                    self.predict_latent((latents[-2],
                                                         latent)))[0]

        for t in range(steps - priming_steps):
            # make a transition
            latents.append(latent)
            latent_dist = self.predict_latent((latents[-2], latent))

            if sampling:
                latent = sample_log2(latent_dist)
            else:
                latent = latent_dist[0]
            generated_frame = self.generator(latent)[0].data
            generations[t + priming_steps].copy_(generated_frame)
        return generations

    def generate_independent(self, priming, steps, sampling=True):
        priming_steps = len(priming)

        generations = torch.Tensor(self.n_latents, steps, opt.batch_size,
                                   *self.image_dim)
        generations[:, :priming_steps].copy_(torch.stack(priming).data)

        latents = [self.z1_prior[0]]
        latent = self.inference(priming[0], self.z1_prior)[0]
        for t in range(1, priming_steps):
            latents.append(latent)
            latent = self.inference(priming[t],
                                    self.predict_latent((latents[-2],
                                                         latent)))[0]

        starting_latent1 = latents[-2].clone()
        starting_latent2 = latent.clone()
        for z_i in range(self.n_latents):
            latent1 = starting_latent1.clone()
            latent2 = starting_latent2.clone()
            trans = self.transitions[z_i]
            for t in range(steps - priming_steps):
                previous1 = latent1[:,
                                    z_i * self.hidden_dim:
                                    (z_i + 1) * self.hidden_dim].clone()
                previous2 = latent2[:,
                                    z_i * self.hidden_dim:
                                    (z_i + 1) * self.hidden_dim].clone()
                predicted_z = trans((previous1, previous2))

                if sampling:
                    new_z = sample_log2(predicted_z)
                else:
                    new_z = predicted_z[0]
                latent1 = latent2.clone()
                latent2[:, z_i * self.hidden_dim:
                        (z_i + 1) * self.hidden_dim] = new_z

                generated_frame = self.generator(latent2)[0].data
                generations[z_i, t + priming_steps].copy_(generated_frame)
        return generations

    def generate_variations(self, priming, steps):
        priming_steps = len(priming)

        generations = torch.Tensor(self.n_latents, steps, opt.batch_size,
                                   *self.image_dim)
        generations[:, :priming_steps].copy_(torch.stack(priming).data)

        latents = [self.z1_prior[0]]
        latent = self.inference(priming[0], self.z1_prior)[0]
        for t in range(1, priming_steps):
            latents.append(latent)
            latent = self.inference(priming[t],
                                    self.predict_latent((latents[-2],
                                                         latent)))[0]

        starting_latent = latent.clone()
        for z_i in range(self.n_latents):
            latent = starting_latent.clone()
            for t in range(steps - priming_steps):
                latent[:, z_i * self.hidden_dim:
                       (z_i + 1) * self.hidden_dim].data.normal_(0, 1)

                generated_frame = self.generator(latent)[0].data
                generations[z_i, t + priming_steps].copy_(generated_frame)
        return generations

    def generate_interpolations(self, priming, steps, scale=15):
        priming_steps = len(priming)

        generations = torch.Tensor(self.n_latents, steps, opt.batch_size,
                                   *self.image_dim)
        generations[:, :priming_steps].copy_(torch.stack(priming).data)

        latents = [self.z1_prior[0]]
        latent = self.inference(priming[0], self.z1_prior)[0]
        for t in range(1, priming_steps):
            latents.append(latent)
            latent = self.inference(priming[t],
                                    self.predict_latent((latents[-2],
                                                         latent)))[0]

        z_const = latent.clone()
        noise = Variable(torch.zeros(
            latent.size(0), self.hidden_dim).normal_(0, 1).type(dtype))
        for batch_noise in noise.data:
            batch_noise.div_(batch_noise.norm()).mul_(scale)

        for z_i in range(self.n_latents):
            latent = z_const.clone()
            single_z_const = z_const[:, z_i * self.hidden_dim:
                                     (z_i + 1) * self.hidden_dim]

            for t, alpha in enumerate(torch.linspace(
                    -1, 1, steps - priming_steps)):
                new_z = single_z_const + alpha * noise
                latent[:, z_i * self.hidden_dim:
                       (z_i + 1) * self.hidden_dim] = new_z
                generated_frame = self.generator(latent)[0].data
                generations[z_i, t + priming_steps].copy_(generated_frame)
        return generations

    def generate_independent_posterior(self, sequence):
        for x in sequence:
            x.volatile = True

        posteriors = []
        posterior_generations = []
        inferred_z_post = self.inference(sequence[0], self.z1_prior)

        latents = [self.z1_prior[0]]
        for t in range(len(sequence)):
            posteriors.append(inferred_z_post[0])
            z_sample = inferred_z_post[0]
            latents.append(z_sample)
            gen_dist = self.generator(z_sample)
            posterior_generations.append(gen_dist[0].cpu())

            if t < len(sequence) - 1:
                cat_prior = self.predict_latent((latents[-2], z_sample))
                inferred_z_post = self.inference(sequence[t + 1],
                                                 cat_prior)

        priming_steps = 2
        all_generations = [posterior_generations]
        for l in range(self.n_latents):
            z = posteriors[1].clone()
            current_gen = [posterior_generations[i]
                           for i in range(priming_steps)]

            start, end = l * self.hidden_dim, (l + 1) * self.hidden_dim
            for t in range(priming_steps, len(sequence)):
                z[:, start: end].data.copy_(
                    posteriors[t].data[:, start: end])
                gen_dist = self.generator(z)
                current_gen.append(gen_dist[0].cpu())
            all_generations.append(current_gen)
        return all_generations


class DeterministicModel(nn.Module):
    def __init__(self, n_latents, hidden_dim, img_size,
                 transition=Transition, first_inference=DCGANFirstInference,
                 inference=DCGANInference, generator=DCGANGenerator):
        super(DeterministicModel, self).__init__()
        self.n_latents = n_latents
        self.hidden_dim = hidden_dim
        self.image_dim = [opt.channels, img_size, img_size]

        self.total_z_dim = n_latents * self.hidden_dim
        single_z1 = (
            Variable(torch.zeros(opt.batch_size, self.hidden_dim).type(dtype)),
            Variable(torch.zeros(opt.batch_size, self.hidden_dim).type(dtype))
        )
        self.z1_prior = (
            torch.cat([single_z1[0] for _ in range(n_latents)], 1),
            torch.cat([single_z1[1] for _ in range(n_latents)], 1)
        )

        self.transitions = nn.ModuleList([transition(self.hidden_dim,
                                                     layers=opt.trans_layers)
                                          for _ in range(n_latents)])

        self.inference = inference(self.image_dim, self.total_z_dim)
        self.generator = generator(self.total_z_dim, self.image_dim)

    def predict_latent(self, latents):
        latent1, latent2 = latents
        z_prior = []
        for i, trans in enumerate(self.transitions):
            previous1 = latent1[:, i *
                                self.hidden_dim: (i + 1) * self.hidden_dim]
            previous2 = latent2[:, i *
                                self.hidden_dim: (i + 1) * self.hidden_dim]
            z_prior.append(trans((previous1, previous2)))

        prediction = torch.cat([prior[0] for prior in z_prior], 1)
        prediction = batch_normalize(prediction)
        return (prediction,
                Variable(torch.zeros(prediction.size()).type(dtype)))

    def infer(self, observation, prior):
        inferred_z = self.inference(observation,
                                    (prior[0].detach(),
                                     prior[1].detach()))[0]
        return batch_normalize(inferred_z)

    def forward(self, sequence, motion_weight=0):
        reshaped_sequence = [x.resize(x.size(0), *self.image_dim)
                             for x in sequence]

        output = {
            'generations': [],
            'prior_variances': [],
            'posterior_variances': [],
            'seq_divergence': Variable(torch.zeros(1).type(dtype)),
            'seq_prior_div': Variable(torch.zeros(1).type(dtype)),
            'seq_trans_div': Variable(torch.zeros(1).type(dtype)),
            'seq_nll': Variable(torch.zeros(1).type(dtype)),
        }

        zero_latent = self.z1_prior[0]
        cat_prior = self.z1_prior
        latents = [zero_latent]
        for t in range(len(sequence)):
            start_div = 2
            inferred_z_post = self.infer(reshaped_sequence[t],
                                         (cat_prior[0].detach(),
                                          cat_prior[1].detach()))

            divergence = 100 * mse(inferred_z_post, cat_prior[0])
            if t >= start_div:
                output['seq_divergence'] += divergence
                output['seq_trans_div'] += divergence

            output['posterior_variances'].append(zero_latent.clone())

            z_sample = inferred_z_post
            latents.append(z_sample)
            noise = z_sample.clone()
            noise.data.normal_().mul_(0.02)
            gen_dist = self.generator(z_sample + noise)
            # gen_dist = self.generator(z_sample)

            if opt.loss == 'normal':
                log_likelihood = LL(gen_dist,
                                    reshaped_sequence[t])
            elif opt.loss == 'bce':
                log_likelihood = -bce(gen_dist[0], reshaped_sequence[t]) / \
                    sequence[0].size(0)
            else:
                raise Exception('Invalid loss function.')

            output['seq_nll'] += - log_likelihood
            if math.isnan(output['seq_nll'].data.sum()):
                pdb.set_trace()

            output['generations'].append(gen_dist)

            if t < len(sequence) - 1:
                if t >= start_div - 1:
                    cat_prior = self.predict_latent((latents[-2], z_sample))
                    output['prior_variances'].append(cat_prior[1])
                else:
                    cat_prior = self.z1_prior

        output['seq_nll'] /= len(sequence)
        output['seq_divergence'] /= len(sequence) - start_div
        if len(sequence) > 1:
            output['seq_trans_div'] /= (len(sequence) - start_div)

        return output

    def generate(self, priming, steps, sampling=True):
        priming_steps = len(priming)

        generations = torch.Tensor(steps, opt.batch_size, *self.image_dim)
        generations[:priming_steps].copy_(torch.stack(priming).data)

        latents = [self.z1_prior[0]]
        latent = self.infer(priming[0], self.z1_prior)
        for t in range(1, priming_steps):
            latents.append(latent)
            latent = self.infer(priming[t],
                                self.predict_latent((latents[-2], latent)))

        for t in range(steps - priming_steps):
            # make a transition
            latents.append(latent)
            latent_dist = self.predict_latent((latents[-2], latent))
            latent = latent_dist[0]
            generated_frame = self.generator(latent)[0].data
            generations[t + priming_steps].copy_(generated_frame)
        return generations

    def generate_independent(self, priming, steps, sampling=True):
        priming_steps = len(priming)

        generations = torch.Tensor(self.n_latents, steps, opt.batch_size,
                                   *self.image_dim)
        generations[:, :priming_steps].copy_(torch.stack(priming).data)

        latents = [self.z1_prior[0]]
        latent = self.infer(priming[0], self.z1_prior)
        for t in range(1, priming_steps):
            latents.append(latent)
            latent = self.infer(priming[t],
                                self.predict_latent((latents[-2],
                                                     latent)))

        starting_latent = latent.clone()
        for z_i in range(self.n_latents):
            latent = starting_latent.clone()
            trans = self.transitions[z_i]
            for t in range(steps - priming_steps):
                previous = latent[:,
                                  z_i * self.hidden_dim:
                                  (z_i + 1) * self.hidden_dim].clone()
                predicted_z = trans(previous)
                new_z = predicted_z[0]
                latent[:, z_i * self.hidden_dim:
                       (z_i + 1) * self.hidden_dim] = new_z
                latent = batch_normalize(latent)
                generated_frame = self.generator(latent)[0].data
                generations[z_i, t + priming_steps].copy_(generated_frame)
        return generations

    def generate_interpolations(self, priming, steps, scale=1):
        priming_steps = len(priming)

        generations = torch.Tensor(self.n_latents, steps, opt.batch_size,
                                   *self.image_dim)
        generations[:, :priming_steps].copy_(torch.stack(priming).data)

        latents = [self.z1_prior[0]]
        latent = self.infer(priming[0], self.z1_prior)
        for t in range(1, priming_steps):
            latents.append(latent)
            latent = self.infer(priming[t],
                                self.predict_latent((latents[-2],
                                                     latent)))

        z_const = latent.clone()
        noise = Variable(torch.zeros(
            latent.size(0), self.hidden_dim).normal_(0, 1).type(dtype))
        for batch_noise in noise.data:
            batch_noise.div_(batch_noise.norm()).mul_(scale)

        for z_i in range(self.n_latents):
            latent = z_const.clone()
            single_z_const = z_const[:, z_i * self.hidden_dim:
                                     (z_i + 1) * self.hidden_dim]

            for t, alpha in enumerate(torch.linspace(
                    -1, 1, steps - priming_steps)):
                new_z = single_z_const + alpha * noise
                latent[:, z_i * self.hidden_dim:
                       (z_i + 1) * self.hidden_dim] = new_z

                latent = batch_normalize(latent)
                generated_frame = self.generator(latent)[0].data
                generations[z_i, t + priming_steps].copy_(generated_frame)
        return generations

    def generate_independent_posterior(self, sequence):
        for x in sequence:
            x.volatile = True

        posteriors = []
        posterior_generations = []
        inferred_z_post = self.infer(sequence[0], self.z1_prior)

        latents = [self.z1_prior[0]]
        for t in range(len(sequence)):
            posteriors.append(inferred_z_post)
            z_sample = inferred_z_post
            latents.append(z_sample)
            gen_dist = self.generator(z_sample)
            posterior_generations.append(gen_dist[0].cpu())

            if t < len(sequence) - 1:
                cat_prior = self.predict_latent((latents[-2], z_sample))
                inferred_z_post = self.infer(sequence[t + 1],
                                             cat_prior)

        priming_steps = 2
        all_generations = [posterior_generations]
        for l in range(self.n_latents):
            z = posteriors[1].clone()
            current_gen = [posterior_generations[i]
                           for i in range(priming_steps)]

            start, end = l * self.hidden_dim, (l + 1) * self.hidden_dim
            for t in range(priming_steps, len(sequence)):
                z[:, start: end].data.copy_(
                    posteriors[t].data[:, start: end])
                z = batch_normalize(z)
                gen_dist = self.generator(z)
                current_gen.append(gen_dist[0].cpu())
            all_generations.append(current_gen)
        return all_generations


class MSEModel(nn.Module):
    def __init__(self, img_size):
        super(MSEModel, self).__init__()
        self.img_size = img_size
        self.hidden_dim = 100

        image_dim = [opt.channels, self.img_size, self.img_size]
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
        # batch_size = sequence[0].size(0)
        generations = [(sequence[0], None)]

        reshaped_sequence = sequence
        if isinstance(self.inference, ConvInference):
            reshaped_sequence = [x.resize(x.size(0), opt.channels, self.img_size, self.img_size)
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
            priming = [x.resize(x.size(0), opt.channels, self.img_size, self.img_size)
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


class VAEModel(nn.Module):
    def __init__(self, hidden_dim, g_size):
        super(VAEModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.img_size = img_size

        self.z1_prior = (
            Variable(torch.zeros(opt.batch_size, self.hidden_dim).type(dtype)),
            Variable(torch.ones(opt.batch_size, self.hidden_dim).type(dtype))
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
        # batch_size = sequence[0].size(0)
        generations = []

        reshaped_sequence = sequence
        if isinstance(self.inference, ConvInference):
            reshaped_sequence = [x.resize(x.size(0),
                                          3, self.img_size, self.img_size)
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
            # log_likelihood = LL(gen_dist, reshaped_sequence[t])
            log_likelihood = LL(gen_dist, reshaped_sequence[t])
            loss = loss - log_likelihood

            generations.append(gen_dist)

            if t < len(sequence) - 1:
                z_prior = self.transition(z_sample)
                # z_prior = (z_prior[0],
                #            Variable(torch.ones(z_prior[1].size()) * 5e-2).type(dtype))
                inferred_z_post = self.inference(
                    reshaped_sequence[t + 1], z_sample)

                z_var_mean += z_prior[1].mean().data[0]
                z_var_min = min(z_var_min, z_prior[1].data.min())
                z_var_max = max(z_var_max, z_prior[1].data.max())

        z_var_mean = z_var_mean / \
            (len(sequence) - 1) if len(sequence) > 1 else -1
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
