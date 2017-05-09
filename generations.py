import torch
from util import *
from params import *
from models import *

def save_all_generations(model, sequence, generations):
    # volatile input -> no saved intermediate values
    # for x in sequence:
    #     x.volatile = True

    priming = sequence[:2]

    save_posterior(model, sequence, generations)
    save_generations(model, priming)
    save_ml(model, priming)
    save_z1_samples(model)
    save_independent_ml(model, priming)
    save_independent_gen(model, priming)
    save_independent_resample(model, priming)
    save_interpolation(model, priming)
    save_single_replacement(model, sequence)

def save_posterior(model, sequence, generations):
    # save some results from the latest batch
    mus_data = [gen[0].data for gen in generations]
    seq_data = [x.data for x in sequence]
    for j in range(5):
        mus = [x[j] #.view(3,image_width,image_width)
               for x in mus_data]
        truth = [x[j] #.view(3,image_width,image_width)
                 for x in seq_data]
        save_tensors_image(opt.save + '/result_'+str(j)+'.png',
                           [truth, mus])

def save_generations(model, priming):
    for p in priming:
        p.volatile = True

    # save sequence generations
    samples = model.generate(priming, 20, True)
    mus = [x.data for x in samples]
    for j in range(5):
        mu = [x[j] #.view(3,image_width,image_width)
              for x in mus]
        save_tensors_image(opt.save + '/gen_'+str(j)+'.png', mu)
        save_gif(opt.save + '/gen_'+str(j)+'.gif', mu)

def save_ml(model, priming):
    for p in priming:
        p.volatile = True

    # save max-likelihood generations
    samples = model.generate(priming, 20, False)
    mus = [x.data for x in samples]
    for j in range(5):
        mu = [x[j] #.view(3,image_width,image_width)
              for x in mus]
        save_tensors_image(opt.save + '/ml_'+str(j)+'.png', mu)
        save_gif(opt.save + '/ml_'+str(j)+'.gif', mu)

def save_z1_samples(model):
    # save samples from the first-frame prior
    if not isinstance(model, MSEModel):
        prior_sample = sample(model.z1_prior)
        image_dist = model.generator(prior_sample)
        image_sample = image_dist[0] #.resize(32, 3, image_width, image_width)
        image_sample = [[image] for image in image_sample]
        save_tensors_image(opt.save + '/prior_samples.png',
                           image_sample)

def save_independent_ml(model, priming):
    for p in priming:
        p.volatile = True

    # save ML generations with only one latent evolving
    if isinstance(model, IndependentModel):
        samples = model.generate_independent(priming, 20, False)
        samples = [[x.data for x in sample_row]
                   for sample_row in samples]
        for j in range(10):
            image = [[x[j] #.view(3,image_width,image_width)
                      for x in sample_row]
                      for sample_row in samples]
            save_tensors_image(opt.save + '/ind_ml_'+str(j)+'.png',
                               image)
            stacked = [image_tensor([image[i][t]
                                     for i in range(len(image))])
                       for t in range(len(image[0]))]
            save_gif(opt.save + '/ind_ml_' + str(j) + '.gif',
                     stacked)

def save_independent_gen(model, priming):
    for p in priming:
        p.volatile = True

    # save samples with only one latent evolving
    if isinstance(model, IndependentModel):
        samples = model.generate_independent(priming, 20, True)
        samples = [[x.data for x in sample_row]
                   for sample_row in samples]
        for j in range(10):
            image = [[x[j] #.view(3,image_width,image_width)
                      for x in sample_row]
                      for sample_row in samples]
            save_tensors_image(opt.save + '/ind_gen_'+str(j)+'.png',
                               image)
            stacked = [image_tensor([image[i][t] for i in range(len(image))])
                       for t in range(len(image[0]))]
            save_gif(opt.save + '/ind_gen_' + str(j) + '.gif',
                     stacked)

def save_independent_resample(model, priming):
    for p in priming:
        p.volatile = True

    # save samples with only one latent randomly sampling
    if isinstance(model, IndependentModel):
        samples = model.generate_variations(priming, 20)
        samples = [[x.data for x in sample_row]
                   for sample_row in samples]
        for j in range(10):
            image = [[x[j] #.view(3,image_width,image_width)
                      for x in sample_row]
                      for sample_row in samples]
            save_tensors_image(
                opt.save + '/ind_resample_'+str(j)+'.png',
                image)

def save_interpolation(model, priming):
    for p in priming:
        p.volatile = True

    # save samples interpolating between noise near the current latent
    if isinstance(model, IndependentModel):
        samples = model.generate_interpolations(priming, 20)

    samples = model.generate_interpolations(priming, 50)
    samples = [[x.data for x in sample_row]
               for sample_row in samples]
    for j in range(10):
        image = [[x[j] #.view(3,image_width,image_width)
                  for x in sample_row]
                  for sample_row in samples]
        save_tensors_image(
            opt.save + '/interp_'+str(j)+'.png',
            image)
        image = [x[2:] for x in image]
        stacked = [image_tensor([image[i][t] for i in range(len(image))])
                   for t in range(len(image[0]))]
        save_gif(opt.save + '/interp_' + str(j) + '.gif',
                 stacked,
                 bounce=True,
                 duration=0.1)


def save_single_replacement(model, sequence):
    samples = model.generate_independent_posterior(sequence)
    samples = [[x.data for x in sample_row]
               for sample_row in samples]

    for j in range(10):
        image = [[x[j]
                  for x in sample_row]
                  for sample_row in samples]
        save_tensors_image(
            opt.save + '/ind_replace_' + str(j) + '.png',
            image)
        stacked = [image_tensor([image[i][t] for i in range(len(image))])
                   for t in range(len(image[0]))]
        save_gif(opt.save + '/ind_replace_' + str(j) + '.gif',
                 stacked)

    # for x in sequence:
    #     x.volatile = True
    #
    # generations, _,_ = model(sequence)
    # mus_data = [gen[0].data for gen in generations]
    # seq_data = [x.data for x in sequence]
    # for j in range(5):
    #     mus = [x[j] #.view(3,image_width,image_width)
    #            for x in mus_data]
    #     truth = [x[j] #.view(3,image_width,image_width)
    #              for x in seq_data]
    #     save_tensors_image(opt.save + '/replace/full_'+str(j)+'.png',
    #                        [truth, mus])
    #
    # generating_latent = z_0[0].clone()
    # generations = model.generator(generating_latent)
    # mus_data = generations[0].data
    #
    # for j in range(5):
    #     mus = mus_data[j].view(3,image_width,image_width)
    #     save_tensors_image(opt.save + '/replace/original_'+str(j)+'.png', [mus])
    #
    #
    # for l in range(8):
    #     generating_latent = z_0[0].clone()
    #     generating_latent[:, l*25 : (l+1)*25].data.copy_(z_1_latents[l].data)
    #     generations = model.generator(generating_latent)
    #     mus_data = generations[0].data
    #     for j in range(5):
    #         mus = mus_data[j].view(3,image_width,image_width)
    #         save_tensors_image(
    #             opt.save + '/replace/replace_'+str(j)+'_'+str(l)+'.png',
    #             [mus])
