import torch
from util import *
from params import *
from models import *
import os
import ipdb

def save_all_generations(step, model, sequence, generations):
    # volatile input -> no saved intermediate values
    for x in sequence:
        x.volatile = True
    priming = sequence[:2]

    def mkpath(name):
        return os.path.join(opt.save, name, str(step) + '-')

    save_posterior(mkpath("reconstruction"), model, sequence, generations)
    save_z1_samples(mkpath("prior_sample"), model)
    save_interpolation(mkpath("interpolation"), model, priming)

    # these depend on priming the model first, then predicting in some way
    if opt.seq_len > 1:
        save_generations(mkpath("generation"), model, priming, sampling=True)
        save_generations(mkpath("ml_generation"), model, priming, False)
        # save_independent_ml(mkpath("ind_ml"), model, priming)
        save_independent_gen(mkpath("ind_gen"), model, priming, sampling=True)
        save_independent_gen(mkpath("ind_ml"), model, priming, sampling=False)
        save_independent_resample(mkpath("ind_resample"), model, priming)
        save_single_replacement(mkpath("ind_replace"), model, sequence)

@ensure_path_exists
def save_posterior(path, model, sequence, generations):
    # save some results from the latest batch
    mus_data = [gen[0].data for gen in generations]
    seq_data = [x.data for x in sequence]
    for j in range(5):
        mus = [x[j] for x in mus_data]
        truth = [x[j] for x in seq_data]
        save_tensors_image(path + str(j) + '.png', [truth, mus])

@ensure_path_exists
def save_paired_sequence(path, s1, s2):
    for j in range(5):
        save_tensors_image(path + str(j) + '.png',
                           [[x[j].data for x in s1],
                            [x[j].data for x in s2]])

@ensure_path_exists
def save_generations(path, model, priming, sampling=True):
    if sampling and isinstance(model, DeterministicModel):
        return
    for p in priming:
        p.volatile = True

    # save sequence generations
    samples = model.generate(priming, 20, sampling)
    for j in range(5):
        mu = samples[:, j]
        save_tensors_image(path +str(j) + '.png', mu)
        save_gif(path + str(j) + '.gif', mu)

@ensure_path_exists
def save_z1_samples(path, model):
    if isinstance(model, DeterministicModel):
        return
    # save samples from the first-frame prior
    if not isinstance(model, MSEModel):
        prior_sample = sample_log2(model.z1_prior)
        image_dist = model.generator(prior_sample)
        image_sample = image_dist[0]
        image_sample = [[image] for image in image_sample]
        save_tensors_image(path + '.png', image_sample)

def extract_kth_sequence(full_sequence, k):
    """
    Pulls out the kth batch element from each prediction and timestep of a
    [latents x timesteps x batchsize x *imagedim] nested list/tensor
    """
    return [[frame_batch[k] for frame_batch in latent_sequences]
            for latent_sequences in full_sequence]

    # return full_sequence[:, :, k]

# @ensure_path_exists
# def save_independent_ml(path, model, priming):
#     for p in priming:
#         p.volatile = True

#     # save ML generations with only one latent evolving
#     if isinstance(model, IndependentModel):
#         samples = model.generate_independent(priming, 20, sampling=False)
#         samples = [[x.data for x in sample_row]
#                    for sample_row in samples]
#         for j in range(10):
#             image = extract_kth_sequence(samples, j)
#             save_tensors_image(path + str(j)+'.png', image)
#             stacked = [image_tensor([image[i][t]
#                                      for i in range(len(image))])
#                        for t in range(len(image[0]))]
#             save_gif(path + str(j) + '.gif', stacked)

@ensure_path_exists
def save_independent_gen(path, model, priming, sampling=True):
    if sampling and isinstance(model, DeterministicModel):
        return
    for p in priming:
        p.volatile = True

    # save samples with only one latent evolving
    if isinstance(model, IndependentModel):
        samples = model.generate_independent(priming, 20, sampling)
        for j in range(10):
            # image is one batch element of every timestep and every latent
            image = samples[:, :, j]
            save_tensors_image(path +str(j)+'.png', image)

            # stacked is a list of images where each image is the concatenated
            # predictions from each latent at one timestep
            stacked = [image_tensor(image[:, t])
                       for t in range(len(image[0]))]
            save_gif(path + str(j) + '.gif', stacked)

@ensure_path_exists
def save_independent_resample(path, model, priming):
    if isinstance(model, DeterministicModel):
        return
    for p in priming:
        p.volatile = True

    # save samples with only one latent randomly sampling
    if isinstance(model, IndependentModel):
        samples = model.generate_variations(priming, 20)
        for j in range(10):
            image = samples[:, :, j]
            save_tensors_image(path + str(j)+'.png', image)

@ensure_path_exists
def save_interpolation(path, model, priming):
    for p in priming:
        p.volatile = True

    samples = model.generate_interpolations(priming, 50)
    for j in range(10):
        image = samples[:, :, j]
        save_tensors_image(path + str(j)+'.png', image)
        gif_image = image[:, 2:]
        stacked = [image_tensor(gif_image[:, t])
                   for t in range(len(gif_image[0]))]
        save_gif(path + str(j) + '.gif',
                 stacked,
                 bounce=True,
                 duration=0.1)

@ensure_path_exists
def save_single_replacement(path, model, sequence):
    samples = model.generate_independent_posterior(sequence)
    samples = [[x.data for x in sample_row]
               for sample_row in samples]

    for j in range(10):
        image = extract_kth_sequence(samples, j)
        save_tensors_image(path + str(j) + '.png', image)
        stacked = [image_tensor([image[i][t] for i in range(len(image))])
                   for t in range(len(image[0]))]
        save_gif(path + str(j) + '.gif', stacked)

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
