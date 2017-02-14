import pymc3
import numpy as np
import random
import math
import scipy.stats
from functools import reduce

class RandomVariable:
    def __init__(self, values, cpt, prior_prob):
        self.values = values
        self.cpt = cpt
        self.prior_prob = prior_prob

    def __str__(self):
        return str(self.cpt)

    def __repr__(self):
        return str(self)

    def sample(self, parents=None):
        if parents == None:
            probs = self.cpt
        else:
            probs = self.cpt[parents]
        return random.choices(values, probs)[0]

    def likelihood(self, value, parents=None):
        if parents == None:
            probs = self.cpt
            prior = self.prior_prob
        else:
            probs = self.cpt[parents]
            prior = self.prior_prob[parents]
        return probs[value] * prior

def make_point():
    a = random.choices([0, 1], [0.2, 0.8])[0]
    c = random.choices([0, 1], [0.6, 0.4])[0]
    # c_likelihood = {
    #     (0): 0.2,
    #     (1): 0.75,
    # }
    # c = 1 if random.random() < c_likelihood[(a)] else 0

    return (a, c)

data = [make_point() for i in range(200)]

def sample_structure():
    structure_prior = {
        (0, 0): 0.7,
        (1, 0): 0.3,
    }
    structures = list(structure_prior.keys())
    weights = [structure_prior[key] for key in structures]
    structure = random.choices(structures, weights)[0]
    return structure, structure_prior[structure]

def clamp_p(x):
    return math.max(0, math.min(x, 1))

def sample_model(structure):
    # def a():
    #     param = np.random.beta(2, 2)
    #     likelihood = scipy.stats.beta(2, 2).pdf(param)
    #     return param, likelihood
    # def c(parent):
    #     if structure == (0, 0):
    #         param = np.random.beta(2, 2)
    #         likelihood = scipy.stats.beta(2, 2).pdf(param)
    #     else:
    #         if parent == 1:
    #             param = np.random.beta(2, 2)
    #             likelihood = scipy.stats.beta(2, 2).pdf(param)
    #     return param, likelihood
    # return (a, c)

    a_prior = np.random.beta(2, 2)
    a_prior_likelihood = scipy.stats.beta(2, 2).pdf(a_prior)
    a = RandomVariable((0, 1), [1 - a_prior, a_prior], a_prior_likelihood)

    if structure == (1, 0):
        c_prior_false = np.random.beta(2, 2)
        c_prior_true = np.random.beta(2, 2)
        c = RandomVariable((0, 1),
                           [[1 - c_prior_false, c_prior_false],
                            [1 - c_prior_true, c_prior_true]],
                           (scipy.stats.beta(2, 2).pdf(c_prior_false),
                            scipy.stats.beta(2, 2).pdf(c_prior_true)))
    else:
        c_prior = np.random.beta(2, 2)
        c_prior_likelihood = scipy.stats.beta(2, 2).pdf(c_prior)
        c = RandomVariable((0, 1), [1 - c_prior, c_prior], c_prior_likelihood)

    return (a, c)



def prod(l):
    return reduce(lambda x, y: x * y, l)

def p_point(structure, model, point):
    if structure == (0, 0):
        independent_odds = [
            model[0].likelihood(point[0]),
            model[1].likelihood(point[1])
        ]
    else:
        independent_odds = [
            model[0].likelihood(point[0]),
            model[1].likelihood(point[1], point[0])
        ]
    return prod(independent_odds)

def p_data(structure, model, dataset):
    point_odds = [p_point(structure, model, point) for point in dataset]
    return prod(point_odds)

def fit(dataset):
    results = []
    for i in range(10000):
        structure, structure_prior = sample_structure()
        model = sample_model(structure)
        likelihood = p_data(structure, model, data)
        # print(structure_prior)
        # print(model_prior)
        posterior = structure_prior * likelihood
        results.append((structure, model, posterior))
    return results

results = fit(data)
(best_structure, best_model, best_posterior) = max(results, key=lambda x: x[2])
print(best_structure)
print(best_model)
print(best_posterior)
# a_vals = [d[0] for d in data]
# c_vals = [d[1] for d in data]
# print(sum(a_vals) / len(a_vals))
# print(sum(c_vals) / len(c_vals))

# I'm not using the structure correctly.
# It should change the probability in p_data on a point-by-point basis
# instead of just sampling once for the whole dataset like sample_model.
