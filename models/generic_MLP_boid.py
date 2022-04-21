from functools import partial

import numpy as np
from scipy.spatial import Delaunay

from evaluation.density import generate_rings
from evaluation.polarisation import polarisation
from models import boid
from neighbourhoods.delaunay import delaunay_neighbourhood
from neighbourhoods.metric import metric_neighbourhood
from neighbourhoods.nearest_n import nearest_n_neighbourhood
from tools.MLP import MLP


class MLP_boid(MLP):
    def __init__(self):
        super().__init__(2, [8, 8], 18, sigmoid_last=False)


def generic_MLP_boid_step(population, nnet=None):
    if nnet is None:
        raise ValueError("generic boid wrapper needs a nnet")

    population = population.copy()

    tris = Delaunay(population[:, 0, :])
    rings = generate_rings(population)

    for i in range(len(population)):
        del_neighbours = delaunay_neighbourhood(i, population, tris=tris)

        neighbourhood_generators = [
            lambda: del_neighbours,
            lambda: [n for n in del_neighbours if rings[n] == rings[i]],
            # partial(metric_neighbourhood, i, population),
            # partial(nearest_n_neighbourhood, i, population)
            lambda: del_neighbours,
            lambda: [n for n in del_neighbours if rings[n] == rings[i]],
        ]

        # todo: add more inputs for each neighbourhood
        inputs = [rings[i], polarisation(population[del_neighbours])]
        output = nnet.feed_forward(inputs)

        coh_neighbours = neighbourhood_generators[output[6:10].index(max(output[6:10]))]()

        if output[6:10].index(max(output[6:10])) == output[10:14].index(max(output[10:14])):
            sep_neighbours = coh_neighbours
        else:
            sep_neighbours = neighbourhood_generators[output[10:14].index(max(output[10:14]))]()

        if output[6:10].index(max(output[6:10])) == output[14:18].index(max(output[14:18])):
            ali_neighbours = coh_neighbours
        elif output[10:14].index(max(output[10:14])) == output[14:18].index(max(output[14:18])):
            ali_neighbours = sep_neighbours
        else:
            ali_neighbours = neighbourhood_generators[output[14:18].index(max(output[14:18]))]()

        delta = np.array([0., 0.])

        if len(coh_neighbours) > 0:
            if output[0] > 0.5:
                delta += output[3] * boid.cohesion(i, coh_neighbours, population, strength=1)

        if len(sep_neighbours) > 0:
            if output[1] > 0.5:
                delta += output[4] * boid.separation(i, sep_neighbours, population, strength=1)

        if len(ali_neighbours) > 0:
            if output[2] > 0.5:
                delta += output[5] * boid.alignment(i, ali_neighbours, population, strength=1)

        population[i, 1] += delta

        population[i, 0] += population[i, 1]

    return population
