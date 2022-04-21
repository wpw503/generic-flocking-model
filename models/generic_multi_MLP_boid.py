import math
import random
from functools import partial

import numpy as np
from scipy.spatial import Delaunay

from evaluation.density import generate_rings
from evaluation.polarisation import polarisation
from models import boid
from neighbourhoods.delaunay import delaunay_neighbourhood
from neighbourhoods.metric import metric_neighbourhood
from neighbourhoods.nearest_n import nearest_n_neighbourhood
from config import *
from tools.MLP import MLP


class multi_MLP_neighbour_boid(MLP):
    def __init__(self):
        super().__init__(2, [4], 4)


class multi_MLP_strength_boid(MLP):
    def __init__(self):
        super().__init__(2, [4], 2, sigmoid_last=False)


def multi_MLP_mate(ind1, ind2):
    crossover_point = random.randint(1, 5)

    crossover_indexes = [len(multi_MLP_neighbour_boid().get_weights()),
                         len(multi_MLP_neighbour_boid().get_weights()),
                         len(multi_MLP_neighbour_boid().get_weights()),
                         len(multi_MLP_strength_boid().get_weights()),
                         len(multi_MLP_strength_boid().get_weights()),
                         len(multi_MLP_strength_boid().get_weights())]

    crossover = 0

    for i in range(crossover_point):
        crossover += crossover_indexes[i]

    temp = ind1[0:crossover] + ind2[crossover:]
    temp2 = ind2[0:crossover] + ind1[crossover:]

    ind1 = temp
    ind2 = temp2

    return ind1, ind2


def generic_multi_MLP_boid_step(population,
                                coh_neighbourhood_nnet=None, sep_neighbourhood_nnet=None, ali_neighbourhood_nnet=None,
                                coh_nnet=None, sep_nnet=None, ali_nnet=None):

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

        coh_neighbourhood_output = coh_neighbourhood_nnet.feed_forward(inputs)
        sep_neighbourhood_output = sep_neighbourhood_nnet.feed_forward(inputs)
        ali_neighbourhood_output = ali_neighbourhood_nnet.feed_forward(inputs)

        coh_output = coh_nnet.feed_forward(inputs)
        sep_output = sep_nnet.feed_forward(inputs)
        ali_output = ali_nnet.feed_forward(inputs)

        coh_neighbours = neighbourhood_generators[coh_neighbourhood_output.index(max(coh_neighbourhood_output))]()

        if coh_neighbourhood_output.index(max(coh_neighbourhood_output)) == \
                sep_neighbourhood_output.index(max(sep_neighbourhood_output)):
            sep_neighbours = coh_neighbours
        else:
            sep_neighbours = neighbourhood_generators[sep_neighbourhood_output.index(max(sep_neighbourhood_output))]()

        if coh_neighbourhood_output.index(max(coh_neighbourhood_output)) == \
                ali_neighbourhood_output.index(max(ali_neighbourhood_output)):
            ali_neighbours = coh_neighbours
        elif sep_neighbourhood_output.index(max(sep_neighbourhood_output)) == \
                ali_neighbourhood_output.index(max(ali_neighbourhood_output)):
            ali_neighbours = sep_neighbours
        else:
            ali_neighbours = neighbourhood_generators[ali_neighbourhood_output.index(max(ali_neighbourhood_output))]()

        delta = np.array([0., 0.])

        if len(coh_neighbours) > 0:
            if coh_output[0] > 0.5:
                delta += coh_output[1] * boid.cohesion(i, coh_neighbours, population, strength=1)

        if len(sep_neighbours) > 0:
            if sep_output[0] > 0.5:
                delta += sep_output[1] * boid.separation(i, sep_neighbours, population, strength=1)

        if len(ali_neighbours) > 0:
            if ali_output[0] > 0.5:
                delta += ali_output[1] * boid.alignment(i, ali_neighbours, population, strength=1)

        population[i, 1] += delta

        population[i, 0] += population[i, 1]

    return population
