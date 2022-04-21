import numpy as np

from models import boid
from neighbourhoods.metric import metric_neighbourhood


def generic_boid_step(population,
                      coh_condition=lambda: True, sep_condition=lambda: True, ali_condition=lambda: True,
                      coh_multiplier=lambda: 1, sep_multiplier=lambda: 1, ali_multiplier=lambda: 1,
                      coh_neighbourhood=metric_neighbourhood,
                      sep_neighbourhood=metric_neighbourhood,
                      ali_neighbourhood=metric_neighbourhood):

    population = population.copy()

    for i in range(len(population)):
        coh_neighbours = coh_neighbourhood(i, population)
        sep_neighbours = sep_neighbourhood(i, population)
        ali_neighbours = ali_neighbourhood(i, population)

        delta = np.array([0., 0.])

        if len(coh_neighbours) > 0:
            if coh_condition():
                delta += coh_multiplier() * boid.cohesion(i, coh_neighbours, population)

        if len(sep_neighbours) > 0:
            if sep_condition():
                delta += sep_multiplier() * boid.separation(i, sep_neighbours, population)

        if len(ali_neighbours) > 0:
            if ali_condition():
                delta += ali_multiplier() * boid.alignment(i, ali_neighbours, population)

        population[i, 1] += delta

        population[i, 0] += population[i, 1]

    return population
