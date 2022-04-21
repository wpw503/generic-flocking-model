import numpy as np
from scipy.spatial import Delaunay

from evaluation import polarisation
from evaluation.density import generate_rings
from models import boid
from neighbourhoods.delaunay import delaunay_neighbourhood
from neighbourhoods.metric import metric_neighbourhood
from neighbourhoods.nearest_n import nearest_n_neighbourhood


def dmbsmf_boid_step(population):
    # def f_i(ring_index):
    #     return 0.5021993109889702 ** (0.6146870503444409 * (ring_index / max(rings)))


    def f_i(ring_index):
        if (ring_index - 1) == 0:
            return 0.97
        return -0.944 * (ring_index - 1) + (1-0.944)

    population = population.copy()

    tris = Delaunay(population[:, 0, :])
    rings = generate_rings(population)

    for i in range(len(population)):
        neighbours = delaunay_neighbourhood(i, population, tris=tris)
        ring_neighbours = [n for n in neighbours if rings[n] == rings[i]]

        delta = np.array([0., 0.])

        if len(neighbours) > 0:
            delta += (1 - f_i(rings[i])) * boid.alignment(i, neighbours, population, strength=5)
            delta += ((1 - f_i(rings[i])) + 0.5) * boid.separation(i, neighbours, population, strength=.1)

        if len(ring_neighbours) > 0:
            delta += (f_i(rings[i])) * boid.cohesion(i, ring_neighbours, population, strength=.1)

        population[i, 1] += 2 * delta

        population[i, 0] += population[i, 1]

    return population
