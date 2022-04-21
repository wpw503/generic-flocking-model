import numpy as np
from scipy.spatial import Delaunay

from evaluation import polarisation
from evaluation.density import generate_rings
from models import boid
from neighbourhoods.delaunay import delaunay_neighbourhood
from neighbourhoods.metric import metric_neighbourhood
from neighbourhoods.nearest_n import nearest_n_neighbourhood


def mti_boid_step(population, state_list):
    population = population.copy()

    a_threshold = 0.99
    b_threshold = 0.95

    tris = Delaunay(population[:, 0, :])
    rings = generate_rings(population)

    for i in range(len(population)):
        metric_neighbours = metric_neighbourhood(i, population, view_distance=100)
        topological_neighbours = nearest_n_neighbourhood(i, population)

        neighbours = delaunay_neighbourhood(i, population, tris=tris)
        ring_neighbours = [n for n in neighbours if rings[n] == rings[i]]

        if len(metric_neighbours) > 0:
            delta = np.array([0., 0.])

            if state_list[i] != 0:
                if polarisation.polarisation(
                        population[np.append(np.random.choice(metric_neighbours), i).astype(int)]) < b_threshold:
                    state_list[i] = 0
                else:
                    delta += boid.alignment(i, metric_neighbours, population, strength=10)
                    delta += boid.separation(i, metric_neighbours, population, strength=.1)

                    if len(ring_neighbours) > 0:
                        delta += boid.cohesion(i, ring_neighbours, population, strength=0.1)

            else:
                if polarisation.polarisation(population[np.append(neighbours, i)]) > a_threshold:
                    state_list[i] = 1
                else:
                    delta += boid.alignment(i, neighbours, population, strength=10)

            population[i, 1] += delta

        population[i, 0] += population[i, 1]

    return population
