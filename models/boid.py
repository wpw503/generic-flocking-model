import numpy as np

from neighbourhoods.delaunay import delaunay_neighbourhood
from neighbourhoods.metric import metric_neighbourhood

from scipy.spatial import Delaunay


def cohesion(index, neighbours, population, strength=.5):
    center = np.array([0., 0.])

    # for each boid in the neighbourhood
    for b in population[neighbours]:
        center += b[0]

    center /= len(neighbours)

    return ((center - population[index][0]) / 100) * strength


def separation(index, neighbours, population, distance=15, strength=3):
    c = np.array([0., 0.])

    # for each boid in the neighbourhood
    for b in population[neighbours]:
        if np.linalg.norm(b[0] - population[index][0]) < distance:
            c -= (b[0] - population[index][0])

    return (c / 100) * strength


def alignment(index, neighbours, population, strength=1):
    average_velocity = np.array([0., 0.])

    # for each boid in the neighbourhood
    for b in population[neighbours]:
        average_velocity += b[1]

    average_velocity /= len(neighbours)

    return ((average_velocity - population[index][1]) / 100) * strength


def boid_step(population, test=False):
    population = population.copy()

    tris = Delaunay(population[:, 0, :])

    for i in range(len(population)):
        if test:
            # if test is true - stay in one flock
            neighbours = delaunay_neighbourhood(i, population, tris=tris)
        else:
            neighbours = metric_neighbourhood(i, population)

        if len(neighbours) > 0:
            delta = np.array([0., 0.])

            if test:
                delta += cohesion(i, neighbours, population, strength=.1)
                delta += separation(i, neighbours, population, strength=1)
                delta += alignment(i, neighbours, population, strength=.1)
            else:
                delta += cohesion(i, neighbours, population)
                delta += separation(i, neighbours, population)
                delta += alignment(i, neighbours, population)

            population[i, 1] += delta

        population[i, 0] += population[i, 1]

    return population
