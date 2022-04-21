import numpy as np


def metric_neighbourhood(index, population, view_distance=50, neighbourhood_max=9999999):
    # generate the neighbourhood for this boid

    distances = np.linalg.norm(population[index][0] - population[:, 0], axis=1)  # generate distances to each other boid
    distances = np.array([[i, distances[i]] for i in range(len(population))])  # generate (index, distance) pairs
    distances = distances[np.argsort(distances[:, 1])][1:]  # sort on distance, discard first as that is self

    n = list()

    for j in range(len(population) - 1):
        # if boid is in view and we have room in the neighbourhood, and add it to the neighbourhood
        if distances[j][1] <= view_distance and j < neighbourhood_max:
            n.append(int(distances[j][0]))

    return np.array(n)
