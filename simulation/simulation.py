import numpy as np

from config import WORLD_WIDTH, WORLD_HEIGHT


def generate_initial_population(num_boids=50, spacing=20, world_width=None, world_height=None):
    """
    Generate the starting positions and directions for all boids in the simulation.
    :return: A numpy array with shape (NUM_BOIDS, 2, 2) that is indexed as
    [boid index][displacement (0) or velocity (1)][x component (0) or y component (1)]
    """
    if world_height is None:
        world_width = WORLD_WIDTH
        world_height = WORLD_HEIGHT

    population = np.empty(shape=(num_boids, 2, 2))

    temp = np.sqrt(num_boids)


    for i in range(len(population)):
        # get a random displacement vector in range
        population[i][0] = np.array(np.random.random(2)) * np.array([world_width // 2, world_height // 2])
        population[i][0] += np.array([world_width // 4, world_height // 4])
        # get a random velocity vector with length in range [-5, 5]
        population[i][1] = (np.array(np.random.random(2)) - np.array([0.5, 0.5])) * 2
        population[i][1] /= np.linalg.norm(population[i][1])

        # population[i][0] = np.array([WORLD_WIDTH // 2, WORLD_HEIGHT // 2])
        # population[i][0] -= np.array([(temp * spacing) // 2, (temp * spacing) // 2])
        # population[i][0] += np.array([spacing * (i // temp), spacing * (i % temp)])
        # # population[i][1] = np.array([1, 1]) + (np.random.random(2) * 0.2)

    return population


def generate_initial_testing_population(num_boids=50, spacing=25, world_width=None, world_height=None):
    """
    Generate the starting positions and directions for all boids in the simulation.
    :return: A numpy array with shape (NUM_BOIDS, 2, 2) that is indexed as
    [boid index][displacement (0) or velocity (1)][x component (0) or y component (1)]
    """
    if world_height is None:
        world_width = WORLD_WIDTH
        world_height = WORLD_HEIGHT

    population = np.empty(shape=(num_boids, 2, 2))

    temp = np.sqrt(num_boids)

    for i in range(len(population)):
        # get a random displacement vector in range
        # population[i][0] = np.array(np.random.random(2)) * np.array([WORLD_WIDTH // 2, WORLD_HEIGHT // 2])
        # population[i][0] += np.array([WORLD_WIDTH // 4, WORLD_HEIGHT // 4])
        # # get a random velocity vector with length in range [-5, 5]
        #population[i][1] = (np.array(np.random.random(2)) - np.array([0.5, 0.5])) * 5

        population[i][0] = np.array([world_width // 2, world_height // 2])
        population[i][0] -= np.array([(temp * spacing) // 2, (temp * spacing) // 2])
        population[i][0] += np.array([spacing * (i // temp), spacing * (i % temp)])
        population[i][1] = np.array([1, 1]) + (np.random.random(2) * 0.2)
        population[i][1] /= np.linalg.norm(population[i][1])

    return population


def display_world_constraints(population, world_width=None, world_height=None):
    edge_pushback = 5
    speed = 3

    if world_height is None:
        world_width = WORLD_WIDTH
        world_height = WORLD_HEIGHT

    for i in range(len(population)):
        # if the boid is out of bounds, nudge it back in
        if population[i][0][0] > world_width:
            population[i][1][0] = -edge_pushback
        if population[i][0][1] > world_height:
            population[i][1][1] = -edge_pushback
        if population[i][0][0] < 0:
            population[i][1][0] = edge_pushback
        if population[i][0][1] < 0:
            population[i][1][1] = edge_pushback

        if np.linalg.norm(population[i][1]) > speed:
            population[i][1] = (population[i][1] / np.linalg.norm(population[i][1])) * speed

    return population


def test_world_constraints(population, world_width=None, world_height=None):
    speed = 3

    if world_height is None:
        world_width = WORLD_WIDTH
        world_height = WORLD_HEIGHT

    for i in range(len(population)):
        population[:, 0, :] -= np.average(population[:, 0, :], axis=0) / 4
        population[:, 0, :] += np.array([world_width // 2, world_height // 2]) / 4

        population[i][0][0] %= world_width
        population[i][0][1] %= world_height

        if np.linalg.norm(population[i][1]) > speed:
            population[i][1] = (population[i][1] / np.linalg.norm(population[i][1])) * speed

    return population
