import sys
import os
import sys

import numpy as np

from config import DEFAULT_IMAGE_SIZE
from models.boid import boid_step
from simulation.simulation import generate_initial_population, display_world_constraints

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


def get_rotation(v):
    """
    Returns the direction (bearing in degrees) of a 2D vector.
    :param v: 2D numpy vector
    :return: Bearing of v in degrees
    """
    if v[1] != 0:
        rot = np.rad2deg(np.arctan(v[0] / v[1]))
    else:
        return 0 if v[1] < 0 else 1

    if v[1] < 0:
        rot = -180 + rot

    return rot


def main():
    world_width = 900
    world_height = 900

    pygame.init()
    window = pygame.display.set_mode((world_width, world_height))

    clock = pygame.time.Clock()

    boid_image = pygame.image.load("boid.png")
    boid_image = pygame.transform.scale(boid_image, DEFAULT_IMAGE_SIZE)

    population = generate_initial_population(num_boids=40)

    frame_number = 0

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # Set background color to our window
        window.fill("black")

        # UPDATE POPULATION

        population = boid_step(population)
        population = display_world_constraints(population, world_width, world_height)

        for i in range(len(population)):
            window.blit(pygame.transform.rotate(boid_image, get_rotation(population[i][1])), population[i][0])

        # Display world
        pygame.display.flip()

        ### Uncomment to save frame renders ###
        # pygame.image.save(window, "./frames/" + f"{frame_number:06d}" + ".png")
        # if frame_number > 30 * 60 * 3:
        #    sys.exit()

        frame_number += 1

        clock.tick(30)


if __name__ == "__main__":
    main()
