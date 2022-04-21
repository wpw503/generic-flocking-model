import numpy as np


def polarisation(population):
    # average of all normalised velocity vectors
    #norm_v = population[:, 1] / np.linalg.norm(population[:, 1], axis=1)
    velocities = []

    for i in range(len(population)):
        velocities.append(population[i, 0] / np.linalg.norm(population[i, 0]))

    if len(velocities) > 0:
        av_vel = sum(velocities) / len(velocities)

        return np.linalg.norm(av_vel)

    else:
        return 0
