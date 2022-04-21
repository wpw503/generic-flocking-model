import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from evaluation.polarisation import polarisation
from simulation.simulation import test_world_constraints, generate_initial_testing_population


def count_collisions(population):
    collisions = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            if np.linalg.norm(population[i, 0] - population[j, 0]) < 5:
                collisions += 1

    return collisions


def flock_size(population):
    sizes = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            sizes.append(np.abs(np.linalg.norm(population[i, 0, :] - population[j, 0, :])))

    return max(sizes)


def average_velocity(population):
    av_vel = np.array([.0, .0])

    for b in population:
        av_vel += b[1]

    return av_vel / len(population)


def average_speed(population):
    av_vel = 0

    for b in population:
        av_vel += np.linalg.norm(b[1])

    return av_vel / len(population)


def average_distance(population):
    sizes = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            sizes.append(np.abs(np.linalg.norm(population[i, 0, :] - population[j, 0, :])))

    return sum(sizes) / len(sizes)


def num_flocks(population):
    scores = [0.5]

    for k in range(2, 8):
        model = KMeans(n_clusters=k)
        model.fit(population[:, 0])
        pred = model.predict(population[:, 0])
        score = silhouette_score(population[:, 0], pred)
        scores.append(score)

    return scores.index(max(scores)) + 1


def evaluate_flocking(algorithm, name="NO_NAME", verbose=True, save_to_file=True):
    pre_sim = 900
    test_sim = 300

    all_stat_collisions = []
    all_stat_polarisation = []
    all_stat_average_distance = []
    all_stat_size = []
    all_stat_average_velocity = []
    all_stat_average_speed = []
    all_stat_num_flocks = []

    for run in range(10):
        stat_collisions = []
        stat_polarisation = []
        stat_average_distance = []
        stat_size = []
        stat_average_velocity = []
        stat_average_speed = []
        stat_num_flocks = []

        population = generate_initial_testing_population(num_boids=30)

        for fn in range(pre_sim):
            if fn % 10 == 0:
                if verbose:
                    print("\r" + "pop " + str(30) + ", run " + str(run) + ", frame " + str(
                        fn) + ", (pre-sim)" + "\t\t\t", end="")

            population = algorithm(population)
            population = test_world_constraints(population)

        for fn in range(test_sim):
            if fn % 10 == 0:
                if verbose:
                    print("\r" + "pop " + str(30) + ", run " + str(run) + ", frame " + str(
                        pre_sim + fn) + ", (testing)" + "\t\t\t", end="")

            population = algorithm(population)
            population = test_world_constraints(population)

            stat_collisions.append(count_collisions(population))
            stat_polarisation.append(polarisation(population))
            stat_average_distance.append(average_distance(population))
            stat_size.append(flock_size(population))
            stat_average_velocity.append(average_velocity(population))
            stat_average_speed.append(average_speed(population))
            stat_num_flocks.append(num_flocks(population))

        all_stat_collisions.append(stat_collisions)
        all_stat_polarisation.append(stat_polarisation)
        all_stat_average_distance.append(stat_average_distance)
        all_stat_size.append(stat_size)
        all_stat_average_velocity.append(stat_average_velocity)
        all_stat_average_speed.append(stat_average_speed)
        all_stat_num_flocks.append(stat_num_flocks)

    if save_to_file:
        with open("stats/collisions/collisions_" + name.replace(" ", "") + ".csv", "w") as f:
            for frame in all_stat_collisions:
                buffer = ""
                for v in frame:
                    buffer += str(v) + ","

                f.write(buffer[:-1] + "\n")

        with open("stats/polarisation/polarisation_" + name.replace(" ", "") + ".csv", "w") as f:
            for frame in all_stat_polarisation:
                buffer = ""
                for v in frame:
                    buffer += str(v) + ","

                f.write(buffer[:-1] + "\n")

        with open("stats/average_distance/average_distance_" + name.replace(" ", "") + ".csv", "w") as f:
            for frame in all_stat_average_distance:
                buffer = ""
                for v in frame:
                    buffer += str(v) + ","

                f.write(buffer[:-1] + "\n")

        with open("stats/size/size_" + name.replace(" ", "") + ".csv", "w") as f:
            for frame in all_stat_size:
                buffer = ""
                for v in frame:
                    buffer += str(v) + ","

                f.write(buffer[:-1] + "\n")

        with open("stats/average_velocity/average_velocity_" + name.replace(" ", "") + ".csv", "w") as f:
            for frame in all_stat_average_velocity:
                buffer = ""
                for v in frame:
                    buffer += str(v) + ","

                f.write(buffer[:-1] + "\n")

        with open("stats/average_speed/average_speed_" + name.replace(" ", "") + ".csv", "w") as f:
            for frame in all_stat_average_speed:
                buffer = ""
                for v in frame:
                    buffer += str(v) + ","

                f.write(buffer[:-1] + "\n")

        with open("stats/num_flocks/num_flocks_" + name.replace(" ", "") + ".csv", "w") as f:
            for frame in all_stat_num_flocks:
                buffer = ""
                for v in frame:
                    buffer += str(v) + ","

                f.write(buffer[:-1] + "\n")
