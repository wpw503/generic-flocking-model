import math
import multiprocessing
import os
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from evaluation.boidlike import get_current_ali_measure, get_current_coh_measure, get_current_sep_measure
from evaluation.cohesion import is_broken
from evaluation.density import density_distribution
from models.generic_MLP_boid import generic_MLP_boid_step, MLP_boid
from models.generic_multi_MLP_boid import generic_multi_MLP_boid_step, multi_MLP_neighbour_boid, multi_MLP_strength_boid
from simulation.simulation import generate_initial_testing_population, test_world_constraints, \
    generate_initial_population

from scipy.spatial import Delaunay
from scipy import interpolate

from sklearn.linear_model import LinearRegression

import warnings

from config import *

from tools.MLP import MLP

warnings.filterwarnings("ignore")


def velocity_fluctuations(population):
    av_vel = np.average(population[:, 1, :], axis=0)
    # return population[:, 1, :] - av_vel
    ds = np.linalg.norm(population[:, 1, :] - av_vel, axis=1)
    vs = population[:, 1, :] - av_vel
    return np.array([vs[i] / ds[i] for i in range(len(ds))])


def vel_correlation(population, r):
    def smooth_delta_d(dist):
        return 1 / (1 + np.e ** (-(-(4 * dist) + 6)))

    # sizes = []
    # for i in range(len(population)):
    #     for j in range(i + 1, len(population)):
    #         sizes.append(np.abs(np.linalg.norm(population[i, 0, :] - population[j, 0, :])))
    #
    # size = round(max(sizes))
    #
    tops = []
    bottoms = []
    threshold = 30
    # mult = (np.logspace(1, 10, threshold, base=10) / 10)#[::-1]
    # mult -= mult[0]
    # mult = np.abs(mult)
    # mult += 1

    u = velocity_fluctuations(population)

    for i in range(len(population)):
        distances = np.linalg.norm(population[i][0] - population[:, 0], axis=1)
        distances = r - distances
        distances = np.abs(distances)

        for j in range(i + 1, len(population)):
            if distances[j] < threshold:  # -0.1 < distances[j] < 0.1: #
                diff = (u[i, 0] * u[j, 0]) + (u[i, 1] * u[j, 1])
                # pairs.append(diff)
                tops.append(diff * smooth_delta_d(distances[j]))
                bottoms.append(smooth_delta_d(distances[j]))

    if len(tops) > 0:
        return sum(tops) / sum(bottoms)
    else:
        return None

def scale_test_run(algorithm=None, name="NO_NAME",
                   save_to_file=True, verbose=True, optimiser=False,
                   sample_freq=10, num_samples=20, pop=30, pre_sim=900, tid="", just_boidlike=True):
    broken_this_run = 0
    broken = 0
    population = generate_initial_population(num_boids=pop)
    frame_number = 0
    density_scores = []
    boidlike_scores = []

    xs = []
    ys = []

    have_solution = False

    just_density = False

    for fn in range(pre_sim):
        population = algorithm(population)
        population = test_world_constraints(population)
        frame_number += 1

    num_sampled = 0

    while (num_sampled < num_samples):
        have_solution = False
        frame_number += 1

        population = algorithm(population)
        population = test_world_constraints(population)

        for b in range(len(population)):
            if math.isnan(population[b, 0, 0]) or math.isnan(population[b, 0, 1]) \
                    or math.isnan(population[b, 1, 0]) or math.isnan(population[b, 1, 1]):
                with open("tid-" + tid + ".csv", "w") as f:
                    # print("broke")
                    f.write("-100, -100, 9999999999999")
                return

        tris = Delaunay(population[:, 0, :])

        #ali = get_current_ali_measure(population, tris)
        sep = get_current_sep_measure(population, tris)
        #coh = get_current_coh_measure(population, tris)

        #boidlike_scores.append(ali + sep + coh)
        boidlike_scores.append(sep)

        if (not just_boidlike) and frame_number % sample_freq == 0:
            if is_broken(population):
                broken_this_run += 1
                broken += 1
                if broken_this_run > 10:
                    broken_this_run = 0
                    num_sampled -= 1

                if optimiser and broken > 50:
                    with open("tid-" + tid + ".csv", "w") as f:
                        # print("broke")
                        f.write("-100, -100, 9999999999999")
                    return

            else:
                dist = density_distribution(population)

                # tscores = []
                #
                # if len(dist) > 3:
                #     for i in range(1, len(dist)):
                #         # if dist[i] > dist[i - 1]:
                #         #     tscores.append(1)
                #         # else:
                #         #     tscores.append(-1)
                #
                #         tscores.append((dist[i - 1] - dist[i])/i)
                #
                #     if len(tscores) > 0:
                #         tscore = sum(tscores) #/ len(tscores)
                #
                #         if not math.isnan(tscore):
                #             density_scores.append(tscore)

                temp_dist = []

                if len(dist) > 5:
                    datapoints = 20

                    try:
                        x = dist
                        y = [i for i in range(len(x))]
                        f = interpolate.interp1d(y, x)

                        xnew = np.arange(0, len(x) - 1, (len(x) - 1) / datapoints)
                        if xnew[-1] > len(x) - 1:
                            xnew[-1] = len(x) - 1
                        ynew = f(xnew)

                        temp_dist = (ynew)
                    except:
                        pass

                    dist = np.array(temp_dist)

                    tscore = 0

                    for i in range(1, 20):
                        tscore += (dist[i - 1] - dist[i]) / i

                    density_scores.append(tscore)
                else:
                    density_scores.append(0)

                num_sampled += 1

                if just_density:
                    have_solution = True

                if not just_density:
                    sizes = []
                    for i in range(len(population)):
                        for j in range(i + 1, len(population)):
                            sizes.append(np.abs(np.linalg.norm(population[i, 0, :] - population[j, 0, :])))

                    size = round(max(sizes))

                    have_gone_positive = False
                    have_solution = False

                    crs = []
                    rs = []
                    crrs = []

                    num_since_zero = 0

                    for r in range(0, size, 1):
                        cr = vel_correlation(population, r)
                        crrs.append(cr)
                        rs.append(r)

                        if r < 0:
                            num_since_zero += 1
                        else:
                            num_since_zero = 0

                        if num_since_zero > 20:
                            break

                    for i in range(10, len(crrs) - 10):
                        temps = []

                        for v in crrs[i - 10:i + 10]:
                            if v is not None:
                                temps.append(v)

                        crs.append([np.average(temps), rs[i]])

                    for cr, r in crs:
                        if cr > 0.01:
                            have_gone_positive = True
                        if have_gone_positive and (cr is not None) and (
                                math.isclose(cr, 0, rel_tol=0, abs_tol=1e-2) or cr < 0):
                            ys.append(r)
                            xs.append(size)

                            have_solution = True

                            break


    if len(boidlike_scores) == 0:
        with open("tid-" + tid + ".csv", "w") as f:
            # print("no_solution")
            f.write("-100, -100, 9999999999999")
        return

    if just_density:
        with open("tid-" + tid + ".csv", "w") as f:
            f.write(str(0.5 * np.average(density_scores)) + "," + str(np.average(boidlike_scores)) + ",9999999999999")
            return

    if just_boidlike:
        with open("tid-" + tid + ".csv", "w") as f:
            f.write("0," + str(np.average(boidlike_scores)) + ",9999999999999")
            return

    if not have_solution or len(xs) < 15:
        with open("tid-" + tid + ".csv", "w") as f:
            # print("no_solution")
            f.write("-100, -100, 9999999999999")
        return

    if len(xs) > 0:
        a, b = np.polyfit(xs, ys, 1)

        with open("tid-" + tid + ".csv", "w") as f:
            f.write(str(min(a, 0.8) + min(np.average(density_scores), 1.8)) +
                    "," + str(np.average(boidlike_scores)) +
                    ",0")
            return

    with open("tid-" + tid + ".csv", "w") as f:
        # print("got to end")
        f.write("-100, -100, 9999999999999")
    return


def scale_test(algorithm, name="NO_NAME",
               save_to_file=True, verbose=True, optimiser=False,
               sample_freq=10, num_samples=20, num_runs=10, num_pop=30, just_boidlike=False,
               individual=None, model=None, broken_test=True):
    pre_sim = 900

    if optimiser:
        num_runs = multiprocessing.cpu_count() - 6
        proc_pool = []

        if individual is not None:
            for i in range(num_runs):
                if model == "MLP":
                    nnetwork = MLP_boid()
                    nnetwork.set_weights(individual)
                    algorithm = partial(generic_MLP_boid_step, nnet=nnetwork)

                if model == "multi_MLP":
                    coh_neighbourhood_nnet = multi_MLP_neighbour_boid()
                    sep_neighbourhood_nnet = multi_MLP_neighbour_boid()
                    ali_neighbourhood_nnet = multi_MLP_neighbour_boid()
                    coh_nnet = multi_MLP_strength_boid()
                    sep_nnet = multi_MLP_strength_boid()
                    ali_nnet = multi_MLP_strength_boid()

                    running_index = 0

                    coh_neighbourhood_nnet.set_weights(individual[0:len(coh_neighbourhood_nnet.get_weights())])
                    running_index += len(coh_neighbourhood_nnet.get_weights())
                    sep_neighbourhood_nnet.set_weights(individual[running_index:running_index+len(sep_neighbourhood_nnet.get_weights())])
                    running_index += len(sep_neighbourhood_nnet.get_weights())
                    ali_neighbourhood_nnet.set_weights(individual[running_index:running_index+len(ali_neighbourhood_nnet.get_weights())])
                    running_index += len(ali_neighbourhood_nnet.get_weights())

                    coh_nnet.set_weights(individual[running_index:running_index+len(coh_nnet.get_weights())])
                    running_index += len(coh_nnet.get_weights())
                    sep_nnet.set_weights(individual[running_index:running_index + len(sep_nnet.get_weights())])
                    running_index += len(sep_nnet.get_weights())
                    ali_nnet.set_weights(individual[running_index:running_index + len(ali_nnet.get_weights())])

                    algorithm = partial(generic_multi_MLP_boid_step,
                                        coh_neighbourhood_nnet=coh_neighbourhood_nnet,
                                        sep_neighbourhood_nnet=sep_neighbourhood_nnet,
                                        ali_neighbourhood_nnet=ali_neighbourhood_nnet,
                                        coh_nnet=coh_nnet,
                                        sep_nnet=sep_nnet,
                                        ali_nnet=ali_nnet,
                                        )


                proc_pool.append(
                    multiprocessing.Process(target=scale_test_run, args=(algorithm, "NO_NAME",
                                                                         True, True, True,
                                                                         10, 20, 30, 900, str(i), just_boidlike)))
                proc_pool[i].start()
        else:
            for i in range(num_runs):
                proc_pool.append(
                    multiprocessing.Process(target=scale_test_run, args=(algorithm, "NO_NAME",
                                                                         True, True, True,
                                                                         10, 20, 30, 900, str(i), just_boidlike)))
                proc_pool[i].start()

        for i in range(num_runs):
            proc_pool[i].join()

        temp = []
        for i in range(num_runs):
            if os.path.exists("tid-" + str(i) + ".csv"):
                temp.append(np.loadtxt("tid-" + str(i) + ".csv", delimiter=","))
                os.remove("tid-" + str(i) + ".csv")

        averages = np.mean(temp, axis=0)

        if len(averages) == 3:
            return averages[0] + 0.05 * (averages[1]), averages[2]
        # print(temp)

        return averages[0] + 0.1 * (averages[1]), 9999999999

    xs = []
    ys = []

    density_scores = []

    num_broken = 0
    current_c = -1
    past_c = []

    have_solution = False

    if save_to_file:
        open("stats/density_dists/density_" + name.replace(" ", "") + ".txt", 'w').close()
        open("stats/scale_outs/scale_" + name.replace(" ", "") + ".csv", 'w').close()

    for pop in [num_pop]:
        run = 0
        while run < num_runs:
            run += 1
            broken_this_run = 0
            population = generate_initial_population(num_boids=pop)
            # run_history = [population.copy()]
            frame_number = 0

            for fn in range(pre_sim):
                if fn % 10 == 0:
                    if verbose:
                        print("\r" + "pop " + str(pop) + ", run " + str(run) + ", frame " + str(
                            fn) + ", (pre-sim)" + ", broken (" + str(num_broken) + ")" + ", c (" + str(
                            current_c) + "), density (" + str(np.mean(density_scores)) + ")" + "\t\t\t", end="")

                population = algorithm(population)
                population = test_world_constraints(population)
                # run_history.append(population.copy())
                frame_number += 1

            num_sampled = 0

            while (num_sampled < num_samples) and (broken_this_run <= 20):
                frame_number += 1
                if verbose:
                    printing = "\r" + "pop " + str(pop) + ", run " + str(run) + ", frame " + str(
                        frame_number) + ", broken (" + str(num_broken) + ")" + ", c (" + str(
                        current_c) + "), density (" + str(np.mean(density_scores)) + ")"
                    print(printing + "\t\t\t", end="")

                population = algorithm(population)
                population = test_world_constraints(population)

                # run_history.append(population.copy())

                if (not broken_test) or (not is_broken(population)):
                    dist = density_distribution(population)

                    if save_to_file:
                        with open("stats/density_dists/density_" + name.replace(" ", "") + ".txt", "a") as f:
                            buffer = ""
                            for d in dist:
                                buffer += str(d) + ","

                            f.write(buffer[:-1] + "\n")

                    # tscores = []
                    #
                    # if len(dist) > 3:
                    #     for i in range(1, len(dist)):
                    #         # if dist[i] > dist[i - 1]:
                    #         #     tscores.append(1)
                    #         # else:
                    #         #     tscores.append(-1)
                    #
                    #         tscores.append((dist[i - 1] - dist[i])/i)
                    #
                    #     if len(tscores) > 0:
                    #         tscore = sum(tscores)# / len(tscores)
                    #
                    #         if not math.isnan(tscore):
                    #             density_scores.append(tscore)

                    temp_dist = []

                    if len(dist) > 5:
                        datapoints = 20

                        try:
                            x = dist
                            y = [i for i in range(len(x))]
                            f = interpolate.interp1d(y, x)

                            xnew = np.arange(0, len(x) - 1, (len(x) - 1) / datapoints)
                            if xnew[-1] > len(x) - 1:
                                xnew[-1] = len(x) - 1
                            ynew = f(xnew)

                            temp_dist = (ynew)
                        except:
                            pass

                        dist = np.array(temp_dist)

                        tscore = 0

                        for i in range(1, 20):
                            tscore += (dist[i-1] - dist[i]) / i

                        density_scores.append(tscore)

                if frame_number % sample_freq == 0:

                    if is_broken(population) and broken_test:
                        num_broken += 1
                        broken_this_run += 1
                        if broken_this_run > 20:
                            num_sampled -= 1

                        if optimiser and num_broken > 100:
                            return -100, -100

                    else:
                        num_sampled += 1

                        sizes = []
                        for i in range(len(population)):
                            for j in range(i + 1, len(population)):
                                sizes.append(np.abs(np.linalg.norm(population[i, 0, :] - population[j, 0, :])))

                        size = round(max(sizes))

                        have_gone_positive = False
                        have_solution = False

                        crs = []
                        rs = []
                        crrs = []

                        for r in range(0, size, 1):
                            if verbose:
                                print(printing + " -> " + "{:.2%}".format(r / size), end="")
                            cr = vel_correlation(population, r)
                            crrs.append(cr)
                            rs.append(r)

                        for i in range(10, len(crrs) - 10):
                            temps = []

                            for v in crrs[i - 10:i + 10]:
                                if v is not None:
                                    temps.append(v)

                            crs.append([np.average(temps), rs[i]])

                        for cr, r in crs:
                            if cr > 0.01:
                                have_gone_positive = True
                            if have_gone_positive and (cr is not None) and (
                                    math.isclose(cr, 0, rel_tol=0, abs_tol=1e-2) or cr < 0):

                                ys.append(r)
                                xs.append(size)

                                if save_to_file:
                                    with open("stats/scale_outs/scale_" + name.replace(" ", "") + ".csv", "a") as f:
                                        f.write(str(size) + "," + str(r) + "\n")

                                have_solution = True

                                break

                        # plt.scatter(np.array(crs)[:, 1], np.array(crs)[:, 0])
                        # plt.show()

            if not have_solution:
                num_broken += 1
                broken_this_run += 1
                run -= 1

            # plt.scatter(xs, ys, marker="x")
            # plt.show()

            # with open("replays/" + name.replace(" ", "") + "_" + f"{xs[-1]:02d}" + "-" + f"{ys[-1]:02d}" + ".csv", "a") as f:
            #    for timestep in run_history:
            #        buffer = ""
            #        for agent in timestep:
            #            buffer += (str(agent[0, 0]) + " " + str(agent[0, 1]) + " " + str(agent[1, 0]) + " " + str(agent[1, 1]) + ",")
            #        f.write(buffer[:-1] + "\n")

            if len(xs) > 0:
                a, b = np.polyfit(xs, ys, 1)

                current_c = a
                past_c.append(a)

                # plt.plot(range(len(past_c)), past_c)
                # plt.show()

            # print(printing + " c=" + str(a))

        if save_to_file:
            with open("stats/density_outs/density_" + name.replace(" ", "") + ".txt", "w") as f:
                buffer = ""
                for d in density_scores:
                    buffer += str(d) + ","

                f.write(buffer[:-1])

        if verbose:
            print("density", np.mean(density_scores))

    return a, np.mean(density_scores)
