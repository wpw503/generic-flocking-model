import copy
import math
import os.path
import pickle
import random
import time
from datetime import datetime, timezone

import numpy as np

from deap import base
from deap import creator
from deap import tools

from evaluation.scale import scale_test
from models.generic_MLP_boid import generic_MLP_boid_step, MLP_boid
from models.generic_multi_MLP_boid import multi_MLP_mate, generic_multi_MLP_boid_step, multi_MLP_neighbour_boid, \
    multi_MLP_strength_boid


np.set_printoptions(suppress=True)


def evaluation_function(individual, model="NONE", record_times=True):
    if model not in ["MLP", "multi_MLP"]:
        raise ValueError("evaluation function: model not available")

    tic = time.perf_counter()
    if model == "MLP":
        a, b = scale_test(generic_MLP_boid_step, individual=individual, model=model,
                       verbose=False, save_to_file=False, optimiser=True)

    if model == "multi_MLP":
        a, b = scale_test(generic_multi_MLP_boid_step, individual=individual, model=model,
                       verbose=False, save_to_file=False, optimiser=True)

    if math.isnan(a):
        a = -100

    if record_times:
        with open("stats/ea_training/" + model + "/train_times.csv", "a") as f:
            f.write(str(time.perf_counter() - tic) + ",")

    return a,


def main(checkpoint=None, model="MLP", max_gen=500, mutation_sigma=0.5, num_pop=50):
    if model == "MLP":
        individual_size = len(MLP_boid().get_weights())

    if model == "multi_MLP":
        individual_size = len(multi_MLP_neighbour_boid().get_weights())
        individual_size += len(multi_MLP_neighbour_boid().get_weights())
        individual_size += len(multi_MLP_neighbour_boid().get_weights())
        individual_size += len(multi_MLP_strength_boid().get_weights())
        individual_size += len(multi_MLP_strength_boid().get_weights())
        individual_size += len(multi_MLP_strength_boid().get_weights())

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=individual_size)
    toolbox.register("evaluate", evaluation_function)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=mutation_sigma, indpb=0.1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if model == "multi_MLP":
        toolbox.register("mate", multi_MLP_mate)

    if checkpoint:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"] + 1
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        best = cp["best"]
        recorded_best = True

        print("loading from checkpoint")
    else:
        # Start a new evolution
        population = toolbox.population(n=num_pop)
        start_gen = 0
        halloffame = tools.HallOfFame(5)
        logbook = tools.Logbook()
        best = None
        recorded_best = False

    if start_gen >= max_gen:
        return

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("evaluating population before first generation...")
    fitnesses = [toolbox.evaluate(indiv, model, False) for indiv in population]
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    num_random = 0
    num_best = 0

    NGEN = max_gen
    checkpoint_freq = 1

    for g in range(start_gen, NGEN):
        gen_start_time = time.perf_counter()

        print("\r-- Generation %i --" % (g + 1), end="")

        if best is None or tools.selBest(population, 1)[0].fitness.values > best.fitness.values:
            best = tools.selBest(population, 1)[0]
            recorded_best = False

        if not recorded_best:
            with open("stats/ea_training/" + model + "/" + "best_" + str(best.fitness.values[0]) + ".csv", "w") as f:
                buffer = ""

                for v in best:
                    buffer += str(v) + ","

                f.write(buffer[:-1] + "\n")

            recorded_best = True

        offspring = toolbox.select(population, num_pop - num_random - num_best)
        #offspring += tools.selBest(population, num_best)
        #offspring += toolbox.population(n=num_random)

        offspring = list(map(toolbox.clone, offspring))

        if model != "MLP":
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(indiv, model) for indiv in invalid_ind]

        print("\r-- Generation %i --" % (g + 1), " average: ",
              np.around(np.average(fitnesses), decimals=4),
              " max: ", np.around(np.max(fitnesses), decimals=4),
              " best: ", np.around(best.fitness.values, decimals=4), "                                ")

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        record = stats.compile(population)
        logbook.record(gen=g, **record)
        halloffame.update(population)

        if g % checkpoint_freq == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=g, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate(), best=best)

            with open("stats/ea_training/" + model + "_checkpoint.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

        with open("stats/ea_training/" + model + "/generation_times.csv", "a") as f:
            f.write(str(time.perf_counter() - gen_start_time) + "\n")
        with open("stats/ea_training/" + model + "/train_times.csv", "a") as f:
            f.write("\n")

    logbook.header = "gen", "avg", "evals", "std", "min", "max"

    gen = logbook.select("gen")
    avgs = logbook.select("avg")
    stds = logbook.select("std")
    maxs = logbook.select("max")

    utctime = datetime.now(timezone.utc).strftime("%Y%m%d%H%S")

    if best is not None:
        with open("stats/ea_training/" + model + "/" + str(utctime) + "_best_" + str(best.fitness.values[0]) + ".csv", "w") as f:
            buffer = ""

            for v in best:
                buffer += str(v) + ","

            f.write(buffer[:-1] + "\n")

    with open("stats/ea_training/" + model + "/" + str(utctime) + "_population" + ".csv", "w") as f:
        for ind in population:
            buffer = ""

            buffer += str(ind.fitness.values[0]) + ","

            for v in ind:
                buffer += str(v) + ","

            f.write(buffer[:-1] + "\n")

    with open("stats/ea_training/" + model + "/" + str(utctime) + "_hof" + ".csv", "w") as f:
        for ind in halloffame:
            buffer = ""

            buffer += str(ind.fitness.values[0]) + ","

            for v in ind:
                buffer += str(v) + ","

            f.write(buffer[:-1] + "\n")

    with open("stats/ea_training/" + model + "/" + str(utctime) + "_avgs.csv", "a") as f:
        buffer = ""

        for v in avgs:
            buffer += str(v) + ","

        f.write(buffer[:-1] + "\n")

    with open("stats/ea_training/" + model + "/" + str(utctime) + "_maxs.csv", "a") as f:
        buffer = ""

        for v in maxs:
            buffer += str(v) + ","

        f.write(buffer[:-1] + "\n")


if __name__ == "__main__":
    runs = [30, 60, 90]
    sigmas = [0.5, 0.1, 0.1]
    pops = [50, 50, 30]

    for i in range(len(runs)):
        for meval in ["MLP", "multi_MLP"]:  # , "MLP_delta"]:
            print("-" * 7, "evaluating", meval, "for", runs[i], "runs, sigma=" + str(sigmas[i]), "pops=" + str(pops[i]),
                  "-" * 7)
            if os.path.exists("stats/ea_training/" + meval + "_checkpoint.pkl"):
                main(checkpoint="stats/ea_training/" + meval + "_checkpoint.pkl",
                     model=meval, max_gen=runs[i], mutation_sigma=sigmas[i], num_pop=pops[i])
            else:
                main(model=meval, max_gen=runs[i], mutation_sigma=sigmas[i], num_pop=pops[i])
