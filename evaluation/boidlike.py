import numpy as np
from scipy.spatial import Delaunay

from evaluation.polarisation import polarisation


def get_current_coh_measure(population, tris=None):
    # average distance to another in neighbourhood
    def get_neighbor_vertex_ids_from_vertex_id(vertex_id, tri):
        index_pointers, indices = tri.vertex_neighbor_vertices
        result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
        return result_ids

    if tris is None:
        tris = Delaunay(population[:, 0, :])

    sis = []

    for index in range(len(population)):
        n = get_neighbor_vertex_ids_from_vertex_id(index, tris)

        dists = []

        for ni in n:
            dists.append(np.linalg.norm(population[index, 0] - population[ni, 0]))

        if len(dists) > 0:
            sis.append(sum(dists) / len(dists))

    if len(sis) > 0:
        # try and maximise, aka get as close to 0 as possible
        return -((sum(sis) / len(sis))/35)

    return -100


def get_current_ali_measure(population, tris=None):
    # between 0 and 1
    return 2 * polarisation(population)


def get_current_sep_measure(population, tris=None):
    # negate for every boid too close
    def get_neighbor_vertex_ids_from_vertex_id(vertex_id, tri):
        index_pointers, indices = tri.vertex_neighbor_vertices
        result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
        return result_ids

    if tris is None:
        tris = Delaunay(population[:, 0, :])

    sis = []

    for index in range(len(population)):
        n = get_neighbor_vertex_ids_from_vertex_id(index, tris)

        dists = []

        for ni in n:
            if (np.linalg.norm(population[index, 0] - population[ni, 0])) < 4:
                sis.append(-1)

    if len(sis) > 0:
        # try and maximise, aka get as close to 0 as possible
        return sum(sis)

    return 0
