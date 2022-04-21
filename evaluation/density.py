import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from config import WORLD_WIDTH, WORLD_HEIGHT


def generate_rings(population, tris=None):
    def get_neighbor_vertex_ids_from_vertex_id(vertex_id, tri):
        index_pointers, indices = tri.vertex_neighbor_vertices
        result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
        return result_ids

    def classify_rings(tris, rings):
        current_ring = max(rings)

        for i in range(len(rings)):
            if rings[i] == current_ring:
                ns = get_neighbor_vertex_ids_from_vertex_id(i, tris)
                for n in ns:
                    if rings[n] == -1:
                        rings[n] = current_ring + 1

        if min(rings) == -1:
            classify_rings(tris, rings)

    # rings is a list where rings[i] is the ring of boid i
    temp_population = population.copy().tolist()
    for x in range(-100, WORLD_WIDTH + 100, 100):
        for y in [-100, 100]:
            temp_population.append([[x, y], [0, 0]])

    for y in range(-100, WORLD_HEIGHT + 100, 100):
        for x in [-100, 100]:
            temp_population.append([[x, y], [0, 0]])

    temp_population = np.array(temp_population)

    if tris is None:
        tris = Delaunay(temp_population[:, 0, :])

    over_rings = [-1 for _ in range(len(population))] + [0 for _ in range(len(temp_population) - len(population))]

    classify_rings(tris, over_rings)

    rings = over_rings[:len(population)]

    return rings


def density_distribution(population, rings=None, tris=None):
    def get_neighbor_vertex_ids_from_vertex_id(vertex_id, tri):
        index_pointers, indices = tri.vertex_neighbor_vertices
        result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
        return result_ids

    #tris = Delaunay(population[:, 0, :])

    if rings is None:
        rings = generate_rings(population, tris)

    ring_avgs = []

    # for ring_index in range(1, max(rings) + 1):
    #     dists = []
    #     for i in range(len(rings)):
    #         if rings[i] == ring_index:
    #             ns = boid.neighbourhood(i, population, view_distance=20, neighbourhood_max=99999)#get_neighbor_vertex_ids_from_vertex_id(i, tris)
    #             if len(ns) > 0:
    #                 for n in ns:
    #                     # distance between members of same ring
    #                     #if rings[n] == ring_index:
    #                     dists.append(np.linalg.norm(population[i, 0] - population[n, 0]))
    #
    #     if len(dists) > 0:
    #         ring_avgs.append(np.mean(dists))
    #     else:
    #         ring_avgs.append(0)

    outer_ring = np.array([population[index, 0] for index, ring in enumerate(rings) if ring == 1])
    center_of_flock = np.average(outer_ring, axis=0)

    def shrink_ring(ring_points):
        ring_points -= center_of_flock
        new_ring_points = []

        for point in ring_points:
            point_len = np.linalg.norm(point)
            if point_len < 1:
                return None
            norm_point = point / point_len
            new_ring_points.append(norm_point * (point_len - 1))

        return np.array(new_ring_points) + center_of_flock

    for _ in range(20):
        outer_ring = shrink_ring(outer_ring)
        if outer_ring is None:
            return []

    def density_at_ring(ring_points):
        count = 0
        for pos in population[:, 0]:
            # if the boid is close to any point on the current ring
            for ring_point in ring_points:
                if np.linalg.norm(ring_point - pos) < 20:
                    count += (20 - np.linalg.norm(ring_point - pos)) / 20
                    break

        return count

    while outer_ring is not None:
        ring_avgs.append(density_at_ring(outer_ring))
        outer_ring = shrink_ring(outer_ring)

    return ring_avgs
