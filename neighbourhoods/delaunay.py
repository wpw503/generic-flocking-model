from scipy.spatial import Delaunay


def delaunay_neighbourhood(index, population, tris=None):
    def get_neighbor_vertex_ids_from_vertex_id(vertex_id, tri):
        index_pointers, indices = tri.vertex_neighbor_vertices
        result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
        return result_ids

    if tris is None:
        tris = Delaunay(population[:, 0, :])

    n = get_neighbor_vertex_ids_from_vertex_id(index, tris)

    return n
