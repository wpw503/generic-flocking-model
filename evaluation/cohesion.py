import numpy as np

from config import WORLD_HEIGHT, WORLD_WIDTH
from evaluation.flocking import average_speed


def is_broken(population):
    sizes = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            sizes.append(np.abs(np.linalg.norm(population[i, 0, :] - population[j, 0, :])))

    size = round(max(sizes))

    if size > min(WORLD_WIDTH, WORLD_HEIGHT) * 0.7:
        return True

    if average_speed(population) < 0.2:
        return True

    # t = 0.5
    #
    # for k in range(2, 8):
    #     model = KMeans(n_clusters=k)
    #     model.fit(population[:, 0])
    #     pred = model.predict(population[:, 0])
    #     score = silhouette_score(population[:, 0], pred)
    #     #print('Silhouette Score for k = {}: {:<.3f}'.format(k, score))
    #
    #     if score > t:
    #         return True

    return False
