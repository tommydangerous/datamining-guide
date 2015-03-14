import collections
import random

from clusterer import Clusterer

class KmeansClusterer(Clusterer):
    def __init__(self, k=3):
        super(KmeansClusterer, self).__init__()
        self.clusters = {}
        self.k        = k
        self.scatter  = 0

    def print_clusters(self):
        index = 0
        print('')
        for key, value in self.clusters.items():
            print('-' * 100)
            print('Cluster {}'.format(index))
            print('-' * 50)
            print('')
            print([x[0] for x in value])
            print('')
            index += 1
        print('Scatter: {}'.format(self.scatter))
        print('')

    # Clustering methods
    def cluster(self, points):
        new_clusters = dict((key, []) for key in points)

        items = [self.item_at_index(i) for i in range(len(self.data[0]))]
        for item in items:
            distances = []
            for cluster in new_clusters:
                vector1 = list(cluster)
                vector2 = list(item[1])
                distance = self.distance(vector1, vector2)
                distances.append((distance, cluster, item))

            closest = min(distances)
            new_clusters[closest[1]].append(closest[2])

        old_clusters  = self.clusters
        self.clusters = new_clusters

        self.scatter = self.sum_squared_error()
        print('Scatter: {}'.format(self.scatter))

        if not self.convergence(old_clusters, new_clusters):
            new_points = [
                self.compute_cluster_mean(v) for k, v in self.clusters.items()
            ]
            self.cluster([tuple(x) for x in new_points])

    def compute_cluster_mean(self, cluster):
        vector = []
        tuples = [x[1] for x in cluster]
        for i in range(len(tuples[0])):
            mean = sum([x[i] for x in tuples]) / float(len(tuples))
            vector.append(mean)
        return vector

    def convergence(self, old_clusters, new_clusters):
        if len(old_clusters) == 0:
            return False
        else:
            keys1 = old_clusters.keys()
            keys2 = new_clusters.keys()

            total_changed = 0
            total_items   = len(self.data[0])

            for key1 in keys1:
                distances = []
                for key2 in keys2:
                    vector1 = list(key1)
                    vector2 = list(key2)
                    distance = self.distance(vector1, vector2)
                    distances.append((distance, key2))

                closest = min(distances)
                key_to_remove = closest[1]
                keys2.remove(key_to_remove)

                vector1 = old_clusters[key1]
                vector2 = new_clusters[key_to_remove]
                if len(vector1) > len(vector2):
                    change = set(vector1) - set(vector2)
                else:
                    change = set(vector2) - set(vector1)

                number_changed = len(change)
                total_changed += number_changed

            percentage = total_changed / float(total_items)

            return percentage < 0.01

    def initialize_clusters(self, plus_plus=True):
        if plus_plus:
            random_points = self.kmeans_plus_plus(self.k)
        else:
            random_points = self.random_points(self.k)
        self.cluster(random_points)

    def item_at_index(self, index):
        vector = self.vector_at_index(index)
        classification = self.data[0][index]
        return (classification, tuple(vector))

    def kmeans_plus_plus(self, k):
        points = [self.vector_at_index(i) for i in range(len(self.data[0]))]
        random.shuffle(points)

        centroids = [points.pop()]

        while len(centroids) < k:
            distances = collections.defaultdict(list)
            for point in points:
                for centroid in centroids:
                    distance = self.distance(centroid, point)
                    distances[tuple(centroid)].append((distance, point))

            distances = [value for key, value in distances.items()]

            closest = []
            for i in range(len(distances[0])):
                vector = [x[i] for x in distances]
                closest.append(min(vector))

            total = float(sum([x[0] for x in closest]))

            closest = [(x / total, y) for x, y in closest]

            centroid = self.weighted_probability(closest)
            points.remove(centroid)
            centroids.append(centroid)

        return [tuple(x) for x in centroids]

    def weighted_probability(self, vector):
        i = 0
        n = vector[0][0]
        random.seed()
        while n < random.random():
            i += 1
            n += vector[i][0]
        return vector[i][1]

    def random_points(self, k):
        indices = range(len(self.data[0]))
        random.shuffle(indices)

        points = []
        for i in range(k):
            index  = indices.pop()
            vector = self.vector_at_index(index)
            points.append(tuple(vector))

        return points

    def sum_squared_error(self):
        distances = []
        for point, items in self.clusters.items():
            vector1 = list(point)
            for item in items:
                vector2  = list(item[1])
                distance = self.distance(vector1, vector2)
                distances.append(distance)

        return sum(distances)

    def vector_at_index(self, index):
        return [x[index] for x in self.data[1:]]

    # Distance methods
    def distance(self, vector1, vector2):
        if len(vector1) == 2:
            return self.euclidean(vector1, vector2)
        else:
            return self.manhattan(vector1, vector2)

def dogs():
    c = KmeansClusterer(4)
    c.import_data('sets/dogs/dogs.csv')
    c.normalize()
    c.initialize_clusters()
    c.print_clusters()

def cereal():
    c = KmeansClusterer(10)
    c.import_data('sets/cereal/cereal.csv')
    c.normalize()
    c.initialize_clusters()
    c.print_clusters()

dogs()
# cereal()
