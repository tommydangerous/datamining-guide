from clusterer import Clusterer

class HierarchicalClusterer(Clusterer):
    def print_cluster(self, cluster):
        t1, t2 = cluster
        if type(t1) is tuple:
            print(t1)
            self.print_cluster(t1)

        if type(t2) is tuple:
            print(t2)
            self.print_cluster(t2)

    # Clustering methods
    def cluster(self):
        while self.priority_queue.qsize() > 1:
            item1    = self.priority_queue.get()
            item2    = self.priority_queue.get()
            combined = self.combine_items(item1, item2)
            self.priority_queue.put(combined)

        return self.priority_queue.get()

    def combine_clusters(self, cluster1, cluster2):
        if len(cluster1) == 1 and len(cluster2) == 1:
            return (cluster1, cluster2)
        elif len(cluster1) > len(cluster2):
            return (cluster1, cluster2)
        else:
            return (cluster2, cluster1)

    def combine_distances(self, dict1, dict2):
        distances = {}
        for k in dict1:
            if k in dict2:
                distance1 = dict1[k]
                distance2 = dict2[k]
                if distance1 < distance2:
                    distances[k] = distance1
                else:
                    distances[k] = distance2
        return distances

    def combine_items(self, item1, item2):
        distances = self.combine_distances(item1[4], item2[4])

        if len(distances) == 0:
            distance = 0
            neighbor = ()
        else:
            nearest = self.nearest_neighbor(distances)
            nearest_index = nearest[1]

            distance = nearest[0]
            neighbor = (self.data[0][nearest_index], distance, nearest_index)

        index   = min([item1[1], item2[1]])
        cluster = self.combine_clusters(item1[2], item2[2])

        return (distance, index, cluster, neighbor, distances)

    def initialize_cluster(self):
        for i in range(len(self.data[0])):
            nearest       = self.nearest_neighbor(self.distances[i])
            nearest_index = nearest[1]

            distance  = nearest[0]
            cluster   = (self.data[0][i])
            neighbor  = (self.data[0][nearest_index], distance, nearest_index)
            distances = self.distances

            self.priority_queue.put(
                (distance, i, cluster, neighbor, self.distances[i])
            )

    # Distance methods
    def calculate_distances(self):
        indices = range(len(self.data[0]))
        for i in indices:
            self.distances[i] = {}
            for index in indices:
                if i != index:
                    if len(self.data[1:]) == 2:
                        coord1 = (self.data[1][i], self.data[2][i])
                        coord2 = (self.data[1][index], self.data[2][index])
                        self.distances[i][index] = self.euclidean(
                            coord1, coord2
                        )
                    else:
                        vector1 = [x[i] for x in self.data[1:]]
                        vector2 = [x[index] for x in self.data[1:]]
                        self.distances[i][index] = self.manhattan(
                            vector1, vector2
                        )

    def nearest_neighbor(self, dictionary):
        return sorted([(v, k) for k, v in dictionary.items()])[0]

def dogs():
    c = HierarchicalClusterer()
    c.import_data('sets/dogs/dogs.csv')
    c.normalize()
    c.calculate_distances()
    c.initialize_cluster()
    t = c.cluster()[2]

    print(t)
    c.print_cluster(t)

def cereal():
    c = HierarchicalClusterer()
    c.import_data('sets/cereal/cereal.csv')
    c.normalize()
    c.calculate_distances()
    c.initialize_cluster()
    t = c.cluster()[2]

    print(t)
    c.print_cluster(t)

dogs()
cereal()
