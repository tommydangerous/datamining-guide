from math import sqrt
from recommender import Recommender

class ContentRecommender(Recommender):
    def absolute_standard_deviation(self, values):
        # Uses median instead of average to account for outliers
        median = self.median(values)
        num    = sum([abs(x - median) for x in values])
        return num / float(len(values))

    def average(self, values):
        return sum([x for x in values]) / float(len(values))

    def classify(self, user, item, dict1, items):
        nearest = self.compute_nearest_neighbor(item, dict1, items)
        return self.data[user][nearest[0][1]]

    def compute_nearest_neighbor(self, key, dict1, items):
        array = []
        for k in items:
            if k != key:
                vector1 = []
                vector2 = []
                items_k = items[k]
                for attribute in dict1:
                    if attribute in items_k:
                        vector1.append(dict1[attribute])
                        vector2.append(items_k[attribute])
                manhattan = self.manhattan(vector1, vector2)
                array.append((manhattan, k))
        return sorted(array)

    def manhattan(self, vector1, vector2):
        return sum([abs(vector1[i] - vector2[i]) for i in range(len(vector1))])

    def median(self, values):
        values.sort()
        middle = len(values) / 2
        if len(values) % 2 == 0:
            left  = values[middle - 1]
            right = values[middle]
            return (left + right) / 2
        else:
            return values[middle]

    def modified_standard_score(self, value, values):
        # Uses median instead of average
        num = value - self.median(values)
        return num / self.absolute_standard_deviation(values)

    def standard_deviation(self, values):
        avg = self.average(values)
        num = sum([pow(x - avg, 2) for x in values])
        return sqrt(num / float(len(values)))

    def standard_score(self, value, values):
        return (value - self.average(values)) / self.standard_deviation(values)

r      = ContentRecommender()
data   = r.load_data('salary')
values = data.values()

sd = r.standard_deviation(values)
ss = r.standard_score(data['Rita A'], values)
manhattan = r.manhattan([5, 5, 4, 2, 1, 1, 1], [1, 5, 2.5, 1, 1, 5, 1])
# print(manhattan)

items = r.load_data('music')
data  = r.load_data('users_like')
dict1 = {
    "piano": 1,
    "vocals": 5,
    "beat": 2.5,
    "blues": 1,
    "guitar": 1,
    "backup vocals": 5,
    "rap": 1
}
nearest = r.compute_nearest_neighbor('Cagle', dict1, items)
classify = r.classify('Angelica', 'Cagle', dict1, items)
# print(classify)
