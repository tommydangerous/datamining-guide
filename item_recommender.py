from math import sqrt
from recommender import Recommender

class ItemRecommender(Recommender):
    def average_deviations(self, key1, key2):
        array = []
        for (key, value) in self.data.items():
            if key1 in value and key2 in value:
                i = value[key1]
                j = value[key2]
                array.append(i - j)
        return float(sum(array)) / len(array)

    # Adjusted cosine similarity
    def cosine_similarity(self, key1, key2):
        sum_x_y = 0
        sum_x_2 = 0
        sum_y_2 = 0
        for (key, value) in self.data.items():
            if key1 in value and key2 in value:
                array = [float(v) for (k, v) in value.items()]
                average = sum(array) / len(array)
                x = value[key1]
                y = value[key2]
                x_adjusted = x - average
                y_adjusted = y - average
                sum_x_y += x_adjusted * y_adjusted
                sum_x_2 += pow(x_adjusted, 2)
                sum_y_2 += pow(y_adjusted, 2)
        den = sqrt(sum_x_2) * sqrt(sum_y_2)
        return sum_x_y / den

    def denormalize(self, value, min_r, max_r):
        return (0.5 * ((value + 1) * (max_r - min_r))) + min_r

    def normalize(self, value, min_r, max_r):
        num = (2 * (value - min_r)) - (max_r - min_r)
        den = max_r - min_r
        return float(num) / float(den)

    def predict(self, user, item, min_r=1, max_r=5):
        sum_num = 0
        sum_den = 0
        for key in self.data[user]:
            if key != item:
                similarity = self.cosine_similarity(key, item)
                sum_num += (similarity *
                    self.normalize(self.data[user][key], min_r, max_r))
                sum_den += abs(similarity)
        return float(sum_num) / float(sum_den)

    def weighted_slope_one(self, u, j):
        num = 0
        den = 0
        for i in self.data[u]:
            if i != j:
                dev_j_i = self.average_deviations(j, i)
                u_i = self.data[u][i]
                card_j_i = len([key for key in self.data[u] if key != j])
                num += (dev_j_i + u_i) * card_j_i
                den += card_j_i
        return num / den

item = ItemRecommender()
data = item.load_data('users')

cosine = item.cosine_similarity("Kacey Musgraves", "Imagine Dragons")
predict = item.predict("David", "Kacey Musgraves")
average_deviations = item.average_deviations("Whitney Houston", "PSY")
weighted_slope_one = item.weighted_slope_one("Ben", "Whitney Houston")
print(weighted_slope_one)
