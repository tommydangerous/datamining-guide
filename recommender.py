import json

from math import sqrt

class Recommender(object):
    def __init__(self, data={}):
        self.data = data

    def compute_nearest_neighbor(self, key):
        array = []
        for k in self.data:
            if k != key:
                total = self.manhattan(self.data[key], self.data[k])
                array.append((total, k))
        array.sort()
        #=> [(2.0, 'James'), (3.5, 'Kim')]
        return array

    def cosine_similarity(self, list1, list2):
        sum_x_y = 0
        sum_x_2 = 0
        sum_y_2 = 0
        for key in list1:
            if key in list2:
                x = list1[key]
                y = list2[key]
                sum_x_y += x * y
                sum_x_2 += pow(x, 2)
                sum_y_2 += pow(y, 2)
        den = sqrt(sum_x_2) * sqrt(sum_y_2)
        return sum_x_y / den

    def euclidean(self, list1, list2):
        return self.minkowski(list1, list2, 2)

    def load_data(self, file_name):
        f = open('{}.json'.format(file_name), 'r')
        self.data = json.loads(f.read())
        return self.data

    def load_text(self, file_name, columns):
        self.data = {}
        f = open('{}.txt'.format(file_name), 'r')
        for line in f.readlines():
            line_array = line.split(',')
            dictionary = {}
            for i in range(len(columns)):
                dictionary[columns[i]] = line_array[i + 1].strip().replace(
                    '\n', '')
            self.data[line_array[0]] = dictionary
        return self.data

    def manhattan(self, list1, list2):
        total = 0
        for key in list1:
            if key in list2:
                total += abs(list1[key] - list2[key])
        #=> 2.0
        return total

    def minkowski(self, list1, list2, r=5):
        total = 0
        for key in list1:
            if key in list2:
                total += pow(abs(list1[key] - list2[key]), r)
        if total > 0:
            return pow(total, 1.0 / r)
        else:
            return 0

    def pearson(self, list1, list2):
        n = 0
        sum_x = 0
        sum_y = 0
        sum_x_y = 0
        sum_x_2 = 0
        sum_y_2 = 0
        for key in list1:
            if key in list2:
                x = list1[key]
                y = list2[key]
                n += 1
                sum_x += x
                sum_y += y
                sum_x_y += x * y
                sum_x_2 += pow(x, 2)
                sum_y_2 += pow(y, 2)
        num = sum_x_y - (sum_x * sum_y / n)
        den = (sqrt(sum_x_2 - (pow(sum_x, 2) / n)) *
            sqrt(sum_y_2 - (pow(sum_y, 2) / n)))
        return num / den

    def recommend(self, key):
        nearest_neighbor = self.compute_nearest_neighbor(key)[0]
        neighbor = nearest_neighbor[1]
        array = []
        for k in self.data[neighbor]:
            if k not in self.data[key]:
                array.append((k, self.data[neighbor][k]))
        #=> [('Em', 4.0), ('Drake', 5.0)]
        return sorted(array,
            key=lambda data_tuple: data_tuple[1], reverse = True)


recommender = Recommender()
data = recommender.load_data('users')

manhattan = recommender.manhattan(data['Jordyn'], data['Hailey'])
euclidean = recommender.euclidean(data['Jordyn'], data['Hailey'])
minkowski = recommender.minkowski(data['Jordyn'], data['Hailey'], 2)
pearson = recommender.pearson(data['Angelica'], data['Jordyn'])
cosine = recommender.cosine_similarity(data['Clara'], data['Robert'])
# print(cosine)

data = recommender.load_data('music')
euclidean = recommender.euclidean(data["Glee Cast/Jessie's Girl"],
    data["Lady Gaga/Alejandro"])
# print(euclidean)
