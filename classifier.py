from content_recommender import ContentRecommender

class Classifier(ContentRecommender):
    def __init__(self, data=[]):
        self.data   = data
        self.format = []
        self.mean_and_deviation   = []
        self.median_and_deviation = []
        self.normalize_min_max    = []

    def classify(self, vector, strategy='normalize'):
        return self.nearest_neighbor(self.standardize(vector, strategy))[1][0]

    def standardize(self, vector, strategy):
        if strategy == 'standardize':
            self.standardize_columns()
            return self.standardize_vector(vector)
        elif strategy == 'normalize':
            self.normalize_columns()
            return self.normalize_vector(vector)

    def euclidean(self, vector1, vector2):
        array = map(lambda v1, v2: pow(v1 - v2, 2), vector1, vector2)
        return pow(sum(array), 0.5)

    def load_data(self, file_name):
        self.data = []

        f = open('sets/{}.txt'.format(file_name), 'r')
        lines = f.readlines()
        f.close()

        self.format = [w.strip().replace('\n', '') for w in lines[0].split(',')]

        for line in lines[1:]:
            fields = [w.strip().replace('\n', '') for w in line.split(',')]
            ignore = []
            vector = []
            for i in range(len(fields)):
                if self.format[i] == 'num':
                    vector.append(float(fields[i]))
                elif self.format[i] == 'comment':
                    ignore.append(fields[i])
                elif self.format[i] == 'class':
                    classification = fields[i]
            self.data.append((classification, vector, ignore))

    def manhattan(self, vector1, vector2):
        # length = min(len(vector1), len(vector2))
        # distance = 0
        # for i in range(length):
        #     distance += abs(vector1[i] - vector2[i])
        # return distance

        array = map(lambda v1, v2: abs(v1 - v2), vector1, vector2)
        return sum(array)

    def nearest_neighbor(self, vector):
        # array = []
        # for v in self.data:
        #     distance = self.manhattan(vector, v[1])
        #     array.append((distance, v))
        # return sorted(array)

        return min(self.nearest_neighbors(vector))

    def nearest_neighbors(self, vector):
        return sorted([(self.euclidean(vector, v[1]), v) for v in self.data])

    def normalize_column(self, col_number):
        values = [v[1][col_number] for v in self.data]
        max_r  = max(values)
        min_r  = min(values)
        self.normalize_min_max.append((min_r, max_r))
        for v in self.data:
            v[1][col_number] = (v[1][col_number] - min_r) / (max_r - min_r)

    def normalize_columns(self):
        for i in range(len(self.data[0][1])):
            self.normalize_column(i)

    def normalize_vector(self, vector):
        vector = list(vector)
        for i in range(len(vector)):
            max_r = self.normalize_min_max[i][1]
            min_r = self.normalize_min_max[i][0]
            vector[i] = (vector[i] - min_r) / (max_r - min_r)
        return vector

    def standardize_column(self, col_number):
        values = [v[1][col_number] for v in self.data]
        self.median_and_deviation.append(
            (self.median(values), self.absolute_standard_deviation(values))
        )
        for v in self.data:
            v[1][col_number] = self.modified_standard_score(
                v[1][col_number], values
            )

    def standardize_columns(self):
        """Standardize all the existing columns."""
        for i in range(len(self.data[0][1])):
            self.standardize_column(i)

    def standardize_vector(self, vector):
        vector = list(vector)
        for i in range(len(vector)):
            median, asd = self.median_and_deviation[i]
            vector[i] = (vector[i] - median) / asd
        return vector


def unit_test():
    list1 = [54, 72, 78, 49, 65, 63, 75, 67, 54]
    c = Classifier()
    c.load_data('atheletes_training_set')
    m1 = c.median(list1)
    assert(round(m1, 3) == 65)
    print('median works correctly')

# unit_test()

def test_training_set(set_type, standardize=None):
    c1 = Classifier()
    c1.load_data('{}_training_set'.format(set_type))
    c2 = Classifier()
    c2.load_data('{}_test_set'.format(set_type))

    correct = 0
    for v in c2.data:
        if v[0] == c1.classify(v[1], standardize):
            correct += 1

    accuracy   = float(correct) / len(c2.data)
    percentage = accuracy * 100

    print('-' * 100)
    print('{} ({}): {}{} accuracy'.format(set_type.capitalize(),
        standardize, percentage, '%'))

def test_all_training_sets():
    for method in ['standardize', 'normalize']:
        for name in ['atheletes', 'iris', 'mpg']:
            test_training_set(name, method)

# list1 = [54, 72, 78, 49, 65, 63, 75, 67, 54]

# c = Classifier()
# c.load_data('atheletes_training_set')
# m = c.median(list1)
# asd = c.absolute_standard_deviation(list1)

# print(c.classify([59, 90]))
