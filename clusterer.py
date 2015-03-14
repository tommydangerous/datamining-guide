import csv
import Queue

class Clusterer(object):
    def __init__(self):
        self.columns        = []
        self.data           = []
        self.distances      = {}
        self.priority_queue = Queue.PriorityQueue()

    def import_data(self, path):
        f     = open(path, 'r')
        csv_f = csv.reader(f)
        rows  = [row for row in csv_f]
        f.close()

        # Set the columns
        self.columns = rows[0]
        for column in self.columns:
            self.data.append([])

        # Import the rows as vectors
        for row in rows[1:]:
            for i in range(len(row)):
                try:
                    value = float(row[i])
                except ValueError:
                    value = row[i]
                self.data[i].append(value)

    # Distance methods
    def euclidean(self, coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        return pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)

    def manhattan(self, vector1, vector2):
        square_sums = []
        for i in range(len(vector1)):
            square_sums.append(pow(vector1[i] - vector2[i], 2))
        return pow(sum(square_sums), 0.5)

    # Normalization methods
    def normalize(self):
        for i in range(len(self.data[1:])):
            index = i + 1
            self.data[index] = self.normalize_list(self.data[index])

    def absolute_standard_deviation(self, median, array):
        return sum([abs(x - median) for x in array]) / float(len(array))

    def median(self, array):
        array  = sorted(array)
        length = len(array)
        if length % 2 == 0:
            index = length / 2
            return (array[index - 1] + array[index]) / 2.0
        else:
            return array[length / 2]

    def modified_standard_score(self, median, asd, value):
        return (value - median) / asd

    def normalize_list(self, array):
        """Median, absolute standard deviation, modified standard score."""
        median = self.median(array)
        asd    = self.absolute_standard_deviation(median, array)
        return [self.modified_standard_score(median, asd, x) for x in array]
