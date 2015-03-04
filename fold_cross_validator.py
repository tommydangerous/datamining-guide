import random

from classifier import Classifier

class FoldCrossValidator(Classifier):
    def __init__(self, name, column_format, number_of_buckets=10):
        """Initializer"""
        super(FoldCrossValidator, self).__init__()
        self.confusion_matrix  = {}
        self.column_format     = column_format
        self.name              = name
        self.number_of_buckets = number_of_buckets

        self.reset_data()

    # Private methods

    def __bucket_filename(self, bucket_number):
        return "{}-{}".format(self.name, bucket_number)

    def load_data_from_lines(self, lines, training):
        for line in lines:
            classification = None
            ignore         = []
            vector         = []
            fields         = self.split_line(line)

            for i in range(len(fields)):
                column_format = self.column_format[i]
                field_value   = fields[i]
                if column_format =='class':
                    classification = field_value
                elif column_format == 'num':
                    vector.append(float(field_value))
                elif column_format == 'comment':
                    ignore.append(field_value)

            if classification:
                tup = (classification, vector, ignore)
                if training:
                    self.data.append(tup)
                else:
                    self.test_data.append(tup)

    def load_training_buckets(self, exclude_bucket_number):
        """Load buckets for training data, excluding a particular bucket."""
        numbers = range(self.number_of_buckets)
        slice1  = slice(0, exclude_bucket_number)
        slice2  = slice(exclude_bucket_number + 1, self.number_of_buckets)
        bucket_numbers = numbers[slice1] + numbers[slice2]

        lines = []
        for i in bucket_numbers:
            f = self.__open_file(self.__bucket_filename(i), self.name)
            lines += f.readlines()
            f.close()

        self.load_data_from_lines(lines, True)

    def load_test_buckets(self, bucket_number):
        """Load bucket for test data."""
        f = self.__open_file(self.__bucket_filename(bucket_number), self.name)
        lines = f.readlines()
        f.close()

        self.load_data_from_lines(lines, False)

    def __open_file(self, filename, folder=None):
        if folder:
            folder = '/' + folder
        else:
            folder = ''

        f = None
        for extension in ['', '.txt']:
            try:
                f = open('sets{}/{}{}'.format(folder, filename, extension))
                break
            except IOError as e:
                e

        return f

    def print_confusion_matrix(self):
        top_line = '   | {}'.format(
            ' | '.join(sorted(self.confusion_matrix.keys()))
        )
        print(top_line)

        for key in sorted(self.confusion_matrix.keys()):
            values = self.confusion_matrix[key]
            array  = []

            for k in sorted(values.keys()):
                array.append(values[k])

            line_string = ''

            for number in array:
                string = str(number)
                diff = 4 - len(string)

                string = ' {}'.format(string)
                for i in range(diff - 1):
                    string += ' '

                string += '|'

                line_string += string

            print('{} |{}'.format(key, line_string))

    def reset_data(self):
        self.data      = []
        self.test_data = []

    def split_line(self, line):
        array = []
        if len(line.split(',')) > 1:
            array = line.split(',')
        else:
            array = line.split('\t')
        return [w.strip().replace('\n', '') for w in array]

    # Public methods

    def classify_vector(self, vector, strategy=None):
        self.load_all_training_buckets()

        return super(FoldCrossValidator, self).classify(vector, strategy)

    def create_buckets(self, class_column=0):
        """Seperate the data into X buckets and stratify them so that there is
        the same amount of representation of each category in each bucket."""

        f = self.__open_file('{}_training_set'.format(self.name))
        lines = f.readlines()
        f.close()

        data = {}

        for line in lines:
            # Get the category
            category = line.split()[class_column]
            data.setdefault(category, [])
            data[category].append(line)
        # Initialize the buckets
        buckets = []
        for i in range(self.number_of_buckets):
            buckets.append([])
        # For each category, put the data into the buckets
        for key in data.keys():
            # Randomize order of instances for each class
            random.shuffle(data[key])
            bucket_number = 0
            # Divide into buckets
            for item in data[key]:
                buckets[bucket_number].append(item)
                bucket_number = (bucket_number + 1) % self.number_of_buckets
        # Write to file
        for bucket_number in range(self.number_of_buckets):
            f = open('sets/{}/{}'.format(
                self.name, self.__bucket_filename(bucket_number)
            ), 'w')
            for item in buckets[bucket_number]:
                f.write(item)
            f.close()

    def kappa_interpretation(self, kappa):
        if kappa < 0.01:
            return 'less than chance performance'
        elif kappa >= 0.01 and kappa <= 0.2:
            return 'slightly good'
        elif kappa > 0.2 and kappa <= 0.4:
            return 'fair performance'
        elif kappa > 0.4 and kappa <= 0.6:
            return 'moderate performance'
        elif kappa > 0.6 and kappa <= 0.8:
            return 'substantially good performance'
        elif kappa > 0.8:
            return 'near perfect performance'

    def kappa_statistic(self, real_accuracy, random_accuracy):
        """Tell us how much better the real classifier is compared to this
        random one."""
        return (real_accuracy - random_accuracy) / (1 - random_accuracy)

    def knn(self, vector, k, strategy):
        vector = self.standardize(vector, strategy)
        neighbors = self.nearest_neighbors(vector)[slice(0, k)]
        votes = {}
        for n in neighbors:
            values = n[1]
            category = values[0]
            votes.setdefault(category, 0)
            votes[category] += 1

        winning_category = None
        category_count   = 0

        for k, v in votes.items():
            if v > category_count:
                winning_category = k
                category_count   = v

        return winning_category

    def load_all_training_buckets(self):
        self.reset_data()
        self.load_training_buckets(self.number_of_buckets)

    def optimal_k(self):
        array = []

        for i in range(50):
            print('K = {}'.format(i))

            percentage = c.test_training_bucket('normalize', i)
            array.append((percentage, i))

        return sorted(array, reverse=True)[0][1]

    def random_classifier_accuracy(self):
        """Calculate the accuracy of a random classifier."""
        column_totals = {}
        row_totals    = {}

        for category in self.confusion_matrix:
            row = self.confusion_matrix[category]
            row_totals[category] = sum([v for (k, v) in row.items()])

            for column in row:
                column_totals.setdefault(column, 0)
                column_totals[column] += self.confusion_matrix[category][column]

        total_instances = float(sum([v for (k, v) in column_totals.items()]))

        column_percentage = {}

        for k, v in column_totals.items():
            column_percentage[k] = v / total_instances

        total = sum([column_percentage[k] * v for (k, v) in row_totals.items()])
        return total / total_instances

    def test_training_bucket(self, strategy, k=None):
        correct = 0

        all_categories = []
        for i in range(self.number_of_buckets):
            print('Testing bucket {}'.format(i))

            self.reset_data()
            self.load_training_buckets(i)
            self.load_test_buckets(i)

            if len(all_categories) == 0:
                all_categories = set([v[0] for v in self.data + self.test_data])
                for cat in all_categories:
                    self.confusion_matrix.setdefault(
                        cat, dict.fromkeys(all_categories, 0)
                    )

            for value in self.test_data:
                category   = value[0]
                vector     = value[1]
                if k:
                    classified = self.knn(vector, k, strategy)
                else:
                    classified = self.classify(vector, strategy)

                self.confusion_matrix[category][classified] += 1

                if category == classified:
                    correct += 1

        total_length = len(self.data) + len(self.test_data)
        accuracy     = float(correct) / total_length
        percentage   = accuracy * 100

        self.print_confusion_matrix()

        print('')
        print('{} percent accurate'.format(percentage))
        print('total of {} instances'.format(total_length))

        return accuracy

# c = FoldCrossValidator(
#     'mpg', ['class', 'num', 'num', 'num', 'num', 'num', 'comment']
# )
# c.create_buckets()
# c.test_training_bucket('normalize')
# print(c.classify_vector([4, 91.00, 53.00, 1795, 17.4])) #=> 35

# c = FoldCrossValidator(
#     'pima2', ['num', 'num', 'num', 'num', 'num', 'num', 'num', 'num', 'class']
# )
# p_c = c.test_training_bucket('normalize', 32)
# p_r = c.random_classifier_accuracy()
# k   = c.kappa_statistic(p_c, p_r)
# print(k, c.kappa_interpretation(k))
