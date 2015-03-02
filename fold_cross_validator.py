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

        self.__reset_data()

    # Private methods

    def __bucket_filename(self, bucket_number):
        return "sets/{}/{}-{}.txt".format(self.name, self.name, bucket_number)

    def __load_data_from_lines(self, lines, training):
        for line in lines:
            classification = None
            ignore         = []
            vector         = []
            fields         = self.__split_line(line)

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

    def __load_training_buckets(self, exclude_bucket_number):
        """Load buckets for training data, excluding a particular bucket."""
        numbers = range(self.number_of_buckets)
        slice1  = slice(0, exclude_bucket_number)
        slice2  = slice(exclude_bucket_number + 1, self.number_of_buckets)
        bucket_numbers = numbers[slice1] + numbers[slice2]

        lines = []
        for i in bucket_numbers:
            f = open(self.__bucket_filename(i))
            lines += f.readlines()
            f.close()

        self.__load_data_from_lines(lines, True)

    def __load_test_buckets(self, bucket_number):
        """Load bucket for test data."""
        f = open(self.__bucket_filename(bucket_number))
        lines = f.readlines()
        f.close()

        self.__load_data_from_lines(lines, False)

    def __reset_data(self):
        self.data      = []
        self.test_data = []

    def __split_line(self, line):
        return [w.strip().replace('\n', '') for w in line.split(',')]

    # Public methods

    def create_buckets(self, class_column=0):
        """Seperate the data into X buckets and stratify them so that there is
        the same amount of representation of each category in each bucket."""
        f = open('sets/{}_training_set.txt'.format(self.name))
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
            f = open(self.__bucket_filename(bucket_number), 'w')
            for item in buckets[bucket_number]:
                f.write(item)
            f.close()

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

    def test_training_bucket(self, standardize='standardize'):
        correct = 0

        all_categories = []
        for i in range(self.number_of_buckets):
            print('Testing bucket {}'.format(i))

            self.__reset_data()
            self.__load_training_buckets(i)
            self.__load_test_buckets(i)

            if len(all_categories) == 0:
                all_categories = set([v[0] for v in self.data + self.test_data])
                for cat in all_categories:
                    self.confusion_matrix.setdefault(
                        cat, dict.fromkeys(all_categories, 0)
                    )

            for value in self.test_data:
                category   = value[0]
                vector     = value[1]
                classified = self.classify(vector, standardize)

                self.confusion_matrix[category][classified] += 1

                if category == classified:
                    correct += 1

        accuracy   = float(correct) / (len(self.data) + len(self.test_data))
        percentage = accuracy * 100

        print('-' * 100)
        print('{}: {}{} accuracy'.format(
            standardize.capitalize(), percentage, '%')
        )

c = FoldCrossValidator(
    'mpg', ['class', 'num', 'num', 'num', 'num', 'num', 'comment']
)
c.create_buckets()
c.test_training_bucket('normalize')
c.print_confusion_matrix()
