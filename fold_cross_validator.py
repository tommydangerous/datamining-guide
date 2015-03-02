import random

from classifier import Classifier

class FoldCrossValidator(Classifier):
    def __init__(self, name, number_of_buckets=10):
        """Initializer"""
        super(FoldCrossValidator, self).__init__()
        self.confusion_matrix  = {}
        self.name              = name
        self.number_of_buckets = number_of_buckets
        self.test_data         = []

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
                column_format = self.format[i]
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

    def load_training_buckets(self):
        """Load self.number_of_buckets - 1 files for training."""
        lines = []
        for i in range(self.number_of_buckets - 1):
            f = open(self.__bucket_filename(i))
            lines += f.readlines()
            f.close()

        self.format = self.__split_line(lines[0])

        self.__load_data_from_lines(lines[1:], True)

    def load_test_buckets(self):
        """Load last data bucket for testing."""
        f = open(self.__bucket_filename(self.number_of_buckets - 1))
        lines = f.readlines()
        f.close()

        self.__load_data_from_lines(lines, False)

    def print_confusion_matrix(self):
        top_line = '__ | {}'.format(
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
        all_categories = set([v[0] for v in self.data + self.test_data])

        for cat in all_categories:
            self.confusion_matrix.setdefault(
                cat, dict.fromkeys(all_categories, 0)
            )

        correct = 0
        for value in self.test_data:
            category   = value[0]
            vector     = value[1]
            classified = self.classify(vector, standardize)

            if category == classified:
                correct += 1

            self.confusion_matrix[category][classified] += 1


        accuracy   = float(correct) / len(self.test_data)
        percentage = accuracy * 100

        print('-' * 100)
        print('{}: {}{} accuracy'.format(
            standardize.capitalize(), percentage, '%')
        )


c = FoldCrossValidator('mpg')
c.create_buckets()
c.load_training_buckets()
c.load_test_buckets()
c.test_training_bucket('standardize')

c.print_confusion_matrix()
