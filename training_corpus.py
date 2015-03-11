import math
import os
import re

from naive_bayes import NaiveBayes

# 1. Read in the stop list
# 2. Read the training directories in the sub directories
# 3. For each sub directory, train and count the occurence of each word
# 4. Compute the probability

class TrainingCorpus(NaiveBayes):
    def __init__(self, name):
        super(TrainingCorpus, self).__init__(name, [])
        self.categories    = []
        self.probabilities = {}
        self.stop_words    = {}
        self.totals        = {}
        self.vocabulary    = {}
        self.word_counts   = {}

    def __test_directory_name(self):
        return 'sets/{}/test'.format(self.name)

    def __training_directory_name(self):
        return 'sets/{}/train'.format(self.name)

    def classify(self, document_path):
        results = dict.fromkeys(self.categories, 0)
        f = open('sets/{}/{}'.format(self.name, document_path), 'r')
        for line in f:
            words = line.split()
            for text in words:
                match = self.match_from_text(text)
                if match:
                    word = match.group(0).lower()
                    if word in self.vocabulary:
                        for category in self.categories:
                            probability = self.probabilities[category][word]
                            if probability:
                                results[category] += math.log(probability)

                        # print('neg', self.probabilities['neg'].get('extras'))
                        # print('pos', self.probabilities['pos'].get('extras'))

        f.close()
        results = list(results.items())
        results.sort(key=lambda tup: tup[1], reverse = True)
        return results[0][0]

    def process(self):
        self.read_training_documents()
        self.trim_vocabulary()
        self.compute_probabilities()

    def test(self):
        score = 0
        total = 0
        for category in self.categories:
            print('-' * 50)
            print('Category: {}'.format(category))
            print('-' * 25)

            files = os.listdir('{}/{}'.format(
                self.__test_directory_name(), category
            ))

            print('Testing {} document(s)'.format(len(files)))

            for name in files:
                classification = self.classify('test/{}/{}'.format(
                    category, name
                ))

                if classification == category:
                    score += 1
                total += 1
        accuracy = float(score) / float(total)

        print('{} percent accuracy ({} out of {})'.format(
            accuracy * 100, score, total
        ))

    def compute_probabilities(self):
        vocabulary_lenth = len(self.vocabulary)
        index = 0

        # prob = dict.fromkeys(self.categories, {})
        prob = {}
        for category in self.categories:
            prob[category] = {}

        for category in self.categories:
            n = self.totals[category]
            vocabulary = vocabulary_lenth
            denominator = n + vocabulary

            for word in self.vocabulary:
                if word in self.word_counts[category]:
                    count = self.word_counts[category][word]
                else:
                    count = 1
                n_k = float(count)
                numerator = n_k + 1

                p = numerator / denominator

                # if word == 'extras':
                #     print(category, numerator, denominator, p)
                prob[category][word] = p

                # self.probabilities[category][word] = p

        # print('neg', self.probabilities['neg'].get('extras'))
        # print('pos', self.probabilities['pos'].get('extras'))
        # print(prob)

        self.probabilities = prob

    def match_from_text(self, text):
        return re.search(r'[A-Za-z0-9]+', text)

    def read_stop_words(self, file_name='stop_words'):
        f = open('sets/{}/{}.txt'.format(self.name, file_name), 'r')
        text = f.read()
        f.close()
        self.stop_words = dict.fromkeys(
            [x.strip() for x in text.split('\n') if len(x) > 0], 0
        )

    def read_training_documents(self):
        self.categories    = os.listdir(self.__training_directory_name())
        self.probabilities = dict.fromkeys(self.categories, {})
        self.totals        = dict.fromkeys(self.categories, 0)

        for category in self.categories:
            print('-' * 50)
            print('Category: {}'.format(category))
            print('-' * 25)

            tup                          = self.train_directory(category)
            self.probabilities[category] = tup[0]
            self.totals[category]        = tup[1]

    def train_directory(self, dir_name):
        files = os.listdir('{}/{}'.format(
            self.__training_directory_name(), dir_name
        ))

        print('Reading {} file(s)'.format(len(files)))

        counts = {}
        total  = 0

        for name in files:
            f = open('{}/{}/{}'.format(
                self.__training_directory_name(), dir_name, name
            ))
            for line in f:
                words = line.split()
                for text in words:
                    match = self.match_from_text(text)
                    if match:
                        word = match.group(0).lower()
                        if word not in self.stop_words:
                            self.vocabulary.setdefault(word, 0)
                            self.vocabulary[word] += 1
                            counts.setdefault(word, 0)
                            counts[word] += 1
                            total += 1
            f.close()

        return (counts, total)

    def trim_vocabulary(self):
        delete = []
        for word in self.vocabulary:
            if self.vocabulary[word] < 3:
                delete.append(word)
        for word in delete:
            del self.vocabulary[word]

    def fold_cross_validation(self, buckets=10):
        # 1. Start at 1 and go to n
        # 2. Loop through the categories
        # 3. Train each category, leaving out 1 bucket
        # 4. Loop through each category
        # 5. Test the bucket that was left out
        # 6. Repeat
        # 7. Count the correct and divide by the amount of attempts
        self.number_of_buckets = buckets
        self.categories        = os.listdir(self.__training_directory_name())

        score = 0
        total = 0

        for category in self.categories:
            self.confusion_matrix.setdefault(
                category, dict.fromkeys(self.categories, 0)
            )

        for i in range(self.number_of_buckets):
            print('-' * 100)
            print('Bucket: {}'.format(i))
            print('-' * 30)

            self.probabilities = dict.fromkeys(self.categories, {})
            self.totals        = dict.fromkeys(self.categories, 0)
            self.word_counts   = dict.fromkeys(self.categories, {})

            for category in self.categories:
                tup = self.train_category(category, i)
                self.word_counts[category] = tup[0]
                self.totals[category]      = tup[1]

                self.trim_vocabulary()
                self.compute_probabilities()

            for category in self.categories:
                print('Testing category: {}'.format(category))

                files = os.listdir('{}/{}/{}'.format(
                    self.__training_directory_name(), category, i
                ))

                for name in files:
                    file_name = 'train/{}/{}/{}'.format(category, i, name)
                    classification = self.classify(
                        file_name
                    )

                    self.confusion_matrix[category][classification] += 1

                    if classification == category:
                        score += 1

                    total += 1

        accuracy = float(score) / float(total)

        print('')
        self.print_confusion_matrix()
        print('')
        print('{} percent accuracy ({} out of {})'.format(
            accuracy * 100, score, total
        ))

        return accuracy

    def train_category(self, category, exclude_bucket_number):
        numbers = range(self.number_of_buckets)
        slice1  = slice(0, exclude_bucket_number)
        slice2  = slice(exclude_bucket_number + 1, self.number_of_buckets)
        buckets = numbers[slice1] + numbers[slice2]

        counts = {}
        total  = 0

        print('Training category: {}'.format(category))

        for bucket in buckets:
            files = os.listdir('{}/{}/{}'.format(
                self.__training_directory_name(), category, bucket
            ))

            for name in files:
                f = open('{}/{}/{}/{}'.format(
                    self.__training_directory_name(), category, bucket, name
                ))
                for line in f:
                    words = line.split()
                    for text in words:
                        match = self.match_from_text(text)
                        if match:
                            word = match.group(0).lower()
                            if word not in self.stop_words:
                                self.vocabulary.setdefault(word, 0)
                                self.vocabulary[word] += 1
                                counts.setdefault(word, 0)
                                counts[word] += 1
                                total += 1
                f.close()

        return (counts, total)


c = TrainingCorpus('reviews')
c.read_stop_words()
real   = c.fold_cross_validation(10)
random = c.random_classifier_accuracy()
kappa  = c.kappa_statistic(real, random)
print(kappa, c.kappa_interpretation(kappa))
