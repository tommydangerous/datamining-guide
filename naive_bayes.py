import math

from fold_cross_validator import FoldCrossValidator

class NaiveBayes(FoldCrossValidator):
    def __init__(self, name, column_format):
        super(NaiveBayes, self).__init__(name, column_format)

    def classify(self, array):
        """h.map = argmax.h->H[P(D|h) * P(h)]"""
        hypotheses = []

        for classification in self.prior_probabilities:
            # P(h|D) = P(D|h) * p(h) / P(D)
            # P(h|D & E & F) = P(D|h) * P(E|h) * P(F|h) * P(h)

            p_h = self.prior_probabilities[classification]
            probability_product = 1

            index = 0
            for column in self.column_format:
                if column == 'class':
                    continue

                value = array[index]
                probability = self.data[classification][column][value]
                probability_product *= probability

                index += 1

            hypotheses.append((probability_product * p_h, classification))

        return max(hypotheses)[1]

    def load_data_from_lines(self, lines, training):
        for line in lines:
            classification = None
            dict1          = {}
            fields         = self.split_line(line)
            vector         = []

            for i in range(len(self.column_format)):
                column = self.column_format[i]
                value  = fields[i]

                if column == 'class':
                    classification = value
                else:
                    dict1[column] = value
                    try:
                        value = float(value)
                    except:
                        continue
                    vector.append(value)

            if classification:
                if training:
                    self.data.setdefault(classification, {})
                    for key in dict1:
                        self.data[classification].setdefault(key, [])
                        self.data[classification][key].append(dict1[key])
                else:
                    self.test_data.append((classification, vector))

    def calculate_means_and_sample_standard_deviation(self):
        """Store all the means and sample standard deviations for use when
        calculating the probability density function."""
        data = self.data

        for cat in data:
            columns = [x for x in self.column_format if x != 'class']
            self.means.setdefault(cat, dict.fromkeys(columns, 0))
            self.sample_standard_deviations.setdefault(
                cat, dict.fromkeys(columns, 0)
            )

            for column in data[cat]:
                array   = [float(x) for x in data[cat][column]]
                length  = float(len(array))
                average = sum(array) / length

                self.means[cat][column] = average
                deviations = [pow(x - average, 2) for x in array]
                self.sample_standard_deviations[cat][column] = (
                    pow(sum(deviations) / (length - 1), 0.5)
                )

    def probabilities_from_data(self):
        classification_occurence = {}
        new_dict = {}

        for classification, dict1 in self.data.items():
            classification_occurence.setdefault(classification, 0)

            all_dicts = {}

            for key, array in dict1.items():
                all_dicts.setdefault(key, {})

                count_dict = {}
                for value in array:
                    count_dict.setdefault(value, 0)
                    count_dict[value] += 1

                prob_dict = {}
                for k, v in count_dict.items():
                    prob_dict[k] = float(v) / len(array)

                    classification_occurence[classification] = len(array)

                all_dicts[key] = prob_dict

            new_dict[classification] = all_dicts

        self.data = new_dict

        for k, v in classification_occurence.items():
            total = sum([y for x, y in classification_occurence.items()])
            self.prior_probabilities[k] = v / float(total)

    def probability_density_function(self, array):
        """P(x.i | y.i) = (1 / (sqrt(2 * pi) * o.i,j)) *
        e^-((x.i - u.i,j)^2 / (2 * o.i,j^2))"""
        hypotheses = []

        for cat in self.data:
            probability_product = 1

            index = 0
            for column in self.column_format:
                if column == 'class':
                    continue

                x_i = array[index]
                avg = self.means[cat][column]
                ssd = self.sample_standard_deviations[cat][column]
                if ssd == 0:
                    ssd = 1 * pow(10, -100)

                part_1      = 1 / (pow(2 * math.pi, 0.5) * ssd)
                exponent    = -1 * pow(x_i - avg, 2) / (2 * pow(ssd, 2))
                part_2      = pow(math.e, exponent)

                probability = part_1 * part_2
                probability_product *= probability

                index +=1

            hypotheses.append((probability_product, cat))

        return max(hypotheses)[1]

    def reset_data(self):
        super(NaiveBayes, self).reset_data()
        self.data                            = {}
        self.means                           = {}
        self.prior_probabilities             = {}
        self.sample_standard_deviations      = {}
        self.test_means                      = {}
        self.test_sample_standard_deviations = {}

    def test_training_bucket(self):
        correct = 0

        for i in range(self.number_of_buckets):
            print('Testing bucket {}'.format(i))

            self.reset_data()

            self.load_training_buckets(i)
            self.load_test_buckets(i)

            self.calculate_means_and_sample_standard_deviation()

            for value in self.test_data:
                cat    = value[0]
                vector = value[1]

                classified = self.probability_density_function(vector)

                if cat == classified:
                    correct += 1

        length = sum([len(v.items()[0][1]) for (k, v) in self.data.items()])
        total_length = length + len(self.test_data)
        accuracy     = float(correct) / total_length
        percentage   = accuracy * 100

        print('')
        print('{} percent accurate'.format(percentage))
        print('total of {} instances'.format(total_length))

        return accuracy

def ihealth():
    c = NaiveBayes(
        'ihealth', [
            'main interest',
            'current exercise level',
            'how motivated',
            'comfortable with tech devices',
            'class'
        ]
    )
    c.load_all_training_buckets()
    c.probabilities_from_data()
    r = c.classify(['health', 'moderate', 'moderate', 'yes']) #=> i500
    print(r)

def house_votes():
    c = NaiveBayes(
        'house_votes', [
            'class',
            'handicapped_infants',
            'water_project_cost_sharing',
            'adoption_of_the_budget_resolution',
            'physician_fee_freeze',
            'el_salvador_aid',
            'religious_groups_in_schools',
            'anti_satellite_test_ban',
            'aid_to_nicaraguan_contras',
            'mx_missle',
            'immigration',
            'synfuels_corporation_cutback',
            'education-spending',
            'superfund_right_to_sue',
            'crime',
            'duty_free_exports',
            'export_administration_act_south_africa'
        ]
    )
    c.load_all_training_buckets()
    c.probabilities_from_data()
    r = c.classify('y y y n n y y y y y n n n n n y'.split(' '))
    print(r)

def pima():
    c = NaiveBayes('pima2', [
        'pregnant',
        'plasma_glucose_concentration',
        'blood_pressure',
        'triceps_skin_fold_thickness',
        'serum_insulin_level',
        'body_mass_index',
        'diabetes_pedigress_function',
        'age',
        'class'
    ])
    # c.load_all_training_buckets()
    # c.calculate_means_and_sample_standard_deviation()
    # c.probability_density_function([4, 111, 72, 47, 207, 37.1, 1.390, 56])
    c.test_training_bucket()

def mpg():
    c = NaiveBayes('mpg', [
        'class',
        'cylinders',
        'ci',
        'horsepower',
        'weight',
        'zero_to_60'
    ])
    c.test_training_bucket()

# ihealth()
# house_votes()
# pima()
# mpg()
