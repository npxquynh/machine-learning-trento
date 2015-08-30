import sys
import numpy
import sklearn
from sklearn import datasets, cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import utility
from pdb import set_trace as debugger

DEFAULT_SEED = 1234
SECONDARY_SEED = 5678

class Metrics():
    def __init__(self, y_test, pred):
        self.accuracy = sklearn.metrics.accuracy_score(y_test, pred)
        self.f1 = sklearn.metrics.f1_score(y_test, pred)
        self.auc = sklearn.metrics.roc_auc_score(y_test, pred)

    def __repr__(self):
        return '%.3f\t%.3f\t%.3f' % (
            self.accuracy,
            self.f1,
            self.auc)


class MetricsList():
    def __init__(self):
        self.metrics_list = list()

    def calculate_mean(self):
        accuracy_scores = [x.accuracy for x in self.metrics_list]
        self.accuracy = numpy.mean(accuracy_scores)

        f1_scores = [x.f1 for x in self.metrics_list]
        self.f1 = numpy.mean(f1_scores)

        auc_scores = [x.auc for x in self.metrics_list]
        self.auc = numpy.mean(auc_scores)

    def append(self, metrics):
        self.metrics_list.append(metrics)

        # re-evaluate
        self.calculate_mean()

    def __getitem__(self, index):
        if index < len(self.metrics_list):
            return self.metrics_list[index]
        else:
            return None

    def __repr__(self):
        accuracy_scores = [round(x.accuracy,3) for x in self.metrics_list]
        f1_scores = [round(x.f1,3) for x in self.metrics_list]
        auc_scores = [round(x.auc, 3) for x in self.metrics_list]

        return '%.3f\t%.3f\t%.3f\n%s\n%s\n%s' % (
            self.accuracy,
            self.f1,
            self.auc,
            accuracy_scores,
            f1_scores,
            auc_scores)


class Data():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


class CrossValidation():
    def __init__(self, dataset, n_folds=10, random_state=None):
        self.dataset = dataset
        n = len(dataset[0])
        self.kfold = cross_validation.KFold(n, n_folds=n_folds, shuffle=True,
            random_state=random_state)

    def get_dataset(self, index):
        """Returns X_train, y_train, X_test, y_test
        """
        if index >= self.kfold.n_folds:
            return None

        for (i, (train_index, test_index)) in enumerate(self.kfold):
            if i == index:
                X_train, X_test = self.dataset[0][train_index], self.dataset[0][test_index]
                y_train, y_test = self.dataset[1][train_index], self.dataset[1][test_index]

                return Data(X_train, y_train, X_test, y_test)

        return None


class ExperimentResult():
    def __init__(self):
        self.configuration = list()
        self.metrics_list = MetricsList()

        # self.accuracy_scores = list()
        # self.f1_scores = list()
        # self.auc_scores = list()

    def add(self, metrics, **config):
        self.configuration.append(config)
        self.metrics_list.append(metrics)

        # self.accuracy_scores.append(metrics.accuracy)
        # self.f1_scores.append(metrics.f1)
        # self.accuracy_scores.append(metrics.auc)

    def __repr__(self):
        return ''

    def print_result(self):
        tracked_key = set()

        # in this order: ('n_samples', 'n_features', 'NB', 'SVC', 'RF')
        output = list()

        for conf in self.configuration:
            temp_row = [conf['n_samples'], conf['n_features']]

            key = '%s%%%s' % (conf['n_samples'], conf['n_features'])
            if key not in tracked_key:
                tracked_key.add(key)
                values = self._find_result('f1', conf['n_samples'],
                    conf['n_features'])
                if len(values) == 3:
                    [temp_row.append(v) for v in values]
                    output.append(temp_row)

        # Write to csv file
        numpy.savetxt('exp_result.csv', output, fmt='%.2f', delimiter=',')
        utility.print_output_for_latex(output)


    def _find_result(self, attr, n_samples=None, n_features=None):
        """Returns the list of value for attr from different algorithm. Filter according to n_samples and n_features
        """
        algorithms = ['GaussianNB', 'SVC', 'RandomForestClassifier']

        root_conf = {
            'n_samples': n_samples,
            'n_features': n_features,
        }
        result = list()
        for algo in algorithms:
            flag = False
            conf_to_match = root_conf
            conf_to_match['algorithm'] = algo
            for (index, conf) in enumerate(self.configuration):
                if conf == conf_to_match:
                    value = getattr(self.metrics_list[index], attr)
                    result.append(value)
                    flag = True
                    break

            if flag == False:
                result.append(-1)

        return result


class Experiment():
    def __init__(self, n_samples=[], n_features=[]):
        args = {
            'n_samples': 1000,
            'n_features': 10,
            'n_informative': 2,
            'n_redundant': 2,
            'n_repeated': 0,
            'n_classes': 2,
            'random_state': 1234,
        }

        self.datasets = list()
        self.descriptions = list()
        for s in n_samples:
            for f in n_features:
                args['n_samples'] = s
                args['n_features'] = f
                self.datasets.append(datasets.make_classification(**args))
                self.descriptions.append({
                    'n_samples': s,
                    'n_features': f,
                })

        self.result = ExperimentResult()

    def run(self, n_folds=10, random_state=1234):
        for (index, dataset) in enumerate(self.datasets):
            rep = Repetition(dataset)
            # Quick
            rep.run(n_folds=n_folds, random_state=random_state)
            self.add_repetition_result(rep, index)

    def add_repetition_result(self, rep, dataset_index):
        rep_config = self.descriptions[dataset_index]
        print "@@@ %s" % rep_config

        for (algo, metrics) in rep.total_metrics.iteritems():
            config = rep_config
            config['algorithm'] = algo
            self.result.add(metrics, **config)
            # print "algo = %s \t metrics = %s" % (algo, metrics)

    def print_result(self):
        print self.result.print_result()


class Repetition():
    def __init__(self, dataset):
        self.dataset = dataset

        """
        Manually design the steps for running 1 repetition.
        The experiment will contains running with 3 different classification
        algorithms
        """
        self.algorithms = ['GaussianNB', 'SVC', 'RandomForestClassifier']
        self.predefined_params = [
            {},
            {
                'C': [1e-2, 1e-1, 1e0, 1e1, 1e2],
                'length': 5,
            },
            {
                'n_estimators': [10, 100, 1000],
                'length': 3,
            },
        ]

        # For quick testing
        # self.algorithms = ['SVC']
        # self.predefined_params = [
        #     {
        #         'C': [1e-2, 1e-1, 1e0, 1e1, 1e2],
        #         'length': 5,
        #     },
        # ]

        self.total_metrics = dict()
        self.best_params = dict()
        for algo in self.algorithms:
            self.total_metrics[algo] = MetricsList()
            self.best_params[algo] = list()

    def run(self, n_folds=10, random_state=1234):
        cv = CrossValidation(self.dataset, n_folds=n_folds, random_state=random_state)
        for i in range(n_folds):
            print 'cross validation %i' % i
            self._run_with_different_algorithms(cv, i)

    def _run_with_different_algorithms(self, cv, index):
        data = cv.get_dataset(index)

        for (index, algorithm) in enumerate(self.algorithms):
            current_params = self.predefined_params[index]
            self._run_with_one_specific_algorithm(data, algorithm, current_params)

    def _run_with_one_specific_algorithm(self, data, algorithm, algorithm_params):
        l = algorithm_params.get('length', 0)
        if l == 0:
            metrics = ML.evaluate(data, algorithm)
            # print "*** %s" % algorithm
            # print metrics

            # save the repetition result
            self.total_metrics[algorithm].append(metrics)
            self.best_params[algorithm].append({})
        else:
            pe = ParameterEstimation(data, algorithm, algorithm_params)
            best_kwargs = pe.select_best_parameter()
            metrics = ML.evaluate(data, algorithm, **best_kwargs)
            # print "### %s \t %s" % (algorithm, best_kwargs)
            # print metrics

            # save the repetition result
            self.total_metrics[algorithm].append(metrics)
            self.best_params[algorithm].append(best_kwargs)


class ParameterEstimation():
    def __init__(self, data, algorithm, parameters):
        self.data = data
        self.algorithm = algorithm

        self.parameters = parameters

        self.n = self.parameters.get('length', 0)
        self.metrics_list = list()

        if self.n != 0:
            best_param = self.select_best_parameter()

    def select_best_parameter(self):
        l = self.parameters.get('length')

        inner_score = list()
        for i in range(l):
            kwargs = utility.build_kwargs(self.parameters, i)
            avg_f1_score = self._run_with_paramter(**kwargs)
            inner_score.append(avg_f1_score)

        best_param_index = numpy.argmax(inner_score)
        kwargs = utility.build_kwargs(self.parameters, best_param_index)
        return kwargs

    def _run_with_paramter(self, **kwargs):
        """Returns the average f1-score for running with **kwargs
        """
        # Perform inner cross validation
        n_folds = 5
        inner_kfold = CrossValidation([self.data.X_train, self.data.y_train],
            n_folds=n_folds, random_state=5678)

        inner_f1 = list()
        for j in range(n_folds):
            inner_data = inner_kfold.get_dataset(j)
            metrics = ML.evaluate(inner_data, self.algorithm)
            inner_f1.append(metrics.f1)

        return sum(inner_f1) / len(inner_f1)


class ML():
    @staticmethod
    def evaluate(data, algorithm_name, **kwargs):
        """Returns the object of Metrics class
        """
        algorithm = eval(algorithm_name)(**kwargs)
        algorithm.fit(data.X_train, data.y_train)
        pred = algorithm.predict(data.X_test)
        return Metrics(data.y_test, pred)

def short_run():
    n_samples_params = [1000]
    n_features_params = [10]

    exp = Experiment(n_samples_params, n_features_params)
    exp.run(n_folds=10, random_state=1234)
    exp.print_result()

def medium_run():
    n_samples_params = [1000, 2000]
    n_features_params = [10, 15]

    exp = Experiment(n_samples_params, n_features_params)
    exp.run(n_folds=10, random_state=1234)
    exp.print_result()

def long_run():
    n_samples_params = [1000, 2000, 5000]
    n_features_params = [10, 15, 20]

    exp = Experiment(n_samples_params, n_features_params)
    exp.run(n_folds=10, random_state=1234)
    exp.print_result()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        argument = sys.argv[1]

        if argument == 'longrun':
            long_run()
        elif argument == 'mediumrun':
            medium_run()
    else:
        short_run()