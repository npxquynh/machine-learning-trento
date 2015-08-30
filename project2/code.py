import numpy
import sklearn
from sklearn import datasets, cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import utility
from pdb import set_trace as debugger

class Metrics():
    def __init__(self, y_test, pred):
        self.accuracy = sklearn.metrics.accuracy_score(y_test, pred)
        self.f1 = sklearn.metrics.f1_score(y_test, pred)
        self.auc = sklearn.metrics.roc_auc_score(y_test, pred)

    def __str__(self):
        return '%.3f\t%.3f\t%.3f' % (
            self.accuracy,
            self.f1,
            self.auc)


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


class Experiment():
    def __init__(self, dataset):
        self.dataset = dataset


    def run(self):
        n_folds = 10
        cv = CrossValidation(self.dataset, n_folds=n_folds, random_state=1234)
        for i in range(n_folds):
            self._run_with_different_algorithms(cv, i)

    def _run_with_different_algorithms(self, cv, index):
        data = cv.get_dataset(index)

        algorithms = [GaussianNB, SVC, RandomForestClassifier]
        predefined_params = [
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

        for (index, algorithm) in enumerate(algorithms):
            current_params = predefined_params[index]
            self._run_with_one_specific_algorithm(data, algorithm, current_params)


    def _run_with_one_specific_algorithm(self, data, algorithm, algorithm_params):
        l = algorithm_params.get('length', 0)
        if l == 0:
            print "*** %s" % algorithm
            metrics = ML.evaluate(data, algorithm)
            print metrics
        else:
            pe = ParameterEstimation(data, algorithm, algorithm_params)
            best_kwargs = pe.select_best_parameter()
            print "### %s \t %s" % (algorithm, best_kwargs)
            metrics = ML.evaluate(data, algorithm, **best_kwargs)
            print metrics


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
        algorithm = algorithm_name(**kwargs)
        algorithm.fit(data.X_train, data.y_train)
        pred = algorithm.predict(data.X_test)
        return Metrics(data.y_test, pred)


if __name__ == '__main__':
    args = {
        'n_samples': 1000,
        'n_features': 10,
        'n_informative': 2,
        'n_redundant': 2,
        'n_repeated': 0,
        'n_classes': 2
    }
    dataset = datasets.make_classification(**args)


    experiment = Experiment(dataset)
    experiment.run()