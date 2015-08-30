import numpy
import sklearn
from sklearn import datasets, cross_validation
from pdb import set_trace as debugger

# get header
data = numpy.genfromtxt('./heart.dat', dtype=bool, delimiter=',', names=True)
original_header = ','.join(data.dtype.names)

# read CSV data
data = numpy.genfromtxt('./heart.dat', dtype=None, delimiter=',', skip_header=1)
y_data = data[:, 1]
X_data = data[:, 2:]

# randomly sample 200 samples from heart.dat
indices = set()
numpy.random.seed(1234)
while len(indices) != 200:
    indices.add(numpy.random.randint(0, len(y_data)))

indices = list(indices)
y_data = y_data[indices]
X_data = X_data[indices]
data = data[indices]

kfold = sklearn.cross_validation.StratifiedKFold(y_data, n_folds=2, shuffle=True, random_state=1234)

for (i, (train_index, test_index)) in enumerate(kfold):
    debugger()
    train_data = data[train_index]
    test_data = data[test_index]

    # write to csv
    numpy.savetxt('cv_%i_train.dat' % i, train_data, fmt='%i',
        delimiter=',', header=original_header)
    numpy.savetxt('cv_%i_test.dat' % i, test_data, fmt='%i',
        delimiter=',', header=original_header)
