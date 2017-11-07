import pytest
from main import KNN

X_train = [
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2]
]
y_train = [0, 1, 1, 2, 2, 2, 2]


@pytest.mark.parametrize(('n_neighbors'),[1,3,5])
def test_KNN_should_be_initialised_with_n_neighbors(n_neighbors):
    clf = KNN(n_neighbors)

    clf.fit(X_train, y_train)

    assert clf.n_neighbors == n_neighbors

@pytest.mark.parametrize(('n_neighbors'),[1,3,5])
def test_should_be_able_to_pass_training_data_to_classifier(n_neighbors):
    clf = KNN(n_neighbors)

    clf.fit(X_train, y_train)

    assert clf.X_train == X_train
    assert clf.y_train == y_train

X_test = [[0, 0, 0, 0]]
@pytest.mark.parametrize(('n_neighbors', 'y_test'),[(1, [0]),(3, [1]), (7, [2])])
def test_predict_should_return_label_for_test_data(n_neighbors, y_test):
    clf = KNN(n_neighbors)

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    assert predictions == y_test
