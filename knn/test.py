from main import KNN

X_train = [
    [4.6,  3.1,  1.5,  0.2],
    [5.9,  3.0,   5.1,  1.8],
]
y_train = [0, 2]

X_test = [[5.8,  2.8,  5.1,  2.4]]
y_test = [2]


def setup():
    clf = KNN()

    clf.fit(X_train, y_train)

    return clf


def test_KNN_should_be_initialised_with_n_neighbors():
    clf = setup()

    assert clf.n_neighbors == 1


def test_should_be_able_to_pass_training_data_to_classifier():
    clf = setup()

    assert clf.X_train == X_train
    assert clf.y_train == y_train


def test_predict_should_return_label_for_test_data():
    clf = setup()

    predictions = clf.predict(X_test)

    assert predictions == y_test
