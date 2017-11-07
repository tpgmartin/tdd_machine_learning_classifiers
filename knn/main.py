from scipy.spatial import distance
from collections import Counter

class KNN():

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            prediction = self.__closest(row)
            predictions.append(prediction)
        return predictions

    def __closest(self, row):
        distances = []
        for i in range(len(self.X_train)):
            dist = distance.euclidean(row, self.X_train[i])
            distances.append((self.y_train[i], dist))
        sorted_distances = sorted(distances, key=lambda x: x[1])
        return self.__vote(sorted_distances)

    def __vote(self, distances):
        # labels = []
        # for i in range(self.n_neighbors):
        #     labels.append(distances[i][0])
        # return Counter(labels).most_common(1)[0][0]
        return Counter(x[0] for x in distances[:self.n_neighbors]).most_common(1)[0][0]
