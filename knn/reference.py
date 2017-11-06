import math
from scipy.spatial import distance
from collections import Counter


def euc(a, b):
    return distance.euclidean(a,b)


class ScrappyKNN():

    def fit(self, X_train, y_train, k):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            # label = self.closest(row)
            predictions.append(label)
        return predictions

    # def closest(self, row):
    #   best_dist = euc(row, self.X_train[0])
    #   best_index = 0
    #   for i in range(1, len(self.X_train)):
    #     dist = euc(row, self.X_train[i])
    #     if dist < best_dist:
    #       best_dist = dist
    #       best_index = i
    #   return self.y_train[best_index]

    def closest(self, row):
        # best_dist = euc(row, self.X_train[0])
        # best_index = 0
        distances = []
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            distances.append([self.y_train[i], dist])
        return sorted(distances, key=lambda x: x[1])

    def vote(self, row):
        distances = self.closest(row)
        labels = []
        for i in range(self.k):
            labels.append(distances[i][0])
        return Counter(labels).most_common(1)[0][0]


from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

print(X_train[0])
print(y_train[0])
print(X_train[1])
print(y_train[1])
print(X_test[0])
print(y_test[0])

# from sklearn.neighbors import KNeighborsClassifier
clf = ScrappyKNN()

clf.fit(X_train, y_train, 3)

predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
