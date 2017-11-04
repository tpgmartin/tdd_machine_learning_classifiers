import pandas as pd

class DecisionTreeClassifier(object):

    def __init__(self, data):
        self.node = Node(data)

class Leaf((object)):

    def __init__(self, data):
        self.predictions = self.class_counts(data)

    def class_counts(self, data):
        return data['Label'].value_counts()



class Node(object):

    def __init__(self, data):
        self.question = Question(data.columns[0], data['Colour'][0])
        self.true_branch, self.false_branch = self.partition(data, self.question)

    def partition(self, data, question):
        true_rows, false_rows = [], []
        for _, row in data.iterrows():
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return pd.DataFrame(true_rows), pd.DataFrame(false_rows)

class Question(object):

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, row):
        row_value = row.get_value(self.column)
        if isinstance(row_value, int):
            return row_value >= self.value
        else:
            return row_value == self.value
