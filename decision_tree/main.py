class DecisionTreeClassifier(object):

    def __init__(self, data):
        self.node = Node(data)


class Node(object):

    def __init__(self, data):
        self.question = Question(data.columns[0], data['Colour'][0])

class Question(object):

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, row):
        row_value = row[self.column]
        if isinstance(row_value, int):
            return row_value >= self.value
        else:
            return row_value == self.value
