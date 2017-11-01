class DecisionTreeClassifier(object):

    def __init__(self, data):
        self.node = Node(data)


class Node(object):

    def __init__(self, data):
        self.question = Question(data)

class Question(object):

    def __init__(self, data):
        self.column = data.columns[0]
        self.value = data[data.columns[0]][0]
