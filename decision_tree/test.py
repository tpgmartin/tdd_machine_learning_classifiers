from main import DecisionTreeClassifier

# colour and size  features, target label
training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

def test_contain_at_least_one_node():
    clf = DecisionTreeClassifier()
    assert clf.node != None
