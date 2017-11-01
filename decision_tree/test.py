import pandas as pd
from main import DecisionTreeClassifier

# colour and size  features, target label
training_data = pd.DataFrame([
    {
        'Colour': 'Green',
        'Size': 3,
        'Label': 'Apple',
    },
    {
        'Colour': 'Yellow',
        'Size': 3,
        'Label': 'Apple',
    },
    {
        'Colour': 'Red',
        'Size': 1,
        'Label': 'Grape',
    },
    {
        'Colour': 'Red',
        'Size': 1,
        'Label': 'Grape',
    },
    {
        'Colour': 'Yellow',
        'Size': 3,
        'Label': 'Lemon',
    }
])

def test_contain_at_least_one_node():
    clf = DecisionTreeClassifier(training_data)
    assert clf.node != None

def test_node_should_contain_question():
    clf = DecisionTreeClassifier(training_data)
    assert clf.node.question != None

def test_node_question_should_contain_a_column_and_value():
    clf = DecisionTreeClassifier(training_data)
    print(clf.node)
    assert clf.node.question.column != None
    assert clf.node.question.value != None

def test_node_question_should_contain_first_column_name_and_value_from_training_data():
    clf = DecisionTreeClassifier(training_data)
    assert clf.node.question.column == training_data.columns[0]
    assert clf.node.question.value == training_data[training_data.columns[0]][0]