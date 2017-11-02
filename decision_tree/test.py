import pandas as pd
from main import DecisionTreeClassifier, Question

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
    assert clf.node.question.column != None
    assert clf.node.question.value != None

def test_question_should_contain_first_column_name_and_value_from_training_data():
    clf = DecisionTreeClassifier(training_data)
    # Pass in column by (string) name
    assert clf.node.question.column == 'Colour'
    # Get values for given column e.g. training_data['Colour']
    assert clf.node.question.value == training_data['Colour'][0]

def test_question_return_match_against_question_instance_for_string_values():
    question = Question('Colour', 'Green')
    assert question.match(training_data.iloc[0]) == True
    assert question.match(training_data.iloc[1]) == False

def test_question_return_match_against_question_instance_for_numeric_values():
    question = Question('Size', 3)
    assert question.match(training_data.iloc[0]) == True
    assert question.match(training_data.iloc[2]) == False