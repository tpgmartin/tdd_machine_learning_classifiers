import pandas as pd
from main import DecisionTreeClassifier, Leaf, Node, Question

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

# DecisionTreeClassifier class
def test_contain_at_least_one_node():
    clf = DecisionTreeClassifier(training_data)
    assert clf.node != None

# Leaf class
def test_leaf_should_contain_dictionary_of_labels_with_frequency():
    leaf = Leaf(training_data)
    assert leaf.predictions.equals(training_data['Label'].value_counts())

# Node class
def test_node_should_contain_question():
    node = Node(training_data)
    assert node.question != None

def test_node_should_partition_data_set_by_question():
    node = Node(training_data)
    assert node.true_branch.equals(training_data.loc[training_data['Colour'] == 'Green'])
    assert node.false_branch.equals(training_data.drop([0]))

# Question class
def test_node_question_should_contain_a_column_and_value():
    node = Node(training_data)
    assert node.question.column != None
    assert node.question.value != None

def test_question_should_contain_first_column_name_and_value_from_training_data():
    # Pass in column by (string) name
    node = Node(training_data)
    assert node.question.column == 'Colour'
    assert node.question.value == training_data['Colour'][0]

def test_question_return_match_against_question_instance_for_string_values():
    question = Question('Colour', 'Green')
    assert question.match(training_data.iloc[0]) == True
    assert question.match(training_data.iloc[1]) == False

def test_question_return_match_against_question_instance_for_numeric_values():
    question = Question('Size', 3)
    assert question.match(training_data.iloc[0]) == True
    assert question.match(training_data.iloc[2]) == False