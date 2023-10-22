# Import necessary libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Function to calculate the accuracy score of a model
def evaluate_model(y_test, y_pred, method):
    if method == 'accuracy_score':
        score = accuracy_score(y_test, y_pred)
    else:
        print("Only available accuracy measure is accuracy_score.")
    return score


# Similarly create these metrics for precision and recall as well and use them instead

from sklearn.metrics import precision_score, recall_score

def evaluate_model_1(y_test, y_pred, method):
    if method == 'accuracy_score':
        score = accuracy_score(y_test, y_pred)
    elif method == 'precision_score':
        score = precision_score(y_test, y_pred)
    elif method == 'recall_score':
        score = recall_score(y_test, y_pred)
    else:
        print("Available metrics are accuracy_score, precision_score, and recall_score.")
        score = None
    return score

