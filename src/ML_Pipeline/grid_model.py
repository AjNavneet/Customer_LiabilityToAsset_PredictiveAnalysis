# Import necessary libraries and suppress warnings
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter(action='ignore')

# Define a function to perform grid search using different models
def grid_model(X_train, y_train, param_grid, model_name):
    # Dictionary mapping model names to their corresponding scikit-learn classifiers
    model_dict = {
        'logistic_reg': LogisticRegression,
        'naive_bayes': GaussianNB,
        'svm_model': svm.SVC,
        'decision_tree': DecisionTreeClassifier,
        'rfcl': RandomForestClassifier,
    }

    # Create an instance of the selected model
    model = model_dict[model_name]()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Perform grid search with cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, refit=True, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best estimator with optimal hyperparameters
    final_predictor = grid_search.best_estimator_
    
    return final_predictor
