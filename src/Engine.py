# Import required packages
import pickle
from sklearn.model_selection import train_test_split
from ML_Pipeline.utils import read_data, merge_dataset, drop_col, null_values
from ML_Pipeline.train_model import train_model
from ML_Pipeline.grid_model import grid_model
# import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from scipy.stats import zscore

# Read initial datasets
data1 = read_data("../input/Data1.csv")
data2 = read_data("../input/Data2.csv")

# Merge the datasets
final_data = merge_dataset(data1, data2, join_type='inner', on_param='ID')

# Drop columns 
final_data = drop_col(final_data, ['ID', 'ZipCode', 'Age'])

# Drop null values
final_data = null_values(final_data)

# Train-test split (75-25) on imbalanced data
x_train, x_test, y_train, y_test = train_test_split(final_data.drop(['LoanOnCard'], axis=1),
                                                    final_data['LoanOnCard'],
                                                    test_size=0.3,
                                                    random_state=1)

# In order to balance the data

# Convert attributes to Z scale
XScaled = final_data.drop(['LoanOnCard'], axis=1).apply(zscore)

# Summarize class distribution
counter = Counter(final_data['LoanOnCard'])

# Define pipeline for balancing data
over = SMOTE(sampling_strategy=0.3, random_state=1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# Transform the dataset
Xb, Yb = pipeline.fit_resample(XScaled, final_data['LoanOnCard'])

# Summarize the new class distribution
counter = Counter(Yb)

# Split balanced data into train and test sets
x_trainb, x_testb, y_trainb, y_testb = train_test_split(Xb, Yb, test_size=0.3, random_state=1)

# Choose hyperparameters using Grid Search
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.25, 0.01],
              'kernel': ['rbf', 'poly', 'sigmoid']}

model_grid_search = grid_model(x_trainb, y_trainb, param_grid, 'svm_model')
print(model_grid_search)

# Train the model
model = train_model(x_trainb, y_trainb, x_testb, y_testb)
print(model)
pickle.dump(model, open('../output/finalized_model.sav', 'wb'))
