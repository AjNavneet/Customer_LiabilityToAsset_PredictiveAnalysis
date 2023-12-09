# Customers liability to Asset - Predictive Analysis

### Business Objective
Bank XYZ wants to expand its borrower base efficiently by improving campaign conversion rates using digital transformation strategies. Develop a machine learning model to identify potential borrowers for focused marketing.

---

### Aim

Build a machine learning model to predict potential customers who will convert from liability customers to asset customers.

---

### Data Description
The dataset consists of two CSV files:
- Data1 (5000 rows, 8 columns)
- Data2 (5000 rows, 7 columns)

#### Attributes:

1. Customer ID
2. Age
3. Customer Since
4. Highest Spend
5. Zip Code
6. Hidden Score
7. Monthly Average Spend
8. Level
9. Mortgage
10. Security
11. Fixed Deposit Account
12. Internet Banking
13. Credit Card
14. Loan on Card



---

### Tech Stack
- Language: `Python`
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`, `pickle`, `imblearn`

## Approach

1. Import required libraries and read the dataset.
2. Exploratory Data Analysis (EDA) including data visualization.
3. Feature Engineering:
   - Remove unnecessary columns
   - Handle missing values
   - Check for intercorrelation and remove highly correlated features
4. Model Building:
   - Split data into training and test sets
   - Train various models: Logistic Regression, Weighted Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest
5. Model Validation:
   - Evaluate models using common metrics: accuracy, confusion matrix, AUC, recall, precision, F1-score
6. Handle imbalanced data using imblearn.
7. Hyperparameter Tuning using GridSearchCV for Support Vector Machine Model.
8. Create the final model and make predictions.
9. Save the model with the highest accuracy as a pickle file.

---

## Modular Code Overview

Folders:
1. `input`: Contains the data (Data1 and Data2).
2. `src`: Contains modularized code for different project steps, including `engine.py` and `ML_Pipeline`.
3. `output`: Contains the best-fitted model.
4. `lib`: Reference folder with the original ipython notebook.

---
