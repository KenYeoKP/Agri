# Import necessary libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedKFold, cross_validate, 
                                     RandomizedSearchCV, GridSearchCV)
from sklearn.metrics import (make_scorer, accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import time
import warnings
warnings.filterwarnings('ignore')

#####################################################################
# Create an engine to connect to the SQLite database
db_path = 'data/agri.db'
engine = create_engine(f'sqlite:///{db_path}')

# Define your SQL query to fetch all data from a specific table
query = """
SELECT * FROM farm_data;
"""

# Execute the query and load results into a DataFrame
df = pd.read_sql_query(query, engine)

print('\nOriginal dataframe:\n')
print(df.info())

#####################################################################
# Data preparation
# Rename coloumns for ease of use
df.rename(columns={
    'System Location Code': 'location',
    'Previous Cycle Plant Type': 'prePlantType',
    'Plant Type': 'currentPlantType',
    'Plant Stage': 'plantStage',
    df.columns[4].strip(): 'temperature',
    'Humidity Sensor (%)': 'humidity',
    'Light Intensity Sensor (lux)': 'lightIntSensor',
    'CO2 Sensor (ppm)': 'co2Sensor',
    'EC Sensor (dS/m)': 'ecSensor',
    'O2 Sensor (ppm)': 'o2Sensor',
    'Nutrient N Sensor (ppm)': 'nSensor',
    'Nutrient P Sensor (ppm)': 'pSensor',
    'Nutrient K Sensor (ppm)': 'kSensor',
    'pH Sensor': 'phSensor',
    'Water Level Sensor (mm)': 'waterLevel'
}, inplace=True)

# Convert specified categorical columns to lower case
categorical_columns = ['location', 'prePlantType', 'currentPlantType', 'plantStage']

for col in categorical_columns:
    df[col] = df[col].str.lower()

# Define the order of the categories
stage_order = ['seedling', 'vegetative', 'maturity']

# Change the plantStage column to categorical with ordered categories
df['plantStage'] = pd.Categorical(df['plantStage'], categories=stage_order, ordered=True)

# Change the specified columns to category data type
df['location'] = df['location'].astype('category')
df['prePlantType'] = df['prePlantType'].astype('category')
df['currentPlantType'] = df['currentPlantType'].astype('category')
df['ecSensor'] = df['ecSensor'].astype('int64')

# Remove ' ppm' from the nSensor, pSensor, and kSensor columns
df['nSensor'] = df['nSensor'].str.replace(' ppm', '', regex=False).astype(float)
df['pSensor'] = df['pSensor'].str.replace(' ppm', '', regex=False).astype(float)
df['kSensor'] = df['kSensor'].str.replace(' ppm', '', regex=False).astype(float)

# Correct negative values by taking the absolute value using .loc
df.loc[:, 'temperature'] = df['temperature'].abs()
df.loc[:, 'lightIntSensor'] = df['lightIntSensor'].abs()
df.loc[:, 'ecSensor'] = df['ecSensor'].abs()

#####################################################################
# Handling duplicates
# Check for duplicates
duplicates = df.duplicated()

# Count the number of duplicate rows
duplicate_count = duplicates.sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

# Remove duplicates
df = df.drop_duplicates()
print('\nDuplicates removed and updated dataframe:\n')
print(df.info())

#####################################################################
# Impute median values for columns with nulls in df (excluding humidity)
for column in df.columns:
    if column != 'humidity' and df[column].isnull().any():
        median_value = df[column].median()
        df.loc[df[column].isnull(), column] = median_value

print('\nUpdated dataframe after imputation:\n')  
print(df.info())

#####################################################################
# Capture all rows with non-null humidity data and create a new DataFrame
df_humidity = df[df['humidity'].notnull()]

print('\nSubset of rows with humidity data:\n')  
print(df_humidity.info())

#####################################################################
# Feature engineering

# Combine currentPlantType and plantStage into a new column
df_humidity['plantTypeStage'] = df_humidity['currentPlantType'].astype(str) + '_' + df_humidity['plantStage'].astype(str)

# Convert the new column to categorical
df_humidity['plantTypeStage'] = df_humidity['plantTypeStage'].astype('category')

# Check the result
print(f'\nData type of plantTypeStage column: {df_humidity["plantTypeStage"].dtype}\n')
print(df_humidity[['currentPlantType', 'plantStage', 'plantTypeStage']].head())

#####################################################################
'''
Data pre-processing
Steps: (1)split data into training, validation and test sets, 
       (2)scaling features
'''
#####################################################################
# Split the data into training, validation, and testing sets
# Define the features and target variable
X = df_humidity.drop(columns=['location', 'prePlantType', 'currentPlantType', 'plantStage', 'plantTypeStage'])
y = df_humidity['plantTypeStage']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Display the shapes of each set
print("\nTraining set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)

#####################################################################
# Feature scaling and encoding
# Identify numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns

# Create preprocessor with specified categories
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
    ],
    remainder='passthrough'
)

#####################################################################
# Compare different classification models
# Define models to compare
models = [
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=42)),
    ('RandomForestClassifier', RandomForestClassifier(random_state=42)),
    ('LogisticRegression', LogisticRegression(random_state=42)),
    ('KNeighborsClassifier', KNeighborsClassifier()),
    ('SVC', SVC(random_state=42)),
    ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=42))
]

# Define the cross-validation strategy
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

results = {}
for name, model in models:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    start_time = time.time()
    
    cv_results = cross_validate(pipeline, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    results[name] = cv_results
    results[name]['runtime'] = runtime

# Display cross-validation results along with runtime
for name, result in results.items():
    print(f"\n{name} Model Performance:")
    print(f"Accuracy: {result['test_accuracy'].mean():.2f} +/- {result['test_accuracy'].std():.2f}")
    print(f"Precision: {result['test_precision'].mean():.2f} +/- {result['test_precision'].std():.2f}")
    print(f"Recall: {result['test_recall'].mean():.2f} +/- {result['test_recall'].std():.2f}")
    print(f"F1 Score: {result['test_f1'].mean():.2f} +/- {result['test_f1'].std():.2f}")
    print(f"Runtime: {result['runtime']:.2f} seconds")
print()

#####################################################################
# Hyperparameter tuning
# Define the classifier models
logistic_model = LogisticRegression(random_state=42)
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Define the parameter grid for Logistic Regression
logistic_param_grid = {
    'classifier__C': np.logspace(-3, 3, 7),
    'classifier__solver': ['liblinear', 'lbfgs', 'saga'],
}

# Define the parameter distribution for Decision Tree Classifier
dt_param_dist = {
    'classifier__max_depth': [None, 10, 20, 30, 40],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Create a pipeline for Logistic Regression
logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', logistic_model)
])

# Set up GridSearchCV for Logistic Regression
logistic_grid_search = GridSearchCV(estimator=logistic_pipeline,
                                     param_grid=logistic_param_grid,
                                     scoring='accuracy',
                                     cv=3,
                                     n_jobs=-1,
                                     verbose=2,
                                     return_train_score=True)

# Fit Grid Search for Logistic Regression
logistic_grid_search.fit(X_train, y_train)

# Print best parameters and best score for Logistic Regression
print("Best parameters for Logistic Regression:", logistic_grid_search.best_params_)
print(f"Best score (accuracy) for Logistic Regression: {logistic_grid_search.best_score_:.2f}")
print()

# Create a pipeline for Decision Tree Classifier
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', decision_tree_model)
])

# Set up RandomizedSearchCV for Decision Tree Classifier
dt_random_search = RandomizedSearchCV(estimator=dt_pipeline,
                                      param_distributions=dt_param_dist,
                                      scoring='accuracy',
                                      n_iter=50,
                                      cv=3,
                                      n_jobs=-1,
                                      verbose=2,
                                      random_state=42)

# Fit Random Search for Decision Tree Classifier
dt_random_search.fit(X_train, y_train)

# Print best parameters and best score for Decision Tree Classifier
print("Best parameters for Decision Tree:", dt_random_search.best_params_)
print(f"Best score (accuracy) for Decision Tree:, {dt_random_search.best_score_:.2f}")
print()

#####################################################################
# Model training with best parameters on Logistic Regression and Decision Tree
# Best parameters obtained from RandomizedSearchCV
best_logistic_params = {
    'solver': 'lbfgs',
    'C': 100.0
}

# Create a Logistic Regression model with the best parameters
logistic_model = LogisticRegression(random_state=42, **best_logistic_params)

# Create a pipeline for Logistic Regression
logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', logistic_model)
])

# Fit the Logistic Regression model with the best parameters
logistic_pipeline.fit(X_train, y_train)

# Print confirmation message
print("Logistic Regression model trained with best hyperparameters.")

# Evaluate Logistic Regression model on validation set
logistic_predictions = logistic_pipeline.predict(X_val)

# Print Logistic Regression Model Evaluation with validation set
print("\nLogistic Regression Model Evaluation with validation set:")
print(classification_report(y_val, logistic_predictions))
print(f"Accuracy with best hyperparameters: {accuracy_score(y_val, logistic_predictions):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_val, logistic_predictions))
print()

# Best parameters obtained from RandomizedSearchCV
best_dt_params = {
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# Create a Decision Tree Classifier model with the best parameters
decision_tree_model = DecisionTreeClassifier(random_state=42, **best_dt_params)

# Create a pipeline for Decision Tree Classifier
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', decision_tree_model)
])

# Fit the Decision Tree model with the best parameters
dt_pipeline.fit(X_train, y_train)

# Print confirmation message
print("Decision Tree model trained with best hyperparameters.")

# Evaluate Decision Tree model on validation set
dt_predictions = dt_pipeline.predict(X_val)

# Print Decision Tree Model Evaluation with validation set
print("\nDecision Tree Model Evaluation with Validation Set:")
print(classification_report(y_val, dt_predictions))
print(f"Accuracy with best hyperparameters: {accuracy_score(y_val, dt_predictions):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_val, dt_predictions))
print()

#####################################################################

# Evaluate Logistic Regression model on the test set
y_pred_logistic = logistic_pipeline.predict(X_test)

# Calculate metrics for Logistic Regression
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic, average='weighted')
recall_logistic = recall_score(y_test, y_pred_logistic, average='weighted')
f1_logistic = f1_score(y_test, y_pred_logistic, average='weighted')

# Print Logistic Regression Model Evaluation with Test Set
print("Logistic Regression Model Performance with Test Set:")
print(f"Accuracy: {accuracy_logistic:.2f} +/- 0.00")
print(f"Precision: {precision_logistic:.2f} +/- 0.00")
print(f"Recall: {recall_logistic:.2f} +/- 0.00")
print(f"F1 Score: {f1_logistic:.2f} +/- 0.00")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logistic))


# Evaluate Decision Tree model on the test set
y_pred_dt = dt_pipeline.predict(X_test)

# Calculate metrics for Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

# Print Decision Tree Model Evaluation with Test Set
print("\nDecision Tree Model Performance with Test Set:")
print(f"Accuracy: {accuracy_dt:.2f} +/- 0.00")
print(f"Precision: {precision_dt:.2f} +/- 0.00")
print(f"Recall: {recall_dt:.2f} +/- 0.00")
print(f"F1 Score: {f1_dt:.2f} +/- 0.00")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))  # Confusion matrix

# Conclusion
print("\nConclusion: Best model for categorising Plant Type-Stage is the Decision Tree Model.")







