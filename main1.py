# Import necessary libraries
import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
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
df_humidity['plantTypeStage'] = df_humidity[
    'currentPlantType'].astype(str) + '_' + df_humidity['plantStage'].astype(str)

# Convert the new column to categorical
df_humidity['plantTypeStage'] = df_humidity['plantTypeStage'].astype('category')

# Check the result
print(f'Data type of plantTypeStage column: {df_humidity["plantTypeStage"].dtype}\n')
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
X = df_humidity[['humidity', 'lightIntSensor', 'co2Sensor']]
y = df_humidity['temperature']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Display the shapes of each set
print("\nTraining set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)

#####################################################################

# Feature scaling
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

# Compare different regression models
# Define models to compare
models = [
    ('LinearRegression', LinearRegression()),
    ('Ridge', Ridge(random_state=42)),
    ('Lasso', Lasso(random_state=42)),
    ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=42)),
    ('RandomForestRegressor', RandomForestRegressor(random_state=42)),
    ('SVR', SVR()),
    ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=42))
]

# Define the cross-validation strategy
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the scoring metrics for regression
scoring = {
    'mae': make_scorer(mean_absolute_error),
    'mse': make_scorer(mean_squared_error),
    'r2': make_scorer(r2_score)
}

results = {}
for name, model in models:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
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
    print(f"Mean Absolute Error (MAE): {result['test_mae'].mean():.2f} +/- {result['test_mae'].std():.2f}")
    print(f"Mean Squared Error (MSE): {result['test_mse'].mean():.2f} +/- {result['test_mse'].std():.2f}")
    print(f'Root Mean Squared Error (RMSE): {np.sqrt(result["test_mse"].mean()):.2f} +/- {np.sqrt(result["test_mse"].std()):.2f}')
    print(f"R^2 Score: {result['test_r2'].mean():.2f} +/- {result['test_r2'].std():.2f}")
    print(f"Runtime: {result['runtime']:.2f} seconds")

print()

#####################################################################

# Hyperparameter tuning on Random Forest and Gradient Boosting Models
# Define the models
gb_model = GradientBoostingRegressor(random_state=42)
ridge_model = Ridge(random_state=42)

# Define the parameter grid for Ridge regression
ridge_param_grid = {
    'regressor__alpha': [0.1, 1.0, 10.0, 100.0],
    'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],
}

# Define the parameter grid for Gradient Boosting
gb_param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

# Create pipelines for both models
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', gb_model)
    ])
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ridge_model)
    ])

# Set up GridSearchCV for Ridge regression
ridge_grid_search = GridSearchCV(estimator=ridge_pipeline,
                                 param_grid=ridge_param_grid,
                                 scoring='neg_mean_squared_error',
                                 cv=3,
                                 n_jobs=-1,
                                 verbose=2)

# Fit Grid Search for Ridge regression
ridge_grid_search.fit(X_train, y_train)

# Print best parameters and best score for Ridge regression
print("Best parameters for Ridge regression:", ridge_grid_search.best_params_)
print("Best score (MSE) for Ridge regression:", -ridge_grid_search.best_score_)
print()

# Set up GridSearchCV for Gradient Boosting
gb_grid_search = GridSearchCV(estimator=gb_pipeline,
                               param_grid=gb_param_grid,
                               scoring='neg_mean_squared_error',
                               cv=3,
                               n_jobs=-1,
                               verbose=2)

# Fit Grid Search for Gradient Boosting
gb_grid_search.fit(X_train, y_train)

# Print best parameters and best score for Gradient Boosting
print("Best parameters for Gradient Boosting:", gb_grid_search.best_params_)
print("Best score (MSE) for Gradient Boosting:", -gb_grid_search.best_score_)

#####################################################################

# Model training with the best parameters on Ridge regression and Gradient Boosting models
# Define best parameters for Ridge regression
best_ridge_params = {
    'alpha': 1.0,
    'solver': 'saga', # Change this to 'saga' from 'auto' to speed up training
}

# Create and train the Ridge regression model
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(**best_ridge_params))
])

# Start measuring runtime
start_time = time.time()

# Train the Ridge regression model
ridge_pipeline.fit(X_train, y_train)

# End measuring runtime
end_time = time.time()
runtime = end_time - start_time

print()
print(f"Ridge regression model trained in {runtime:.2f} seconds.")
print("Ridge regression model trained with best parameters.")

# Make predictions on the validation set
ridge_predictions = ridge_pipeline.predict(X_val)

# Evaluate Ridge regression model on validation set
ridge_mae = mean_absolute_error(y_val, ridge_predictions)
ridge_mse = mean_squared_error(y_val, ridge_predictions)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_val, ridge_predictions)

# Print evaluation results
print("\nRidge Regression Model Performance on Validation Set:")
print(f"Mean Absolute Error (MAE): {ridge_mae:.2f}")
print(f"Mean Squared Error (MSE): {ridge_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {ridge_rmse:.2f}")
print(f"R-squared (R2): {ridge_r2:.2f}")

# Define best parameters for Gradient Boosting
best_gb_params = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 100,
    'subsample': 0.8, # Used 80% of samples for training
    'random_state': 42
}

# Create and train the Gradient Boosting model
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(**best_gb_params))
])

# Start measuring runtime
start_time = time.time()

# Train the Gradient Boosting model
gb_pipeline.fit(X_train, y_train)

# End measuring runtime
end_time = time.time()
runtime = end_time - start_time

print()
print(f"Gradient Boosting model trained in {runtime:.2f} seconds.")
print("Gradient Boosting model trained with best parameters.")

# Make predictions on the validation set
gb_predictions = gb_pipeline.predict(X_val)

# Evaluate Gradient Boosting model on validation set
gb_mae = mean_absolute_error(y_val, gb_predictions)
gb_mse = mean_squared_error(y_val, gb_predictions)
gb_rmse = np.sqrt(gb_mse)
gb_r2 = r2_score(y_val, gb_predictions)

# Print evaluation results
print("\nGradient Boosting Model Performance on Validation Set:")
print(f"Mean Absolute Error (MAE): {gb_mae:.2f}")
print(f"Mean Squared Error (MSE): {gb_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {gb_rmse:.2f}")
print(f"R-squared (R2): {gb_r2:.2f}")

#####################################################################

# Evaluate the Ridge model on the test set
y_pred_ridge = ridge_pipeline.predict(X_test)

# Calculate performance metrics for Ridge regression
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Model Evaluation with Test Set:")
print(f"Mean Absolute Error (MAE): {mae_ridge:.2f}")
print(f"Mean Squared Error (MSE): {mse_ridge:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_ridge:.2f}")
print(f"R-squared (R2): {r2_ridge:.2f}")

# Evaluate the Gradient Boosting model on the test set
y_pred_gb = gb_pipeline.predict(X_test)

# Calculate performance metrics for Gradient Boosting
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("\nGradient Boosting Model Evaluation with Test Set:")
print(f"Mean Absolute Error (MAE): {mae_gb:.2f}")
print(f"Mean Squared Error (MSE): {mse_gb:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_gb:.2f}")
print(f"R-squared (R2): {r2_gb:.2f}")

# Conclusion
print("\nConclusion: Best model for predicting temperature condition is the Gradient Boosting model.")


