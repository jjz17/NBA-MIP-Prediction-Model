from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv(f'..{os.path.sep}data{os.path.sep}wrangled_data.csv')

features = data.drop(['Season', 'Outcome', 'Player'], axis=1)
target = data['Outcome']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=3000)

# Create the scaler
scaler = MinMaxScaler()

# Fit the scaler to the training data(features only)
scaler.fit(X_train)

# Transform X_train and X_test based on the (same) scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Replace any potential NaN with 0
X_train_scaled[np.isnan(X_train_scaled)] = 0
X_test_scaled[np.isnan(X_test_scaled)] = 0



# Parameter grids for Validation/Optimization
ridge_param_grid = {"alpha":[0.001, 0.01, 0.1, 1, 10, 100]}
lasso_param_grid = {"alpha":[0.001, 0.01, 0.1, 1, 10, 100]}
knn_param_grid = {"n_neighbors":[1, 5, 10], "metric": ['euclidean', 'manhattan', 'minkowski']}

# Dictionary of models with their parameter grids
estimators = {
    'Ridge': [Ridge(), ridge_param_grid],
    'Lasso': [Lasso(), lasso_param_grid],
    'k-Nearest Neighbor': [KNeighborsRegressor(), knn_param_grid]}

# Dictionaries to store optimized model objects
best_models = {}
best_selected_features_models = {}


# Initial Model Performance Analysis

print("Initial Results for Models Trained on All Features\n")
for estimator_name, estimator_objects in estimators.items():
    estimator_model = estimator_objects[0]

    model = estimator_model.fit(X=X_train_scaled, y=y_train)

    # Prediction results
    print(estimator_name + ":\n")
    print("\tR-squared value for training set: ", r2_score(y_train, model.predict(X_train_scaled)))
    print("\tMean-squared-error value for training set: ", mean_squared_error(y_train, model.predict(X_train_scaled)))
    print("\n")
    print("\tR-squared value for testing set: ", r2_score(y_test, model.predict(X_test_scaled)))
    print("\tMean-squared-error value for testing set: ", mean_squared_error(y_test, model.predict(X_test_scaled)))
    print("\n")


# Training and Validation with all Features using GridSearchCV

# Training and Validation using all Features

scoring = {"Max Error": "max_error", "R-squared": "r2"}

print("Results for Best Models Trained on All Features\n")
for estimator_name, estimator_objects in estimators.items():
    estimator_model = estimator_objects[0]
    param_grid = estimator_objects[1]

    grid_search = GridSearchCV(estimator_model, param_grid, scoring=scoring, refit='R-squared', return_train_score=True,
                               cv=5)

    # Fit the grid search object on the training data (CV will be performed on this)
    grid_search.fit(X=X_train_scaled, y=y_train)

    # Grid search results
    print(estimator_name + ":\n")
    print("\tBest estimator: ", grid_search.best_estimator_)
    print("\tBest parameters: ", grid_search.best_params_)
    print("\tBest cross-validation score: ", grid_search.best_score_)
    print("\n")

    model = grid_search.best_estimator_
    print("\tR-squared value for training set: ", r2_score(y_train, model.predict(X_train_scaled)))
    print("\tMean-squared-error value for training set: ", mean_squared_error(y_train, model.predict(X_train_scaled)))
    print("\n")

    # Add the best model to dictionary
    best_models[estimator_name] = grid_search.best_estimator_

# Best hyperparameters for each model trained on all features
print(best_models)


# Recursive Feature Elimination (RFE)

select = RFE(DecisionTreeRegressor(random_state = 3000), n_features_to_select = 3)

# Fit the RFE selector to the training data
select.fit(X_train_scaled, y_train)

# Transform training and testing sets so only the selected features are retained
X_train_scaled_selected = select.transform(X_train_scaled)
X_test_scaled_selected = select.transform(X_test_scaled)

# Determine selected features
selected_features = [feature for feature, status in zip(features, select.get_support()) if status == True]
print('Selected features:')
for feature in selected_features:
    print('\t' + feature)

# Extract selected features from the training and testing sets
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]


# Create the scaler
scaler = MinMaxScaler()

# Fit the scaler to the training data(features only)
scaler.fit(X_train_selected)

# Transform X_train and X_test based on the (same) scaler
X_train_selected_scaled = scaler.transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Replace any potential NaN with 0
X_train_selected_scaled[np.isnan(X_train_selected_scaled)] = 0
X_test_selected_scaled[np.isnan(X_test_selected_scaled)] = 0



# Training and Validation with selected Features using GridSearchCV

# Training and Validation using selected Features

print("Results for Best Models Trained on Selected Features\n")
for estimator_name, estimator_objects in estimators.items():
    estimator_model = estimator_objects[0]
    param_grid = estimator_objects[1]

    grid_search = GridSearchCV(estimator_model, param_grid, scoring=scoring, refit='R-squared', return_train_score=True,
                               cv=5)

    # Fit the grid search object on the training data (CV will be performed on this)
    grid_search.fit(X=X_train_selected_scaled, y=y_train)

    # Grid search results
    print(estimator_name + ":\n")
    print("\tBest estimator: ", grid_search.best_estimator_)
    print("\tBest parameters: ", grid_search.best_params_)
    print("\tBest cross-validation score: ", grid_search.best_score_)
    print("\n")

    model = grid_search.best_estimator_
    print("\tR-squared value for training set: ", r2_score(y_train, model.predict(X_train_selected_scaled)))
    print("\tMean-squared-error value for training set: ",
          mean_squared_error(y_train, model.predict(X_train_selected_scaled)))
    print("\n")

    # Add the best model to dictionary
    best_selected_features_models[estimator_name] = grid_search.best_estimator_

# Best hyperparameters for each model trained on selected features
print(best_selected_features_models)


# Model Testing

# Testing tuned algorithms on testing set (all features)

print("Final Results for Models Trained on All Features")
for model_name, model in best_models.items():
    print(model_name + ":")
    print("\tR-squared value for testing set: ", r2_score(y_test, model.predict(X_test_scaled)))
    print("\tMean-squared-error value for testing set: ", mean_squared_error(y_test, model.predict(X_test_scaled)))
    print("\n")

# Testing tuned algorithms on testing set (selected features)

print("\nFinal Results for Models Trained on Selected Features")
for model_name, model in best_selected_features_models.items():
    print(model_name + ":")
    print("\tR-squared value for testing set: ", r2_score(y_test, model.predict(X_test_selected_scaled)))
    print("\tMean-squared-error value for testing set: ",
          mean_squared_error(y_test, model.predict(X_test_selected_scaled)))
    print("\n")