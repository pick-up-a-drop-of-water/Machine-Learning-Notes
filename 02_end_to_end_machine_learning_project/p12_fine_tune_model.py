from p10_training_and_evaluating import *


# opt-Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# Let's look at the score of each hyperparameter combination tested during the grid search
cvres = grid_search.cv_results_
print("\nLet's look at the score of each hyperparameter combination")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print("\nBest Parameters for GridSearch is:", grid_search.best_params_)
print("Best Estimator for GridSearch is:", grid_search.best_estimator_)

# or by doing so
# pd.DataFrame(grid_search.cv_results_)

"""
# opt-Randomized Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

# Let's look at the score of each hyperparameter combination tested during the randomized search
cvres = rnd_search.cv_results_
print("\nLet's look at the score of each hyperparameter combination")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print("\nBest Parameters for GridSearch is:", rnd_search.best_params_)
print("Best Estimator for GridSearch is:", rnd_search.best_estimator_)

# the RandomForestRegressor can indicate the relative importance of each
# attributes for making accurate predictions
feature_importances = rnd_search.best_estimator_.feature_importances_
# Let's display these importance scores next to their corresponding attributes names
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))
"""
