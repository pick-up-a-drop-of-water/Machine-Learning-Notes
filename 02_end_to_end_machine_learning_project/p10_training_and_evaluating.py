from p9_transformation_pipelines import *
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)


# opt-model: Linear Regression
# Training
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
print("\nTraining Linear Regression Model...")

# Evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

housing_predictions = lin_reg.predict(housing_prepared)
print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t", list(some_labels))
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Evaluation for Linear Regression Model")
print("RMSE: ", lin_rmse)
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print("MAE: ", lin_mae)


# opt-model: Decision Tree
# Training
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
print("\nTraining Decision Tree Model...")

# Evaluation
housing_predictions = tree_reg.predict(housing_prepared)
print("Predictions:\t", tree_reg.predict(some_data_prepared))
print("Labels:\t", list(some_labels))
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Evaluation for Decision Tree Model")
print("RMSE: ", tree_rmse)


# opt-model: Random Forest
# Training
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
print("\nTraining Random Forest Model...")

# Evaluation
housing_predictions = forest_reg.predict(housing_prepared)
print("Predictions:\t", forest_reg.predict(some_data_prepared))
print("Labels:\t", list(some_labels))
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("Evaluation for Random Forest Model")
print("RMSE: ", forest_rmse)

"""
# opt-model: SVM
# Training
from sklearn.svm import SVR
svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
print("\nTraining SVM Model...")

# Evaluation
housing_predictions = svm_reg.predict(housing_prepared)
print("Predictions:\t", svm_reg.predict(some_data_prepared))
print("Labels:\t", list(some_labels))
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print("Evaluation for SVM Model")
print("RMSE: ", svm_rmse)
"""
