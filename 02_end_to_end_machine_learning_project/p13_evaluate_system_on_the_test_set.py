from p05_opt_stratified_sampling import strat_test_set
from p12_fine_tune_model import *

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("\nEvaluation for the Final Model on the test set")
print("RMSE: ", final_rmse)
