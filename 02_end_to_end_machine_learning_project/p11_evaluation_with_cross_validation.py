from p10_training_and_evaluating import *
from sklearn.model_selection import cross_val_score


def display_scores(scores):
    print("\nCross Validation Result:")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# opt1
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
print("交叉验证概况如下：")
print(pd.Series(tree_rmse_scores).describe())


# opt-operation
'''
# We can save every model we experiment with, so we can
# come back easily to any model we want.

from sklearn.externals import joblib

joblib.dump(my_model, "my_model.pkl")

# and later ...
my_model_load = joblib.load("my_model.pkl")
'''
