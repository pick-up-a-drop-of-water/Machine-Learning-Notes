from p05_opt_stratified_sampling import strat_train_set
import pandas as pd
import numpy as np

# Note that drop() create a copy of the data
# and does not affect strat_train_set
housing = strat_train_set.drop("median_house_value", axis=1)    # drop labels for train set
housing_labels = strat_train_set["median_house_value"].copy()

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print(sample_incomplete_rows)

# fill NaN with median value
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
print(sample_incomplete_rows)

# Data Cleaning
# to handle missing value using a handy class of Scikit-Learn
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
# since the median can only be computed on numerical attributes, we need
# to create a copy of the data without the text attribute ocean_proximity
housing_num = housing.drop("ocean_proximity", axis=1)
# now we can fit the imputer instance to the training data using fit() method
imputer.fit(housing_num)
# then we can use this "trained" imputer to transform the training set
# by replacing missing values by the learned medians
X = imputer.transform(housing_num)
# the result is a plain Numpy array containing the transformed features.
# put it back into Pandas DataFrame, by this:
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# Handling Text and Categorical Attributes
# convert text labels to number
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
housing_cat = housing[["ocean_proximity"]]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print("\nThis is Categorical Attribute Encoded:")
print(housing_cat_encoded)
# we can look at the mapping
print("This is the corespoding mapping:")
print(encoder.classes_)


# Another Way to Encode Called One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
print("\nOne-Hot Encoding Completed! (in p8...for_ML_algorithms file)")
# to convert it to a (dense) Numpy array, just call the toarray() method
# housing_cat_1hot.toarray()

'''
# using a simple way to complete above actions
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot
'''

# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin

# rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room      # no *args or **kargs

    def fit(self, X, y=None):
        return self     # nothing to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
print(housing_extra_attribs.head())
