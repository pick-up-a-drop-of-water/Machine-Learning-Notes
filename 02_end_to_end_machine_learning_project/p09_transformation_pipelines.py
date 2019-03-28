from p08_prepare_data_for_ML_algorithms import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The Pipeline constructor takes a list of name/estimation pairs defining a sequence of steps.
num_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

'''
Now, we have a pipeline for numerical values,and we also need to 
apply the LabelBinarizer on the categorical values: how can we join 
these transformations into a single pipeline? 
Scikit-Learn provides a FeatureUnion class for this.
All full pipeline handling both numerical and categorical attributes may 
look like this:
'''
from p00_preparation import CategoricalEncoder
from sklearn_features.transformers import DataFrameSelector
from sklearn.pipeline import FeatureUnion


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

# we can run rhe whole pipeline simply:
housing_prepared = full_pipeline.fit_transform(housing)
