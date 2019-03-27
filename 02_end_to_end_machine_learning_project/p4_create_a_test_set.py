from p2_load_the_data import housing

# Also, here comes the simplest way to split dataset
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


'''
import numpy as np
import hashlib

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]
    
# The housing dataset does not have an identifier column.
# The simplest solution is to use the row index as the ID
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
print("\n\t", len(train_set), "train +", len(test_set), "test")
    
'''


'''
# The normal method to split dataset into train and test set
train_set, test_set = split_train_test(housing, 0.2)
print("\n\t", len(train_set), "train +", len(test_set), "test")
'''


'''
# You can try to use the most stable features to build a unique identifier
# For example, a district's latitude and longitude are guaranteed to be stable for a million years
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
'''

