#from p01_opt_download_the_data import HOUSING_PATH, os
import os
import pandas as pd

HOUSING_PATH = "datasets/housing"
# Load the data using Pandas
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()



