from p05_opt_stratified_sampling import *
import matplotlib.pylab as plt

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

# option s: the radius of each circle represents the district's population
# option c: the color represents the price
# option cmap: a predefined color map, called jet, which ranges from blue (low values) to red (high prices)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

# looking for correlations
corr_matrix = housing.corr()
# look at how much each attribute correlates with the median house value
print("\n")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.tools.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
