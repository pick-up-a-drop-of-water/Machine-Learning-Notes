from p5_opt_stratified_sampling import strat_train_set

housing = strat_train_set.copy()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
print("\nThe Correlation after Combing Some Attributes")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

import matplotlib.pyplot as plt
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()
print("\n")
print(housing.describe())
