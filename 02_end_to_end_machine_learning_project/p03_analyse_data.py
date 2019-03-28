from p02_load_the_data import housing
import matplotlib.pylab as plt

housing.hist(bins=50, figsize=(20, 15))
plt.show()

'''
Following codes need to be typed in ipython console
Take a quick look at the data structure
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
'''
