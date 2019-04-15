import numpy as np
import matplotlib.pyplot as plt

# The following code solved displaying Chinese issue in plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=plt.cm.gray,
               interpolation="nearest")
    plt.axis("off")
    # plt.show()


def load_data():
    from sklearn.datasets import fetch_mldata
    print("*********************** start loading data ***********************")
    mnist = fetch_mldata('MNIST original')
    x, y_label = mnist["data"], mnist["target"]
    print("There are 70,000 images, and each image has 784 features.This is because\n"
          "each image is 28×28 pixels, and each feature simply represents\n\r"
          "one pixel’s intensity, from 0 (white) to 255 (black).")
    # exit()
    print("# Data Information is as follows: ")
    print("# X's shape: ", x.shape)
    print("# y's shape: ", y_label.shape)
    print("*********************** load data done! ***********************\n")
    return x, y_label


# Data Preparation
X, y = load_data()
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
"""
Let’s also shuffle the training set; this will guarantee that all cross-validation folds will
be similar (you don’t want one fold to be missing some digits).
"""
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
some_digit = X[36000]


# Multi-class Classification
# OvO strategy
# Create a multi-class classifier using the OvO strategy,
# based on a SGDClassifier
def ovo_test_for_sgd_classifier(x, y_label):
    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsOneClassifier
    ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, tol=np.infty, random_state=42))
    ovo_clf.fit(x, y_label)
    ovo_clf.predict([some_digit])
    print("There are", len(ovo_clf.estimators_), "classifiers made by SGDClassifier\n")


# comment to run faster
# ovo_test_for_sgd_classifier(X_train, y_train)

# based on a Random Forest
# This time Scikit-Learn did not have to run OvA or OvO because Random Forest
# classifiers can directly classify instances into multiple classes
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
# print(forest_clf.predict([some_digit]))     # 必须加中括号[]
# get the list of probabilities that the classifier assigned to
# each instance for each class
print("Random Forest 预测概率 for each class\n", forest_clf.predict_proba([some_digit]))


def evaluation_unscaled_and_scaled_data(x_unscaled, y_label):
    # unscaled:
    print("\n原始x的数据类型: ", type(x_unscaled), "\n原始y的数据类型:", type(y_label))
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import cross_val_score
    sgd_clf = SGDClassifier(max_iter=5, tol=np.infty, random_state=42)
    print("Unscaled Data: 3-fold evaluation accuracy values for SGDClassifier:\n", cross_val_score(sgd_clf, x_unscaled, y_label, cv=3, scoring="accuracy"))
    # scaled:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_unscaled.astype(np.float64))
    print("Scaled Data: 3-fold evaluation accuracy values for SGDClassifier:\n", cross_val_score(sgd_clf, x_scaled, y_label, cv=3, scoring="accuracy"))


# comment to run faster
# evaluation_unscaled_and_scaled_data(X_train, y_train)


def plot_matrix_and_error_analysis(x_unscaled, y_label):
    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_unscaled.astype(np.float64))
    sgd_clf = SGDClassifier(max_iter=5, tol=np.infty, random_state=42)
    y_train_pred = cross_val_predict(sgd_clf, x_scaled, y_label, cv=3)
    conf_mx = confusion_matrix(y_label, y_train_pred)
    # print("\nMuti-Class混淆矩阵:\n", conf_mx)
    # plot confusion matrix
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()

    # error analysis
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    # print("row_sums: ", row_sums)
    norm_conf_mx = conf_mx / row_sums
    # print("\nMuti-Class 正则化-混淆矩阵:\n", norm_conf_mx)
    # fill the diagonal with zeros to keep only the errors,
    # and let’s plot the result
    np.fill_diagonal(norm_conf_mx, 0)
    # print("\nfill 0 正则化-混淆矩阵:\n", norm_conf_mx)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()


plot_matrix_and_error_analysis(X_train, y_train)


# Multi-label Classification
def test_for_multilabel_classification(x, y_label):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import f1_score
    y_train_large = (y_label >= 7)
    y_train_odd = (y_label % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(x, y_multilabel)
    print(y[36000], "的预测结果是", knn_clf.predict([some_digit]))
    y_train_knn_pred = cross_val_predict(knn_clf, x, y_label, cv=2, verbose=3)
    print("\nF1 Score: ", f1_score(y_label, y_train_knn_pred, average="macro"))


# the following code may take a very long time
# comment to run faster
# test_for_multilabel_classification(X_train, y_train)


# Multi-output Classification
def test_for_multioutput_classification(x_train, x_test):
    import numpy.random as rnd
    from sklearn.neighbors import KNeighborsClassifier
    noise_train = rnd.randint(0, 100, (len(x_train), 784))
    noise_test = rnd.randint(0, 100, (len(x_test), 784))
    X_train_mod = x_train + noise_train
    X_test_mod = x_test + noise_test
    y_train_mod = x_train
    y_test_mod = x_test
    some_index = 5500
    # plot
    some_index = 5500
    plt.subplot(121)
    plt.title("添加噪音图像")
    plot_digit(X_test_mod[some_index])
    plt.subplot(122)
    plt.title("原始图像")
    plot_digit(y_test_mod[some_index])
    plt.show()
    # build a system that removes noise from images
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([X_test_mod[some_index]])
    plt.subplot(121)
    plt.title("原始图像")
    plot_digit(y_test_mod[some_index])
    plt.subplot(122)
    plt.title("去噪(预测)图像")
    plot_digit(clean_digit)
    plt.show()
    # print("\nraw_digit:", X_test_mod[some_index])
    # print("\nclean_digit:", clean_digit)


test_for_multioutput_classification(X_train, X_test)


