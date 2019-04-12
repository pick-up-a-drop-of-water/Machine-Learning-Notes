import matplotlib.pyplot as plt
import numpy as np


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


def plot_some_digit(data):
    import matplotlib as mpl
    print("All we need to do is grab an instance's feature vector,reshape it to a 28x28 array,\n\r"
          "and display it using Matplotlib's imshow() function")
    some_digit = data[36000]
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap=mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()
    print("The digit's label: ", y[36000])
    print("*********************** plot some digits done! ***********************\n")


"""
But wait! We should always create a test set and set it aside before inspecting the data
closely. The MNIST dataset is actually already split into a training set (the first 60,000
images) and a test set (the last 10,000 images):
"""
X, y = load_data()
# plot_some_digit(X)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
"""
Let’s also shuffle the training set; this will guarantee that all cross-validation folds will
be similar (you don’t want one fold to be missing some digits).
"""
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# Training a Binary Classifier
# to identify one digit -- number 5
def sgd_classifier(x_data, y_label, need_predict_number):
    from sklearn.linear_model import SGDClassifier
    print("*********************** start training and predicting ***********************")
    sgd = SGDClassifier(max_iter=5, tol=np.infty, random_state=42)
    sgd.fit(x_data, y_label)
    print("predict result: ", sgd.predict([need_predict_number]))
    print("*********************** train and predict done! ***********************\n")


y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
some_digit = X[36000]
sgd_classifier(X_train, y_train_5, some_digit)


# Performance Measures
# Measuring Accuracy Using Cross-Validation
def cross_validation(model, x_train, y_train, cv=3, random_state=42):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    print("*********************** start cross validation ***********************")
    skfolds = StratifiedKFold(n_splits=cv, random_state=random_state)

    for train_index, test_index in skfolds.split(x_train, y_train):
        clone_clf = clone(model)
        x_train_folds = x_train[train_index]
        y_train_folds = (y_train[train_index])
        x_test_fold = x_train[test_index]
        y_test_fold = (y_train[test_index])

        clone_clf.fit(x_train_folds, y_train_folds)
        y_pred = clone_clf.predict(x_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print("每折预测结果：包含", len(y_pred), "个数据", y_pred)
        print("# Accuracy: ", n_correct / len(y_pred))
    print("*********************** cross validation done! ***********************\n")


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, tol=np.infty, random_state=42)
cross_validation(model=sgd_clf, x_train=X_train, y_train=y_train_5, cv=3, random_state=42)
# 等价于以下代码
# from sklearn.model_selection import cross_val_score
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


# 混淆矩阵
def get_confusion_matrix(y_train, y_train_pred):
    from sklearn.metrics import confusion_matrix
    print("*********************** start calculating confusion matrix ***********************")
    print("# 混淆矩阵: \n", confusion_matrix(y_train, y_train_pred))
    print("*********************** calculating confusion matrix done! ***********************\n")


# Precision, Recall, and F1 Score
def get_score(y_train, y_train_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score
    print("*********************** start calculating scores ***********************")
    print("# Precision Score: ", precision_score(y_train, y_train_pred))
    print("# Recall Score: ", recall_score(y_train, y_train_pred))
    print("# F1 Score: ", f1_score(y_train, y_train_pred))
    print("*********************** calculating scores done! ***********************\n")


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

get_confusion_matrix(y_train_5, y_train_pred)
get_score(y_train_5, y_train_pred)

'''
# get the scores of a instance in the training set
sgd_clf.fit(X_train, y_train_5)
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)
'''

# get the scores of all instances in the training set
# for compute precision and recall for all possible thresholds
# using the precision_recall_curve() function
from sklearn.metrics import precision_recall_curve


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.title("PR vs threshold curve", fontsize=16)
    plt.ylim([0, 1])
    plt.show()


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls[:-1], precisions[:-1], "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("PR curve", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.figure(figsize=(8, 6))
    plt.show()


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plot_precision_vs_recall(precisions, recalls)

y_train_pred_90 = (y_scores > 500000)
get_score(y_train_5, y_train_pred_90)


# The ROC Curve
# plots the TPR against FPR
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.show()


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
plot_roc_curve(fpr, tpr)
# AUC for comparing classfiers
from sklearn.metrics import roc_auc_score
print("# AUC for SGD: ", roc_auc_score(y_train_5, y_scores))


# Let's train a RandomForestClassifier and compare its ROC curve and ROC AUC
# score to the SGDClassifier
def plot_roc_curve_compare(x1, y1, x2, y2, label1=None, label2=None):
    plt.plot(x1, y1, 'r:', linewidth=2, label=label1)
    plt.plot(x2, y2, 'b-', linewidth=2, label=label2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="lower right", fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Comparing ROC curve', fontsize=16)
    plt.show()


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plot_roc_curve_compare(fpr, tpr, fpr_forest, tpr_forest, "SGD", "Random Forest")
print("# AUC for Random Forest: ", roc_auc_score(y_train_5, y_scores_forest))

print("\n----------------------> start calculating scores for Random Forest")
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
get_score(y_train_5, y_train_pred_forest)
