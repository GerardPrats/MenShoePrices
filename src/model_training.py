import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib


# Define the Decision Tree function:
def decisionTreeShoes(crit, split, depth, xtest, xval, ytest, yval):
    tree = DecisionTreeClassifier(criterion=crit, splitter=split, max_depth=depth)
    tree = tree.fit(xtest, ytest)

    ypred = tree.predict(xval)
    accuracy = metrics.accuracy_score(yval, ypred) * 100.0
    return accuracy


def split_menshoe(df):
    y = df["reviews"]
    X = df.drop(labels=["reviews", "brand", "manufacturernumber", "prices_currency"], axis=1)

    x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=0.8)

    # Define the different possible parameters of the decision tree:
    criterion = ["gini", "entropy"]
    splitters = ["best", "random"]
    max_depth = range(1, df.shape[1])

    # Test them all out to see what the accuracy is:
    bestacc = 0.0
    parameters = []
    for c in criterion:
        for s in splitters:
            for d in max_depth:
                acc = decisionTreeShoes(c, s, d, x_t, x_v, y_t, y_v)
                if acc > bestacc:
                    bestacc = acc
                    parameters = [c, s, d]

    print("Best parameters: { ", parameters[0], ", ", parameters[1], ", ", str(parameters[2]), " }")
    print("Accuracy: ", bestacc)

    # Create decision trees with the best parameters found:
    treeglobal = DecisionTreeClassifier(criterion=parameters[0], splitter=parameters[1], max_depth=parameters[2])
    treeglobal = treeglobal.fit(x_t, y_t)

    joblib.dump(treeglobal, "./models/finalized_tree.joblib")

    # Probabilities of the predictions:
    yprobs = treeglobal.predict_proba(x_v)

    # Building ROC curve for Global Tree:
    fpr, tpr, threshold = metrics.roc_curve(y_v, yprobs[:, 1])
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.title("ROC Curve")
    plt.plot(fpr, tpr, 'b', label="AUC = %f" % roc_auc)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()
