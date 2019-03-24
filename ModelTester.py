
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np


def start_model_scoring(data_inputs, data_targets):

    # Divide data into training and validation.
    training_percentage = 0.70
    train_count = int(len(data_inputs) * training_percentage)

    train_inputs = data_inputs[:train_count]
    train_targets = data_targets[:train_count]

    validation_inputs = data_inputs[train_count:]
    validation_targets = data_targets[train_count:]

    #########################################################
    #   Decision Tree Classifier.                           #
    #########################################################
    clf = DecisionTreeClassifier(
        criterion = 'entropy',
        max_depth = 10,
        min_impurity_decrease = 0.0001
    )
    clf.fit(train_inputs, train_targets)
    print "DecisionTreeClassifier accuracy: " + str(clf.score(
        validation_inputs,
        validation_targets
    ))

    #########################################################
    #   Neural Network.                                     #
    #########################################################
    clf = MLPClassifier(
        activation = 'logistic',
        solver = 'lbfgs',
        hidden_layer_sizes = 10,
        learning_rate_init = 0.0001,
        max_iter = 100000
    )
    clf.fit(train_inputs, train_targets.ravel())
    print "MLPClassifier accuracy: " + str(clf.score(
        validation_inputs,
        validation_targets
    ))

    #########################################################
    #   Random Forest Classifier.                           #
    #########################################################
    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(train_inputs, train_targets.ravel())
    print "RandomForestClassifier accuracy: " + str(clf.score(
        validation_inputs,
        validation_targets
    ))

    #########################################################
    #   Gradient Tree Boosting Classifier.                  #
    #########################################################
    clf = GradientBoostingClassifier()
    clf.fit(train_inputs, train_targets.ravel())
    print "GradientBoostingClassifier accuracy: " + str(clf.score(
        validation_inputs,
        validation_targets
    ))

    #########################################################
    #   Support Vector Machine.                             #
    #########################################################
    clf = SVC(
        kernel = 'linear',
        degree = 5
    )
    clf.fit(train_inputs, train_targets.ravel())
    print "SVC accuracy: " + str(clf.score(
        validation_inputs,
        validation_targets
    ))

    #########################################################
    #   LogisticRegression.                                 #
    #########################################################
    clf = LogisticRegression(
        solver = 'newton-cg',
        max_iter = 100
    )
    clf.fit(train_inputs, train_targets.ravel())
    print "LogisticRegression accuracy: " + str(clf.score(
        validation_inputs,
        validation_targets
    ))
