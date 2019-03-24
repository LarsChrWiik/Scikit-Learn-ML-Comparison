
from scipy.io import arff
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from ModelScorer import ModelScorer
import pandas as pd
from Plotter import *
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.expand_frame_repr', False)

# Machine Learning Classifiers.

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier


Y_col = 'C'


def read_arff(name):
    data = arff.loadarff(name)
    df = pd.DataFrame(data[0])

    # Convert target strings to bits.
    df[Y_col] = df[Y_col].map(lambda x: 1 if str(x)[2:-1]=='True' else 0)
    return df


def score_models():
    df = read_arff('dataset.arff')

    # Normalize.
    df = df.apply(lambda x: (x - x.min()) /(x.max() - x.min()), axis=0)

    # Unsupervised Learning.
    X = df.drop(Y_col, axis=1)
    ocsvm = OneClassSVM()
    ocsvm.fit(X)
    df['Category'] = ocsvm.predict(X)

    # Model Scorer.
    scores = []
    model_scorer = ModelScorer(df=df, Y_col=Y_col)
    scores.append(model_scorer.score_model(clf=DummyClassifier()))
    scores.append(model_scorer.score_model(clf=DecisionTreeClassifier()))
    scores.append(model_scorer.score_model(clf=RandomForestClassifier(n_estimators=100)))
    scores.append(model_scorer.score_model(clf=GradientBoostingClassifier(n_estimators=100)))
    scores.append(model_scorer.score_model(clf=XGBClassifier(n_estimators=100)))
    scores.append(model_scorer.score_model(clf=SGDClassifier()))
    scores.append(model_scorer.score_model(clf=LogisticRegression()))
    scores.append(model_scorer.score_model(clf=GaussianNB()))
    scores.append(model_scorer.score_model(clf=KNeighborsClassifier()))
    scores.append(model_scorer.score_model(clf=BernoulliNB()))
    scores.append(model_scorer.score_model(clf=SVC(kernel='linear', degree=5)))
    scores.append(model_scorer.score_model(clf = MLPClassifier()))
    scores.append(model_scorer.score_model(
        clf = MLPClassifier(
            activation = 'tanh',
            solver = 'lbfgs',
            hidden_layer_sizes = 100,
            learning_rate_init = 0.001,
            max_iter = 100000
        ),
        name='Tuned MLPClassifier')
    )

    df_result = pd.concat(scores).reset_index(drop=True)
    df_result = df_result.sort_values(["accuracy"], ascending=False)
    print(df_result)


def show_feature_importances():
    df = read_arff('dataset.arff')

    # Normalize.
    df = df.apply(lambda x: (x - x.min()) /(x.max() - x.min()), axis=0)

    X = df.drop(Y_col, axis=1)
    Y = df[Y_col]
    plot_feature_importance(X, Y)


if __name__ == "__main__":
    #score_models()
    show_feature_importances()
