
from scipy.io import arff
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
# Machine Learning Classifiers.
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
#from sklearn.dummy import DummyClassifier
#from sklearn.metrics import f1_score
from ModelScorer import ModelScorer


def read_arff(name):
    data = arff.loadarff(name)
    df = pd.DataFrame(data[0])

    # Convert target strings to bits.
    df['C'] = df['C'].map(lambda x: 1 if str(x)[2:-1]=='True' else 0)
    return df

# Read and shuffle dataset.
df = read_arff('dataset.arff')
df = shuffle(df)

# Divide dataset
model_scorer = ModelScorer(
    df=df,
    training_size=0.7,
    Y_col = 'C'
)

# Decision Tree
model_scorer.score(clf=DecisionTreeClassifier())
