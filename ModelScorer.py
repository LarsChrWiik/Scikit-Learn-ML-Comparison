
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.model_selection import KFold
import pandas as pd


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


class ModelScorer():

    def __init__(self, df, Y_col):
        self.Y_col = Y_col
        self.df = df.copy()

    def score_model(self, clf, cv=5, name=None):
        X, Y = self.df.drop(self.Y_col, axis=1), self.df[self.Y_col]
        result_list = []

        for train_index, test_index in KFold(n_splits=cv).split(self.df):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            # train
            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)

            # Score model.
            df_result = pd.DataFrame()
            df_result['classifier'] = [type(clf).__name__ if name == None else name]
            df_result['accuracy'] = [accuracy_score(Y_pred, Y_test)]
            df_result['precision'] = [precision_score(Y_pred, Y_test)]
            df_result['recall'] = [recall_score(Y_pred, Y_test)]
            df_result['f1'] = [f1_score(Y_pred, Y_test)]
            df_result['f1_weighted'] = f1_score(Y_pred, Y_test, average='weighted')

            result_list.append(df_result)

        return pd.concat(result_list).groupby(['classifier'], as_index=False).mean()
