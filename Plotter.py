
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_importance(X, Y):
    clf = DecisionTreeClassifier()
    clf.fit(X, Y)

    features = X.columns.values
    importances = clf.feature_importances_

    df_plot = pd.DataFrame()
    df_plot['Importances'] = importances
    df_plot['Feature'] = features
    df_plot = df_plot.sort_values(["Importances"], ascending=False)

    sns.set(style="whitegrid")
    ax = sns.barplot(x='Importances', y='Feature', data=df_plot) # , ci="sd"
    #ax.set_xlabel('totalCount')
    plt.show()
