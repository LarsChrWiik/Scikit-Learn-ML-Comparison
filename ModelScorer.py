
class ModelScorer():

    def __init__(self, df, training_size, Y_col):
        # Construct Training set.
        self.df_train = df.sample(int(training_size * len(df)), replace=False)
        self.df_train_X, self.df_train_Y = self.get_X_Y(self.df_train, Y_col=Y_col)
        # Contruct Test set.
        self.df_test = df.drop(self.df_train.index, axis=0)
        self.df_test_X, self.df_test_Y = self.get_X_Y(self.df_test, Y_col=Y_col)

    # Function that reperate input feature fector from target
    def get_X_Y(self, df, Y_col):
        return df.drop(columns=[Y_col]), df[Y_col]

    # Train and evaluate the input classifier.
    def score(self, clf):
        clf.fit(self.df_train_X, self.df_train_Y)
        accuracy = clf.score(self.df_test_X, self.df_test_Y)
        #f1 = clf.f1_score(self.df_test_X, self.df_test_Y)
        print("DecisionTreeClassifier")
        print("   * Accuracy: %.2f" % (accuracy*100))
