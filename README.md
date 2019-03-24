# Comparing-Machine-Learning-Models
A comparison between various ML models from scikit-learn:
<br>

classifier | accuracy | precision |recall | f1 | f1_weighted
--- | --- | --- | --- | --- | ---
Tuned MLPClassifier | 0.899 | 0.912726 | 0.877908 | 0.894653 | 0.898958
GradientBoostingClassifier | 0.832 | 0.808642 | 0.831558 | 0.819650 | 0.832235
RandomForestClassifier | 0.821 | 0.812567 | 0.811516 | 0.811214 | 0.821279
XGBClassifier | 0.818 | 0.808126 | 0.808430 | 0.807469 | 0.818304
MLPClassifier | 0.757 | 0.808344 | 0.713088 | 0.757438 | 0.757071
DecisionTreeClassifier | 0.756 | 0.738140 | 0.741797 | 0.739125 | 0.756386
LogisticRegression | 0.713 | 0.751262 | 0.675667 | 0.710398 | 0.713162
KNeighborsClassifier | 0.686 | 0.644972 | 0.673797 | 0.658759 | 0.686488
SVC | 0.683 | 0.869677 | 0.615200 | 0.720035 | 0.691801
SGDClassifier | 0.639 | 0.369239 | 0.719592 | 0.438582 | 0.689179
GaussianNB | 0.594 | 0.588844 | 0.566550 | 0.576690 | 0.593897
BernoulliNB | 0.517 | 0.111552 | 0.582474 | 0.132505 | 0.629275
DummyClassifier | 0.516 | 0.480779 | 0.483824 | 0.481079 | 0.517320

<br>
Feature importance graph:
<br>

![logo](./images/feature_importance_sklearn.png?raw=true)
