import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def main():
    # logistic regression
    train = pd.DataFrame(pd.read_csv('train.csv'))
    validation = pd.DataFrame(pd.read_csv('validation.csv'))
    test = pd.DataFrame(pd.read_csv('test.csv'))

    X = train[train.columns[0:23]]
    y = train['Y']
    print('#logistic regression')
    logistic = LogisticRegression()
    logistic.fit(X, y)
    print(logistic.score(X,y))
    preds = logistic.predict(X=train[train.columns[0:23]])
    print(pd.crosstab(preds, train['Y']))

    print('#random forest')
    rf = RandomForestClassifier( n_estimators=150)
    rf.fit(X, y)
    predicted = rf.predict(X=test[test.columns[0:23]])
    print(pd.crosstab(predicted, test['Y']))

    print('#svm')
    clf = svm.SVC()
    clf.fit(X, y)
    predic = clf.predict(test[test.columns[0:23]])
    print(pd.crosstab(predic, test['Y']))


if __name__ == '__main__':
    main()