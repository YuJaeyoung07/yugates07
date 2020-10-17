import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

titanic_df = pd.read_csv('./titanic_train.csv')
titanic_df.head(3)
test_df = pd.read_csv('./test.csv')

PassengerId = test_df['PassengerId']

def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['Cabin'].fillna('N', inplace = True)
    df['Embarked'].fillna('N', inplace = True)
    df['Fare'].fillna(0, inplace = True)
    return df

def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
    return df

def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df


def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis =1)

test_df = transform_features(test_df)

X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size = 0.1, random_state=11)

params = { 'n_estimators':[100,200,300,400,500], 'learning_rate': [0.005, 0.01,0.05, 0.1]}

gb_clf = GradientBoostingClassifier(random_state=0)

grid_cv = GridSearchCV(gb_clf,param_grid=params,cv=5,verbose=1)

grid_cv.fit(X_train, y_train)

print('최적 하이퍼 파라미터:\n',grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))

pred = grid_cv.best_estimator_.predict(X_test)

print('GBM 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))

params_dt = {'max_depth':[6,8,10,12,16,20,24], 'min_samples_split': [16,24]}
dt_clf = DecisionTreeClassifier(random_state=156)
grid_cv_dt = GridSearchCV(dt_clf, param_grid = params_dt, scoring= 'accuracy', cv=5, verbose=1)
grid_cv_dt.fit(X_train,y_train)

pred_dt = grid_cv_dt.best_estimator_.predict(X_test)

lr_clf = LogisticRegression()

vo_clf = VotingClassifier(estimators=[('LR', lr_clf), ('DT', grid_cv_dt.best_estimator_), ('GB', grid_cv.best_estimator_)])
vo_clf.fit(X_train, y_train)
pred_vo = vo_clf.predict(X_test)

print('Voting 정확도:{0:.4f}'.format(accuracy_score(y_test,pred_vo)))

final = grid_cv.best_estimator_.predict(test_df)
print(final)

final_vo = vo_clf.predict(test_df)
print(final_vo)

output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': final_vo})
output.to_csv('my_submission.csv', index=False)
