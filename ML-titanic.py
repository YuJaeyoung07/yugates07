# Kaggle Titanic 상위 10%를 기록한 MachineLearning Code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# titanic_train.csv, test.csv 파일을 읽어들인다.
titanic_df = pd.read_csv('./titanic_train.csv')
titanic_df.head(3)
test_df = pd.read_csv('./test.csv')

PassengerId = test_df['PassengerId']

# Null(빈값) 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['Cabin'].fillna('N', inplace = True)
    df['Embarked'].fillna('N', inplace = True)
    df['Fare'].fillna(0, inplace = True)
    return df

# 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
    return df

# 레이블 인코딩 수행
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞선 전처리 함수들을 하나로 모으는 함수
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

# y_titanic_df는 'Survived' 열 추출, x_titanic_df는 'Survived' 열 제외한 열 추출
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis =1)

# 테스트 데이터를 전처리함
test_df = transform_features(test_df)

# 피처 데이터 세트를 전처리함
X_titanic_df = transform_features(X_titanic_df)

# 학습 데이터와 테스트 데이터를 추출, 테스트 데이터 세트 크기는 전체의 20%
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size = 0.11, random_state=11)

# GBM Parameter 설정
params = { 'n_estimators':[100,200,300,400,500,1000], 'learning_rate': [0.001,0.005, 0.01,0.05, 0.1]}

# GBM 객체 생성 후 GridSearchCV 수행
gb_clf = GradientBoostingClassifier(random_state=0)
grid_cv = GridSearchCV(gb_clf,param_grid=params,cv=5,verbose=1)
grid_cv.fit(X_train, y_train)

# 최적으로 학습된 Estimator로 예측 수행(GBM)
pred = grid_cv.best_estimator_.predict(X_test)
print('GBM 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))

# DecisionTree Parameter 설정
params_dt = {'max_depth':[6,8,10,12,16,20,24], 'min_samples_split': [8,16,24]}

# DecisionTree 객체 생성 후 GridSearchCV 수행
dt_clf = DecisionTreeClassifier(random_state=156)
grid_cv_dt = GridSearchCV(dt_clf, param_grid = params_dt, scoring= 'accuracy', cv=5, verbose=1)
grid_cv_dt.fit(X_train,y_train)

# 최적으로 학습된 Estimator로 예측 수행(DecisionTree)
pred_dt = grid_cv_dt.best_estimator_.predict(X_test)

# LogisticRegression 객체 생성
lr_clf = LogisticRegression()

# GBM, DecisionTree, LogisticRegression Estimator를 Voting(Hard Voting), 그 후 예측 수행
vo_clf = VotingClassifier(estimators=[('LR', lr_clf), ('DT', grid_cv_dt.best_estimator_),('GB', grid_cv.best_estimator_)])
vo_clf.fit(X_train, y_train)
pred_vo = vo_clf.predict(X_test)

# 정확도 Print(Accuracy)
print('Voting 정확도:{0:.4f}'.format(accuracy_score(y_test,pred_vo)))

final = grid_cv.best_estimator_.predict(test_df)
print(final)

# Voting에서 가장 예측력 높은 Estimator를 통해 테스트 데이터 예측
final_vo = vo_clf.predict(test_df)
print(final_vo)

output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': final_vo})
output.to_csv('my_submission.csv', index=False)
