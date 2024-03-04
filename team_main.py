import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import mod2

data = mod2.df.drop(['customerID'], axis=1)  # 'customerID'는 필요없으니 버림
data = mod2.clean_data(data)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])  # data type : object -> float

def eval_logistic():
    # parameters
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'penalty': ['l1', 'l2']}

    # grid search
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

    # 최종 모델 성능 점검
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Churn', axis=1), data['Churn'].values.ravel(), test_size=0.2, shuffle=True, random_state=1)

    grid_search.fit(X_train, y_train)
    grid_search.score(X_test, y_test)

    print("GridSearchCV with Logistic Regression")
    print("Best Parameter : ", grid_search.best_params_)
    print("Best Score: {:.2f}".format(grid_search.score(X_test, y_test)))


# setting encoder, scaler, model
encoding = ['LabelEncoder', 'get_dummies']
scaling = ['MinMaxScaler', ' StandardScaler', 'RobustScaler', 'MaxAbsScaler']
model = ['LogisticRegression', 'XGBClassifier', 'DecisionTreeClassifier']

# predict best combination ( Open Source )
best = mod2.find_best(data, encoding, scaling, model)

print("\nBest Score : ", best['score'])
print("Best Encoder : ", best['encoder'])
print("Best Scaler : ", best['scaler'])
print("Best Model : ", best['model'])

# evaluation
eval_logistic()



