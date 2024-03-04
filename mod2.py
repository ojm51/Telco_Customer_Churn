import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

url = r"withNaN_WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url, skiprows=0)

def preprocessing_withEncoding(data, scaler, encoder):
    label = data.columns    # label names of column

    for col in data:   # 'col' -> name of each column
        # Encoding(1) - label encoding
        data[col] = encoder.fit_transform(data[col])

    # Scaling
    scaler.fit(data)
    result = pd.DataFrame(scaler.transform(data))
    result.columns = label

    return result

def preprocessing_withDummies(data, scaler):

    # Encoding(2) - get_dummies()
    data_dummies = pd.get_dummies(data)

    # Scaling
    features = data_dummies.columns.values
    scaler.fit(data_dummies)
    result = pd.DataFrame(scaler.transform(data_dummies))
    result.columns = features
    return result

''' === Open Source === '''
def find_best(data, ens, scs, model):
    predict_best = -1
    predict_encoder = ""
    predict_scaler = ""
    predict_model = ""

    for en in ens:
        for sc in scs:
            for mod in model:
                predict_now = get_prediction(data, en, sc, mod)
                if predict_now >= predict_best:
                    predict_encoder = en
                    predict_scaler = sc
                    predict_model = mod
                    predict_best = predict_now

    result_dict = {'score':predict_best, 'encoder':predict_encoder, 'scaler':predict_scaler, 'model':predict_model}
    return result_dict

def get_prediction(data, en, sc, mod):
    # scaler
    if sc == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif sc == 'StandardScaler':
        scaler = StandardScaler()
    elif sc == 'RobustScaler':
        scaler = RobustScaler()
    elif sc == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    else:
        return -1

    # model
    if mod == 'LogisticRegression':
        model = LogisticRegression()
    elif mod == 'XGBClassifier':
        model = XGBClassifier()
    elif mod == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier()
    else:
        return -1

    # encoder
    if en == 'LabelEncoder':
        encoder = LabelEncoder()
        dt = preprocessing_withEncoding(data, scaler, encoder)
        X = dt.drop('Churn', axis=1)
        y = dt['Churn'].values.ravel()

        predicted_target_best = -1

        # holdout method ( 0.2, 0.1 / shuffle=true, false )
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
        model.fit(x_train, y_train)
        predicted_target = model.predict(x_test)
        score1 = metrics.accuracy_score(y_test, predicted_target)
        predicted_target_best = score1

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=1)
        model.fit(x_train, y_train)
        predicted_target = model.predict(x_test)
        score2 = metrics.accuracy_score(y_test, predicted_target)
        if(score2 > predicted_target_best):
            predicted_target_best = score2

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)
        model.fit(x_train, y_train)
        predicted_target = model.predict(x_test)
        score3 = metrics.accuracy_score(y_test, predicted_target)
        if (score3 > predicted_target_best):
            predicted_target_best = score3

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)
        model.fit(x_train, y_train)
        predicted_target = model.predict(x_test)
        score4 = metrics.accuracy_score(y_test, predicted_target)
        if (score4 > predicted_target_best):
            predicted_target_best = score4


        return predicted_target_best

    elif en == 'get_dummies':
        dt = preprocessing_withDummies(data, scaler)
        X = dt.drop('Churn', axis=1)
        y = dt['Churn'].values.ravel()
        predicted_target_best = -1

        # holdout method ( 0.2, 0.1 / shuffle=true, false )
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
        model.fit(x_train, y_train)
        predicted_target = model.predict(x_test)
        score1 = metrics.accuracy_score(y_test, predicted_target)
        predicted_target_best = score1

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=1)
        model.fit(x_train, y_train)
        predicted_target = model.predict(x_test)
        score2 = metrics.accuracy_score(y_test, predicted_target)
        if(score2 > predicted_target_best):
            predicted_target_best = score2

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)
        model.fit(x_train, y_train)
        predicted_target = model.predict(x_test)
        score3 = metrics.accuracy_score(y_test, predicted_target)
        if (score3 > predicted_target_best):
            predicted_target_best = score3

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)
        model.fit(x_train, y_train)
        predicted_target = model.predict(x_test)
        score4 = metrics.accuracy_score(y_test, predicted_target)
        if (score4 > predicted_target_best):
            predicted_target_best = score4


        return predicted_target_best

    else:
        return -1

def clean_data(data):
    data.replace(" ", np.nan, inplace=True)  # 모든 공백값을 nan값으로 변경
    data.dropna(inplace=True)   # nan값이 들어있는 row를 버림

    return data
