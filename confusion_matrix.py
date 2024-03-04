import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
encoder = LabelEncoder()
scaler = MinMaxScaler()
model = LogisticRegression()

url = r"withNaN_WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url, skiprows=0)


def preprocessing_encoding(data):
    label = data.columns    # col들의 라벨 이름
    data.replace(" ", np.nan, inplace=True)  # 모든 공백값을 nan값으로 변경
    data.dropna(inplace=True)   # nan값이 들어있는 row를 버림
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])  # data type : object -> float

    for col in data:   # 'col'에는 각 column의 라벨 이름이 들어감
        if isinstance(df[col][0], str):  # 자료형이 str이라면 인코딩
            data[col] = encoder.fit_transform(data[col])

    # MinMax Scaling
    scaled_data = scaler.fit_transform(data)
    result = pd.DataFrame(data=scaled_data, columns=label)
    return result


data = df.drop(['customerID'], axis=1)  # 'customerID'는 필요없으니 버림
data2 = preprocessing_encoding(data)

encoding_target = data['Churn'].values.ravel()   # 타겟 col = 'Churn'
preprocessed_encoding_data = data2.drop('Churn', axis=1)  # 'Churn'은 타겟이니 버림

x_train, x_test, y_train, y_test = train_test_split(preprocessed_encoding_data, encoding_target, test_size=0.1, shuffle=True, random_state=1)
model.fit(x_train, y_train)
predicted_target = model.predict(x_test)

index = x_test.index
actual_y = []
for i in index:
    actual_y.append(df['Churn'][i])

compare_data = {'predict_y': predicted_target,
                'actual_y': actual_y}
temp = pd.DataFrame(compare_data, columns=['predict_y', 'actual_y'])
confusion_matrix = pd.crosstab(temp['actual_y'], temp['predict_y'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.title('Logistic Regression - Actual Churn VS Predict Churn')
plt.show()
