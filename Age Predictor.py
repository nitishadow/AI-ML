import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

dataset = pd.read_csv('Age Predictor.csv', delimiter=',')
dataset.dropna(inplace=True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1}")

PAQ605 = float(input("If the respondent engages in moderate or vigorous-intensity sports, fitness, or recreational activities in: "))
BMXBMI = float(input("Respondent's Body Mass Index: "))
LBXGLU = float(input("Respondent's Blood Glucose after fasting: "))
LBXGLT = float(input("Respondent's Oral: "))
LBXIN = float(input("Respondent's Blood Insulin Levels: "))

new_data = np.array([[PAQ605, BMXBMI, LBXGLU, LBXGLT, LBXIN]])
new_data = scaler.transform(new_data)
y_new_pred = model.predict(new_data)

if y_new_pred == 0:
    print("Adult")
elif y_new_pred == 1:
    print("Senior")
