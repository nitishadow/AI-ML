import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('Fake Bills Data.csv', sep = ';')
dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset['is_genuine'].map({True:1, False:0}).values
scaler = StandardScaler()
X = scaler.fit_transform(X)


def ScikitLearnMethod(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    degree = 1
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy}.")

    diagonal = float(input("Diagonal: "))
    height_left = float(input("Height Left: "))
    height_right = float(input("Height Right: "))
    margin_low = float(input("Margin Low: "))
    margin_up = float(input("Margin Up: "))
    length = float(input("Length: "))

    new_data = np.array([[diagonal, height_left, height_right, margin_low, margin_up, length]])
    new_data = scaler.transform(new_data)
    new_data_poly = poly.transform(new_data)
    y_newpred = model.predict(new_data_poly)

    if y_newpred == 1:
        print("Genuine")
    else:
        print("Not Genuine")


def GradientDescentMethod(X, y):

    w = np.zeros(shape=X.shape[1])
    b = 0
    k = 0.01
    iterations = 1000

    def model(X, w, b):
        return np.dot(X, w) + b

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def costfn(X, y, w, b):
        m = len(y)
        z = model(X, w, b)
        y_pred = sigmoid(z)
        cost = (-1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost

    def gradientfn(X, y, w, b):
        m = len(y)
        z = model(X, w, b)
        y_pred = sigmoid(z)
        dj_dw = 1 / m * np.dot(X.T, (y_pred - y))
        dj_db = 1 / m * np.sum(y_pred - y)
        return dj_dw, dj_db

    def gradient_descent(X, y, w, b, k, iterations):
        for i in range(iterations):
            dj_dw, dj_db = gradientfn(X, y, w, b)
            w = w - k * dj_dw
            b = b - k * dj_db
        return w, b

    wf, bf = gradient_descent(X, y, w, b, k, iterations)
    z = model(X, wf, bf)
    y_pred = sigmoid(z)
    print(f"Final Cost: {costfn(X, y, wf, bf)}.")

    diagonal = float(input("Diagonal: "))
    height_left = float(input("Height Left: "))
    height_right = float(input("Height Right: "))
    margin_low = float(input("Margin Low: "))
    margin_up = float(input("Margin Up: "))
    length = float(input("Length: "))
    new_data = np.array([[diagonal, height_left, height_right, margin_low, margin_up, length]])
    new_data = scaler.transform(new_data)
    z = model(new_data, wf, bf)
    y_newpred = sigmoid(z)
    if y_newpred > 0.5:
        print("Genuine")
    else:
        print("Not Genuine")


ScikitLearnMethod(X, y)


GradientDescentMethod(X, y)