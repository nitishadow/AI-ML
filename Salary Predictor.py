import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


dataset = pd.read_csv('Salary Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
scaler = StandardScaler()
X = scaler.fit_transform(X)


def ScikitLearnMethod(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
    model = LinearRegression()
    model.fit(X_train, y_train)


    Predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, Predictions)
    print(mse)


    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, Predictions, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")
    plt.legend()
    plt.show()


    experience = float(input("Total Experience: "))
    team_lead_experience = float(input("Team Lead Experience: "))
    project_manager_experience = float(input("Project Manager Experience: "))
    certifications = float(input("Certifications: "))
    new_data = np.array([[experience, team_lead_experience, project_manager_experience, certifications]])
    predicted_salary = model.predict(new_data)
    print(f"Predicted Salary for given input: {predicted_salary[0]:.2f} INR")


def GradientDescentMethod(X, y):
    w = np.zeros(shape=(X.shape[1],))
    b = 0
    k = 0.021
    iterations = 1000


    def model(X, w, b):
        return np.dot(X, w) + b


    def costfn(X, y, w, b):
        m = len(y)
        predictions = model(X, w, b)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost


    def gradientfn(X, y, w, b):
        m = len(y)
        predictions = model(X, w, b)
        dj_dw = (1 / m) * np.dot(X.T, (predictions - y))
        dj_db = (1 / m) * np.sum(predictions - y)
        return dj_dw, dj_db


    def gradient_descent(X, y, w, b, k, iterations):
        for i in range(iterations):
            dj_dw, dj_db = gradientfn(X, y, w, b)
            w = w - k * dj_dw
            b = b - k * dj_db
        return w, b


    wf, bf = gradient_descent(X, y, w, b, k, iterations)
    y_pred = model(X, wf, bf)
    print(costfn(X, y, wf, bf))


    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")
    plt.legend()
    plt.show()


    experience = float(input("Total Experience: "))
    team_lead_experience = float(input("Team Lead Experience: "))
    project_manager_experience = float(input("Project Manager Experience: "))
    certifications = float(input("Certifications: "))
    new_data = np.array([[experience, team_lead_experience, project_manager_experience, certifications]])
    predicted_salary = model(new_data, wf, bf)
    print(f"Predicted Salary for given input: {predicted_salary[0]:.2f} INR")


ScikitLearnMethod(X, y)


GradientDescentMethod(X, y)