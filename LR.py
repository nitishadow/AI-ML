import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv('Salary_dataset.csv')
x = df['YearsExperience'].values.reshape(-1,1)
y = df['Salary'].values


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
# print(f"MSE: {mse}")


exp = float(input("Your Experience: "))
predicted_salary = model.predict([[exp]])
print(f"Predicted Salary for {exp} years of experience: {predicted_salary} INR")


m = len(x)
iterations = 1000
w = 0
b = 0
k = 0.001


def gradient(x, y, m, w, b):
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb_xi = w * x[i] + b

        dj_dw += f_wb_xi - y[i]
        dj_db += (f_wb_xi - y[i]) * x[i]
    dj_db /= m
    dj_dw /= m
    return dj_dw, dj_db


def gradient_descent(m, w, b, k, iterations):
    for i in range(iterations):
        dj_db, dj_dw = gradient(x, y, m, w, b)
        w = w - k * dj_dw
        b = b - k * dj_db
    return w, b


w, b = gradient_descent(m, w, b, k, iterations)


f_wb_x = w * x + b


plt.scatter(x, y, color='blue', label = 'Given data')
plt.plot(x, f_wb_x, color='red', label = 'Fitted line')
plt.legend()
plt.show()


def prediction(exp, w, b):
    exp = float(exp)
    return w * exp + b


exp = input("Your Experience: ")
predicted_salary = prediction(exp, w, b)
print(f"Predicted Salary for {exp} years of experience: {predicted_salary} INR")