import matplotlib.pyplot as plt 
import numpy as np

from sklearn.tree import DecisionTreeRegressor

#Creating random dataset 
rng = np.random.RandomState(1)
X = np.sort(5*rng.rand(80,1),axis=0)
Y = np.sin(X).ravel()
Y[::5] += 3 * (0.5 - rng.rand(16))


#Implementing Regression Model
regr1 = DecisionTreeRegressor(max_depth=2)
regr2 = DecisionTreeRegressor(max_depth=5)
regr1.fit(X,Y)
regr2.fit(X,Y)

#Predict
X_test = np.arange(0.0,5.5,0.01)[:,np.newaxis]
y_1 = regr1.predict(X_test)
y_2 = regr2.predict(X_test)

#Plot the results
plt.figure()
plt.scatter(X, Y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()