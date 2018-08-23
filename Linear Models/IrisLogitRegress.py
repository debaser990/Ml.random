
import pandas
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
#import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
list(iris.keys())

#print(iris.data.shape)

x = iris["data"]
y = (iris["target"]==2).astype(np.int)
print(x)
#print(y)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.5, random_state=7)

log_reg = LogisticRegression()
fitted = log_reg.fit(xtrain,ytrain)
#print(fitted.intercept_) #Beta 0 
#print(fitted.coef_)        #Beta 1
prediction = fitted.predict(xtest)

print(accuracy_score(prediction, ytest))

x_new = np.linspace(0,3,1000).reshape(-1,4)
y_proba = fitted.predict_proba(x_new)

plt.plot(x_new, y_proba[:,1], "g-" , label = "Iris-Virginica")
plt.plot(x_new, y_proba[:,0], "b--", label = "Not Iris-Virginica ")

plt.show()
