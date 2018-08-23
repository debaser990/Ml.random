#IrisNB

import pandas
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

dataset = datasets.load_iris()

#print(dataset)
keys = dataset.keys()
data = dataset.data
target = dataset.target
#print(keys)
#print(dataset.target_names) #0 = setosa 1=versicolor 2 = virginica

x = data
y = target
#print(x)
#print(y)

model = GaussianNB()
model2 = LogisticRegression()

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.5, random_state=7)


fittedmodel = model.fit(xtrain,ytrain)
fittedmodel2 = model2.fit(xtrain,ytrain)

predict1 = fittedmodel.predict(xtest)
predict2 = fittedmodel2.predict(xtest)

score1 = accuracy_score(predict1,ytest)
score2 = accuracy_score(predict2,ytest)

print("NB accuracy:",score1)

print("Logistic Regression accuracy :",score2)
