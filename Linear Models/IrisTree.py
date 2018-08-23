from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import graphviz

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

model = tree.DecisionTreeClassifier()

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2, random_state=7)


fittedmodel = model.fit(xtrain,ytrain)

predict1 = fittedmodel.predict(xtest)
score1 = accuracy_score(predict1,ytest)
print("score:",score1)
print(predict1)
print(ytest)

graphfile = tree.export_graphviz(fittedmodel,out_file = None,filled = True)

graphviz.Source(graphfile)

graphviz.render(engine = 'dot', format = 'pdf',filepath = r'C:\Users\Akshat\Desktop')
