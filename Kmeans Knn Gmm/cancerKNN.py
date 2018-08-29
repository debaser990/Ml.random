import pandas
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import pylab as pl

file = r"F:\AKJ2\Machine Learning\datasets\breast_cancer.csv"

col = ['ID','Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape','Marginal_Adhesion','Single_Epithelial_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_N ucleoli','Mitosis','Class']

data = pandas.read_csv(file,names=col,header=None)

data = data.drop(data[data.Bare_Nuclei == '?'].index)

print(data.shape)

#print(data.info())

data['Class'].replace(2,0,inplace = True)

data['Class'].replace(4,1,inplace = True)

#g = data['Class'].value_counts()

y = data['Class']

x = data.drop(['ID','Class'],axis=1)

X = StandardScaler().fit_transform(x.values)

xtrain, xtest, ytrain, ytest = train_test_split(X,y,random_state = 7,test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(xtrain,ytrain)

predict = knn.predict(xtest)

predict2 = knn.predict(xtrain)

#print(accuracy_score(predict,ytest))
#print(accuracy_score(knn.predict(xtrain),ytrain))
#print(classification_report(predict,ytest))
#print(confusion_matrix(predict,ytest))

#gmm = GaussianMixture()
#gmm.fit(xtrain,ytrain)
#predict2 = knn.predict(xtest)
#print(accuracy_score(predict2,ytest))
#print(classification_report(predict2,ytest))
#print(confusion_matrix(predict2,ytest))

kmeans = KMeans(n_clusters=2)
kmeans.fit(xtrain,ytrain)
predict3 = kmeans.predict(xtest)

print(accuracy_score(predict3,ytest))
print(classification_report(predict3,ytest))
print(confusion_matrix(predict3,ytest))





