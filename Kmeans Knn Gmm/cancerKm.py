#cancerKM

import pandas
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import pylab as pl
from matplotlib import pyplot as plt


file = r"F:\AKJ2\Machine Learning\datasets\breast_cancer.csv"

col = ['ID','Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape','Marginal_Adhesion','Single_Epithelial_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_N ucleoli','Mitosis','Class']

data = pandas.read_csv(file,names=col,header=None)

data = data.drop(data[data.Bare_Nuclei == '?'].index)

print(data.shape)

data['Class'].replace(2,0,inplace = True)

data['Class'].replace(4,1,inplace = True)

#g = data['Class'].value_counts()

y = data['Class']

x = data.drop(['ID','Class'],axis=1)

#X = StandardScaler().fit_transform(x.values)

#xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state = 7,test_size=0.2)

kmeans = KMeans(n_clusters=2)
k =kmeans.fit(x)
#predict3 = kmeans.predict(x)

pca = PCA(n_components = 2).fit(x) 
pcadat = pca.fit_transform(x)

print(sum(pca.explained_variance_ratio_))

x1,y1 = zip(*pcadat)

x1 = np.array(x1)
y1 = np.array(y1)
print(x1)
print('y1',y1)
#print(kmeans.labels_)
plt.scatter(x1,y1,c=k.labels_)
plt.show()




