#cancerGMM


import pandas
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import pylab as pl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


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

x1 = np.linspace(np.min(X),np.max(X),len(X))
gmm = GaussianMixture()
gmm.fit(X)

y = gmm.score_samples(X)

plt.plot(x1, np.exp(y))
plt.show()







