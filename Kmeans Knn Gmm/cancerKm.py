#cancerKM

import pandas
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import  StandardScaler
import pylab as pl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


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

print(type(x))
x = StandardScaler().fit_transform(x.values)

x2 = pandas.DataFrame(x)
x2.columns = ['Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape','Marginal_Adhesion','Single_Epithelial_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_N ucleoli','Mitosis']

#print(x2)

#xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state = 7,test_size=0.2)

kmeans = KMeans(n_clusters=2)
k = kmeans.fit(x)
#predict = kmeans.predict(x)


centroids = k.cluster_centers_
labels = k.labels_

print(len(labels))

x2['Cluster'] = labels

print("Method 1: Plottin Against 2 features")
print('''
sns.lmplot('Clump_Thickness', 'Uniformity_of_Cell_Size', 
           data=x2, 
           fit_reg=False, 
           hue="Cluster",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Clump_Thickness vs Uniformity_of_Cell_Size')
plt.xlabel('Clump_Thickness')
plt.ylabel('Uniformity_of_Cell_Size')
plt.show()
print("\n\n\n")
''')



corr = x2.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(x2.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(x2.columns)
ax.set_yticklabels(x2.columns)
plt.show()



pca = PCA(n_components = 3).fit(x) 
pcadat = pca.fit_transform(x)

print(sum(pca.explained_variance_ratio_))

print("Method 2 plotting against 3 principal components :")
x1,y1,z1 = zip(*pcadat)

x1 = np.array(x1)
y1 = np.array(y1)
z1 = np.array(z1) 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

   # Plot the values
ax.scatter(x1, y1, z1, c = k.labels_, marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()
