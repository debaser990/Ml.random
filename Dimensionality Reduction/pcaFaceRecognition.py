from sklearn.datasets import fetch_lfw_people
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


faces  = fetch_lfw_people(min_faces_per_person = 100)

d,h,w = faces.images.shape

X = faces.data

y = faces.target

x_train,x_test,y_train, y_test =  train_test_split(X,y, test_size=0.3)

pca = PCA(n_components = 100, whiten = True).fit(x_train)
pcaxtrain = pca.fit_transform(x_train)
pcaxtest  = pca.transform(x_test)

print(sum(pca.explained_variance_ratio_))

clf  = MLPClassifier(hidden_layer_sizes = (1024,), batch_size = 256, verbose = True, early_stopping = True).fit(pcaxtrain,y_train)

predict = clf.predict(pcaxtest)
print("Accuracy score of prediction:",accuracy_score(predict,y_test))
print(classification_report(predict,y_test, target_names = faces.target_names))

def plot_gallery(images, titles, h, w,rows=3, cols = 4):
    plt.figure()
    for i in range(rows*cols):
        plt.subplot(rows,cols,i+1)
        plt.imshow(images[i].reshape((h,w)), cmap = plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())


def titles(predict,y_test, target_names):
    for i in range(predict.shape[0]):
        pred_names=target_names[predict[i]].split(' ')[-1]
        true_names= target_names[y_test[i]].split(' ')[-1]
        yield 'predicted:(0)/n true:{1}'.format(pred_names,true_names)

prediction_title = list(titles(predict,y_test,faces.target_names))
plot_gallery(x_test,prediction_title,h,w)
plt.show()

n_components =100 
eigenfaces = pca.components_.reshape(n_components, h, w)

ef_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]

plot_gallery(eigenfaces,ef_titles, h,w)
plt.show()





        
        
