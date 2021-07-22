import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

iris= datasets.load_iris()


features=iris.data
labels=iris.target



labels_names = ['I.setosa', 'I.versicolor', 'I.virginica']
colors=['blue', 'red', 'green']

for i in range(len(colors)):
    px=features[:,0][labels==i]
    py=features[:,1][labels==i]
    plt.scatter(px, py, c=colors[i])
plt.legend(labels_names)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


for i in range(len(colors)):
    px=features[:,1][labels==i]
    py=features[:,2][labels==i]
    plt.scatter(px, py, c=colors[i])
plt.legend(labels_names)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

est=PCA(n_components=2)
x_pca=est.fit_transform(features)


x_train, x_test, y_train, y_test = train_test_split(
    x_pca, labels, test_size=0.4, random_state=33)

print("SVM accuracy  :")
clf=svm.SVC(kernel='linear')
clf.fit(x_train,y_train) 
pred=clf.predict(x_test)
acc=accuracy_score(pred,y_test)

print(metrics.classification_report(
    y_test, pred, target_names=labels_names))
print("SVM Accuracy = " ,acc)



print("KNN accuracy  :")
clf2=KNeighborsClassifier(n_neighbors=1)
clf2.fit(x_train, y_train)
pred = clf2.predict(x_test)
acc2=accuracy_score(pred,y_test)

print(metrics.classification_report(
    y_test, pred, target_names=labels_names))
print("KNN Accuracy = " ,acc2)







