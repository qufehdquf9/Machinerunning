from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import decomposition
from sklearn import svm
from sklearn.model_selection import train_test_split

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X_train, X_test, y_train, y_test = train_test_split(lfw_people.data,lfw_people.target, random_state=2)

pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1)
#clf=rf.fit(X_train, y_train)
#y_pred1 = clf.predict(X_test)

#knn = KNeighborsClassifier(n_neighbors=1, p=2,metric='minkowski')
#knn_1 = knn.fit(X_train, y_train)
#y_pred1 = knn_1.predict(X_test)

#clf = svm.SVC(C=5., gamma=0.001)  # 분류 모델임.

svc_1 = svm.SVC(kernel='linear').fit(X_train, y_train)
svc_2 = svm.SVC(kernel='linear').fit(X_train_pca,y_train)
y_pred1 = svc_1.predict(X_test)
y_pred2 = svc_2.predict(X_test_pca)
print(metrics.classification_report(y_test, y_pred1))
print(metrics.classification_report(y_test, y_pred2))