from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
rom sklearn.neural_network import MLPClassifier
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

#svm
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train, y_train)
s = svm.SVC(kernel='linear').fit(X_train, y_train)
svc_2 = svm.SVC(kernel='linear').fit(X_train_pca,y_train)
pred_svm = s.predict(X_test) #1.0
pred_svm2 = svc_2.predict(X_test_pca)

#knn
knn = KNeighborsClassifier(n_neighbors=5, p=2,metric='minkowski')
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

#random forest
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

#decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
pred_dt = dt.predict(X_test)

#mlp
mlp = MLPClassifier(solver='lbfgs', random_state=0)
mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test) #93%

print(metrics.classification_report(y_test, pred_svm))