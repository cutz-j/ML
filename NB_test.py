from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

all_data = pd.read_csv("d:/data/dt/object_area_recognition_data.csv")

X_train = all_data.iloc[:,:-1]
y_train = all_data.iloc[:,[-1]]



clf = MultinomialNB().fit(X_train, y_train)

clf.predict(X_train)

print(classification_report(y_train, clf.predict(X_train)))

train, test = train_test_split(all_data, test_size=0.3)

X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]
X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]

clf_v2 = MultinomialNB(alpha=.7).fit(X_train, y_train)
ytrain_nb_predicted = clf_v2.predict(X_train)
ytest_nb_predicted = clf_v2.predict(X_test)

print ("\nNaive Bayes - Train Confusion Matrix\n\n",pd.crosstab(y_train, ytrain_nb_predicted, rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nNaive Bayes- Train accuracy",round(accuracy_score(y_train, ytrain_nb_predicted),3))
print ("\nNaive Bayes  - Train Classification Report\n",classification_report(y_train, ytrain_nb_predicted))

print ("\nNaive Bayes - Test Confusion Matrix\n\n",pd.crosstab(y_test, ytest_nb_predicted,rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nNaive Bayes- Test accuracy",round(accuracy_score(y_test, ytest_nb_predicted),3))
print ("\nNaive Bayes  - Test Classification Report\n",classification_report(y_test, ytest_nb_predicted))


#from sklearn.neighbors import KNeighborsClassifier
#
#knn_fit = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
#knn_fit.fit(X_train, y_train)
#
#ytrain_predicted = knn_fit.predict(X_train)
#ytest_predicted = knn_fit.predict(X_test)
#
#print ("\nNaive Bayes - Train Confusion Matrix\n\n",pd.crosstab(y_train, ytrain_predicted, rownames = ["Actuall"],colnames = ["Predicted"]))      
#print ("\nNaive Bayes- Train accuracy",round(accuracy_score(y_train, ytrain_predicted),3))
#print ("\nNaive Bayes  - Train Classification Report\n",classification_report(y_train, ytrain_predicted))
#
#print ("\nNaive Bayes - Test Confusion Matrix\n\n",pd.crosstab(y_test, ytest_predicted,rownames = ["Actuall"],colnames = ["Predicted"]))      
#print ("\nNaive Bayes- Test accuracy",round(accuracy_score(y_test, ytest_predicted),3))
#print ("\nNaive Bayes  - Test Classification Report\n",classification_report(y_test, ytest_predicted))


dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

ytrain_predicted = dt.predict(X_train)
ytest_predicted = dt.predict(X_test)

print ("\nNaive Bayes - Train Confusion Matrix\n\n",pd.crosstab(y_train, ytrain_predicted, rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nNaive Bayes- Train accuracy",round(accuracy_score(y_train, ytrain_predicted),3))
print ("\nNaive Bayes  - Train Classification Report\n",classification_report(y_train, ytrain_predicted))

print ("\nNaive Bayes - Test Confusion Matrix\n\n",pd.crosstab(y_test, ytest_predicted,rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nNaive Bayes- Test accuracy",round(accuracy_score(y_test, ytest_predicted),3))
print ("\nNaive Bayes  - Test Classification Report\n",classification_report(y_test, ytest_predicted))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

ytrain_predicted = rf.predict(X_train)
ytest_predicted = rf.predict(X_test)

print ("\nNaive Bayes - Train Confusion Matrix\n\n",pd.crosstab(y_train, ytrain_predicted, rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nNaive Bayes- Train accuracy",round(accuracy_score(y_train, ytrain_predicted),3))
print ("\nNaive Bayes  - Train Classification Report\n",classification_report(y_train, ytrain_predicted))

print ("\nNaive Bayes - Test Confusion Matrix\n\n",pd.crosstab(y_test, ytest_predicted,rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nNaive Bayes- Test accuracy",round(accuracy_score(y_test, ytest_predicted),3))
print ("\nNaive Bayes  - Test Classification Report\n",classification_report(y_test, ytest_predicted))










