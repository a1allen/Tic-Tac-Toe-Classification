import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import warnings

warnings.filterwarnings("ignore")

#Assign new headers to the DataFrame in the form of a list
headers = ["top-left-square","top-middle-square","top-right-square","middle-left-square",
                "middle-middle-square","middle-right-square","bottom-left-square",
                "bottom-middle-square","bottom-right-square","Class"]

# Import the data, there is no header, and data is separated by commas
data = pd.read_table(r'tic-tac-toe.data', header=None ,delimiter=",", names=headers)

#Preprocess the data by turning each letter into a numerical value for classification
le = LabelEncoder()
data = data.apply(le.fit_transform)

#Split the data into features and target variables
X = data.drop('Class', axis=1)
y = data['Class']

#Split the data for training and testing
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

#Prepare the 10 fold cross-validation for testing of the the classifier effectiveness
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


#Decision Trees with attribute importance, then check the data with Kfold
dt_clf = DecisionTreeClassifier()
dt_clf.fit(xTrain, yTrain)
dt_predY = dt_clf.predict(xTest)
dt_accuracy_score = accuracy_score(yTest, dt_predY)
#Find the importance of the attributes in decision trees
dt_feature_importances = dt_clf.feature_importances_
dt_feature_importance_data = pd.DataFrame({'Feature': X.columns, 'Importance': dt_feature_importances}).sort_values(by='Importance', ascending=False)
#10 fold cross-validation
dt_cv_scores = cross_val_score(dt_clf, X, y, cv=cv, scoring='accuracy')

#Random Forest with attribute importance, then check the data with Kfold
rf_clf = RandomForestClassifier(n_estimators=400, random_state=42)
rf_clf.fit(xTrain, yTrain)
rf_predY = rf_clf.predict(xTest)
rf_accuracy_score = accuracy_score(yTest, rf_predY)
#Find the importance of the attributes in random forest algorithm
rf_feature_importances = rf_clf.feature_importances_
rf_feature_importance_data = pd.DataFrame({'Feature': X.columns, 'Importance': rf_feature_importances}).sort_values(by='Importance', ascending=False)
#10 fold cross-validation
rf_cv_scores = cross_val_score(rf_clf, X, y, cv=cv, scoring='accuracy')

#Logistic Regression, then check the data with Kfold
lr_clf = LogisticRegression()
lr_clf.fit(xTrain, yTrain)
lr_predY = lr_clf.predict(xTest)
lr_accuracy_score = accuracy_score(yTest, lr_predY)
#10 fold cross-validation
lr_cv_scores = cross_val_score(lr_clf, X, y, cv=cv, scoring='accuracy')

#K-Nearest Neighbors, then check the data with Kfold
knn_clf = KNeighborsClassifier()
knn_clf.fit(xTrain, yTrain)
knn_predY = knn_clf.predict(xTest)
knn_accuracy_score = accuracy_score(yTest, knn_predY)
#10 fold cross-validation
knn_cv_scores = cross_val_score(knn_clf, X, y, cv=cv, scoring='accuracy')

#Neural Networks, then check the data with Kfold
nn_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)
nn_clf.fit(xTrain, yTrain)
nn_predY = nn_clf.predict(xTest)
nn_accuracy_score = accuracy_score(yTest, nn_predY)
#10 fold cross-validation
nn_cv_scores = cross_val_score(nn_clf, X, y, cv=cv, scoring='accuracy')

#Support Vector Machines with cross-validation, then check the data with Kfold
svc_clf = SVC(kernel='rbf', C=20.0)
svc_clf.fit(xTrain, yTrain)
svc_predY = svc_clf.predict(xTest)
svc_accuracy_score = accuracy_score(yTest, svc_predY)
#10 fold cross-validation
svc_cv_scores = cross_val_score(svc_clf, X, y, cv=cv, scoring='accuracy')


#Print all of the accuracies from the classifiers
print("\n========================================")
print("Decision Tree Accuracy:", dt_accuracy_score)
print("Random Forest Accuracy:", rf_accuracy_score)
print("Logistic Regression Accuracy:", lr_accuracy_score)
print("K-Nearest Neighbors Accuracy:", knn_accuracy_score)
print("Neural Network Accuracy:", nn_accuracy_score)
print("Support Vector Classification Accuracy:", svc_accuracy_score)
print("========================================\n")

print("========================================")
print("Decision Tree Cross-Validation Scores:", dt_cv_scores)
print("Mean Accuracy:", np.mean(dt_cv_scores),"\n")
print("Random Forest Cross-Validation Scores:", rf_cv_scores)
print("Mean Accuracy:", np.mean(rf_cv_scores),"\n")
print("Logistic Regression Cross-Validation Scores:", lr_cv_scores)
print("Mean Accuracy:", np.mean(lr_cv_scores),"\n")
print("K-Nearest Neighbors Cross-Validation Scores:", knn_cv_scores)
print("Mean Accuracy:", np.mean(knn_cv_scores),"\n")
print("Neural Network Cross-Validation Scores:", nn_cv_scores)
print("Mean Accuracy:", np.mean(nn_cv_scores),"\n")
print("Support Vector Classification Cross-Validation Scores:", svc_cv_scores)
print("Mean Accuracy:", np.mean(svc_cv_scores),"\n")
print("========================================\n")

print("========================================")
print("Decsision Trees Attribute Importance\n")
print(dt_feature_importance_data)
print("========================================\n")

print("========================================")
print("Random Forest Attribute Importance\n")
print(rf_feature_importance_data)
print("========================================")


