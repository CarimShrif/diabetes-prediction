import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

sns.set_theme()

# reading
data = pd.read_csv("diabetes.csv")

# cleaning dataset (zeros or missing values will be replaced by the mean of that particular column )
# only these columns are selected
columns = ['Glucose', 'Insulin', 'BMI', 'Age']
for i in columns:
    data[i] = data[i].replace(0, np.NaN)
    cols_mean = int(data[i].mean(skipna=True))
    data[i] = data[i].replace(np.NaN, cols_mean)
data = data[columns + ['Outcome']]

# data visualisation
sns.pairplot(data=data, hue="Outcome")
plt.show()

# splitting the data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(' Training data =', len(X_train), '\n', 'Testing data =', len(X_test), '\n', 'Total data length = ',
      len(X_train) + len(X_test))


def knn(x_train, Y_train, x_test, Y_test, n, metric):
    Knn = KNeighborsClassifier(n_neighbors=n, metric=metric)
    Knn.fit(x_train, Y_train)
    predict_y = Knn.predict(x_test)
    accuracy = metrics.accuracy_score(Y_test, predict_y)
    matrix = confusion_matrix(Y_test, predict_y)
    sns.heatmap(matrix, annot=True, cmap="Blues", cbar=True)
    plt.show()
    print("Accuracy: " + str(accuracy))


def Svm(x_train, Y_train, x_test, Y_test, kernel, C, gamma):
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(x_train, Y_train)
    predict_y = clf.predict(x_test)
    accuracy = metrics.accuracy_score(Y_test, predict_y)
    matrix = confusion_matrix(Y_test, predict_y)
    sns.heatmap(matrix, annot=True, cmap="Blues", cbar=True)
    plt.show()
    print("Accuracy: " + str(accuracy))


# find the best params to pass to the classifier
def best_params(classifier, param_grid):
    grid = GridSearchCV(classifier, param_grid, cv=4)
    grid.fit(X, y)
    print("best score: " + str(grid.best_score_))
    return grid.best_params_


print("using the knn model")
param_grid_knn = {
    'n_neighbors': np.arange(1, 51),
    'metric': ['minkowski', 'manhattan', 'euclidean']
}
params = best_params(KNeighborsClassifier(), param_grid_knn)
knn(X_train, y_train, X_test, y_test, params['n_neighbors'], params['metric'])

print("using the svm model")
param_grid_svm = {
    'C': [0.001, 0.01, 0.1, 1],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf'],
}
params = best_params(svm.SVC(), param_grid_svm)
Svm(X_train, y_train, X_test, y_test, params['kernel'], params['C'], params['gamma'])
