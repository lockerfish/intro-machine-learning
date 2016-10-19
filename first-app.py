# Load Libraries
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-lenght', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# checking the dataset
def shape():
    # shape - how many instances (rows) and how many attributes (colummns)
    print(dataset.shape)

def select(count):
    # head - look at the data
    print(dataset.head(count))

def summary():
    # descriptions - summary of each attribute (column)
    print(dataset.describe())

def groupby(attr):
    # class distribution
    print(dataset.groupby(attr).size())

# Summarize the dataset
# visualize data
def box_and_whisker():
    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

def histograms():
    # histograms
    dataset.hist()
    plt.show()

def scatter():
    # scatter plot matrix
    scatter_matrix(dataset)
    plt.show()

# def training():


def evaluate_algorithms():
    # split out validation dataset
    array = dataset.values
    # 80% of dataset used for training
    X = array[:,0:4]
    # 20% of dataset used for validation
    Y = array[:,4]

    print(X)
    print(Y)

    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation, = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    print(X_train)
    print(Y_train)
    print(100*'*')
    print(X_validation)
    print(Y_validation)
    
    # test options and evaluation metric
    num_folds = 10
    num_instances = len(X_train)
    scoring = 'accuracy'

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
            kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
            cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
    print(results)
    print(names)

    # compare algorithms
    # fig = plt.figure()
    # fig.suptitle('Algorithm Comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(results)
    # ax.set_xticklabels(names)
    # plt.show()

    # Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    # predictions = knn.predict(X_validation)
    predictions = knn.predict([1.5,1.4,1.8,1.1])
    print(predictions)
    # print(accuracy_score(Y_validation, predictions))
    # print(confusion_matrix(Y_validation, predictions))
    # print(classification_report(Y_validation, predictions))

def main():
    # shape()
    select(150)
    # summary()
    # groupby('class')

    # box_and_whisker()
    # histograms()
    # scatter()

    evaluate_algorithms()


if __name__ == "__main__":
    main()
