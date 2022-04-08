
'''
Author: Danial Baharvand
'''

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(10469290, 'Emily', 'Guan'), (10084983, 'Danial', 'Baharvand'), (10489690, 'Calise', 'Newton')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    ''' 
    # Get data from file  
    X = np.genfromtxt(dataset_path, delimiter=',', dtype = float, usecols = range(2,31))
    y = np.genfromtxt(dataset_path, delimiter=',', dtype = str, usecols = 1)
    
    # Set 'M' to 1 and 'B' to 0
    y[np.argwhere(y == 'M')] = 1
    y[np.argwhere(y == 'B')] = 0
    
    # Convert to array 
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''

    # Fitting Decision Tree classifier to the training set
    clf = DecisionTreeClassifier(random_state=15)

    # Find best hyper param value using cross-validated gridsearch
    clf = GridSearchCV(clf, {'max_depth': np.linspace(1, 100, 100)})

    # Train the model
    clf.fit(X_training,y_training)

    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Initialize classifier
    clf = KNeighborsClassifier()
    # Find the best number of neighbors between 1 and 100 using 10 cross validation
    clf = GridSearchCV(clf, {'n_neighbors': range(1, 100)}, cv=5)
    # Fitting to the training data
    clf = clf.fit(X_training, y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    ####Four types of kernels (ignoring sigmoid):
    clf = svm.SVC(kernel = 'linear',random_state=15)
    #clf = svm.SVC(kernel = 'polynomial', degree = 3)
    #clf = svm.SVC(kernel = 'rbf')
    
    #parameters = {'kernel':('linear', 'rbf'), 'C':range(1, 100)}  
    clf = GridSearchCV(clf, {'C': [0.01, 0.1, 1, 10, 100, 1000]}, cv=10)
    #clf = GridSearchCV(clf, parameters)
    
    
    clf.fit(X_training, y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network classifier (with two dense hidden layers)  
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''

    # Fitting Neural Network classifier to the training set
    clf = MLPClassifier(max_iter=1000,random_state=15)  # 500 iterations to train on data
    # Find best hyper param value using cross-validated gridsearch
    clf = GridSearchCV(clf, {'hidden_layer_sizes': [(i,) for i in range(10,20)]})
    # Train the model with the training data
    clf.fit(X_training, y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ## AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
    ##         "INSERT YOUR CODE HERE"    
def eval_dt():
        # Call Decision Tree classifier
    dt = build_DecisionTree_classifier(X_train,y_train)
    # Generate classification report for test data
    print("Decision Tree Test Data Classification Report:")
    # Predict the test set results with data
    y_predict = dt.predict(X_test)
    print(classification_report(y_test, y_predict))
    print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
    print("Best params: " + str(dt.best_params_))


def eval_KNN():
    # Call KNN classifier
    KNN = build_NearrestNeighbours_classifier(X_train, y_train)
    # Generate classification report for test data
    print("Nearrest Neighbours classifier Test Data Classification Report:")
    # Predict the test set results with data
    y_predict = KNN.predict(X_test)
    print(classification_report(y_test, y_predict))
    print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
    print("Best params: " + str(KNN.best_params_))


def eval_SVM():
    # Call SVM classifier
    svmc = build_SupportVectorMachine_classifier(X_train,y_train)
    # Generate classification report for test data
    print("SVM Test Data Classification Report:")
    # Predict the test set results with data
    y_predict = svmc.predict(X_test)
    print(classification_report(y_test, y_predict))
    print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
    print("Best params: " + str(svmc.best_params_))


def eval_NN():
    # Call Neural Network classifier
    nn = build_NeuralNetwork_classifier(X_train,y_train)
    # Generate classification report for test data
    print("Neural Network Test Data Classification Report:")
    # Predict the test set results with data
    y_predict = nn.predict(X_test)
    print(classification_report(y_test, y_predict))
    print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
    print("Best params: " + str(nn.best_params_))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    # Prepare data
    x,y= prepare_dataset(".\medical_records.data")
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state = 15) # 33% test, 66% training

    # Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Calling functions to build classifiers, evaluate them on the test set and produce report
    eval_dt()
    eval_KNN()
    eval_SVM()
    eval_NN()
