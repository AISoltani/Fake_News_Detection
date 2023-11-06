# important notice...
# This project contain some library that need you install befor...(just on linux, windows not supported)

from matplotlib.pyplot import *
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
import itertools
import os, re
import cv2
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scikitplot.plotters as skplt
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords

# First step Preprocessing

def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)


def cleanup(text):
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def constructLabeledSentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

def getEmbeddings(path,vector_dimension=300):
    data = pd.read_csv(path)

    missing_rows = []
    for i in range(len(data)):
        if data.loc[i, 'text'] != data.loc[i, 'text']:
            missing_rows.append(i)
    data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

    for i in range(len(data)):
        data.loc[i, 'text'] = cleanup(data.loc[i,'text'])

    x = constructLabeledSentences(data['text'])
    y = data['label'].values

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,
                         seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.iter)

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y[i]

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        test_labels[j] = y[i]
        j = j + 1

    return text_train_arrays, text_test_arrays, train_labels, test_labels



def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



##########################################################################################################################

#KNN



def Knn_Classifier(x_train,x_test,y_train,y_test):

#Find besk K for KNN classifier

    # accuracies = []
    # kVals = range(1, 30, 2)
    # # loop over various values of `k` for the k-Nearest Neighbor classifier
    # for k in range(1, 30, 2):
    #     # train the k-Nearest Neighbor classifier with the current value of `k`
    #
    #     knn = KNeighborsClassifier(metric='minkowski', p=2, n_neighbors=k, weights='distance')
    #     knn.fit(x_train, y_train)
    #     y_te_pred = knn.predict(x_test)
    #     acc = accuracy_score(y_test, y_te_pred)
    #
    #     # evaluate the model and update the accuracies list
    #
    #     accuracies.append(acc)
    #     print("k=%d, accuracy=%.2f%%" % (k, acc * 100))
    #
    # bestk = int(np.argmax(accuracies))
    # print("k=%d achieved highest accuracy of %.2f%% on test data" % (kVals[bestk], accuracies[bestk] * 100))

    # Create KNN model object to classification

    knn = KNeighborsClassifier(metric='minkowski', p=2, n_neighbors=2, weights='distance')



    knn.fit(x_train, y_train)
    y_te_pred = knn.predict(x_test)

    print(metrics.classification_report(y_test, y_te_pred))


    acc = accuracy_score(y_test, y_te_pred)
    print("Knn Accuracy Is: %0.2f %%" % (float(acc) * 100))


    plt.figure()
    cfs5 = confusion_matrix(y_test, y_te_pred)
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(cfs5, classes=class_names, title='Knn Confusion matrix, without normalization')

    plt.show()
    # plt.figure()
    # cfs = confusion_matrix(usps_labels_test.argmax(axis=1), y_te_pred.argmax(axis=1))
    # le = preprocessing.LabelEncoder()
    # enc = le.fit(usps_labels_test.argmax(axis=1))
    # class_names = enc.classes_
    # plot_confusion_matrix(cfs, classes=class_names, title='KNN Confusion matrix, without normalization')
    # print("Total calssification report:\n")
    # print(classification_report(usps_labels_test, y_te_pred))
    # plt.figure()
    # cfs2 = confusion_matrix(usps_labels_test.argmax(axis=1), y_te_pred.argmax(axis=1))
    # sns.heatmap(cfs2, square=True, cmap='inferno')
    # title('Confusion Matrix-Color Map:\nKNN Classifier')
    # ylabel('True')
    # xlabel('Predicted label')
    # plt.show()


##########################################################################################################################




# Baysian with gussian distrbution

def Bayesian_Classifier(x_train, x_test, y_train, y_test):

    clf = GaussianNB()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    # acc = accuracy_score(y_test, predicted)
    # print("Result Accuracy : ",(float(acc)*100))
  #  accuracy = (matches.sum() / float(len(matches))) * 100
    print(metrics.classification_report(y_test, predicted))


    plt.figure()
    baysian = confusion_matrix(y_test, predicted)
    sns.heatmap(baysian, square=True, cmap='inferno')
    title('Confusion Matrix:\nBayesian (Gussian) ')
    ylabel('True')
    xlabel('Predicted label')
    total = baysian.sum(axis=None)
    correct = baysian.diagonal().sum()
    print("Bayesian With Gussian Distrbution Accuracy Is: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    cfs5 = confusion_matrix(y_test, predicted)
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(cfs5, classes=class_names, title='Bayesian( Gussian ) Confusion matrix, without normalization')

    plt.show()


##########################################################################################################################

# Random Forest Classifier

def Random_Forest_Classifier(x_train, x_test, y_train, y_test):
    rfc = RandomForestClassifier(max_depth=15, n_estimators=20, max_features=5)
    rfc.fit(x_train, y_train)

    predicted = rfc.predict(x_test)
    acc = accuracy_score(y_test, predicted)
    print("Result Accuracy : ",(float(acc)*100))
    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    rfc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(rfc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nRFC')
    ylabel('True')
    xlabel('Predicted label')
    total = rfc_cm.sum(axis=None)
    correct = rfc_cm.diagonal().sum()
    print("RFC Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(rfc_cm, classes=class_names, title='RFC Confusion matrix, without normalization')
    plt.show()

##########################################################################################################################

# Multilayer Prceptron Classifier

def Multilayer_Prceptron_Classifier(x_train, x_test, y_train, y_test):
    mlp = MLPClassifier(alpha=1)
    mlp.fit(x_train, y_train)

    predicted = mlp.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    mlp_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(mlp_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nMLP')
    ylabel('True')
    xlabel('Predicted label')
    total = mlp_cm.sum(axis=None)
    correct = mlp_cm.diagonal().sum()
    print("MLP Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(mlp_cm, classes=class_names, title='MLP Confusion Matrix, without normalization')
    plt.show()

##########################################################################################################################

# Ada Boost Classifier

def Ada_Boost_Classifier(x_train, x_test, y_train, y_test):
    abc = AdaBoostClassifier()
    abc.fit(x_train, y_train)

    predicted = abc.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    abc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(abc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nAda Boost Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = abc_cm.sum(axis=None)
    correct = abc_cm.diagonal().sum()
    print("Ada Boost Classifier Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(abc_cm, classes=class_names, title='Ada Boost Classifier Confusion Matrix, without normalization')
    plt.show()

##########################################################################################################################

# Decision Tree Classifier Classifier

def Decision_Tree_Classifier(x_train, x_test, y_train, y_test):
    dtc = DTC(max_depth=50)
    dtc.fit(x_train, y_train)

    predicted = dtc.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    dtc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(dtc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nDecision Tree Classifier Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = dtc_cm.sum(axis=None)
    correct = dtc_cm.diagonal().sum()
    print("Decision Tree Classifier Classifier Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(dtc_cm, classes=class_names, title='Decision Tree_Classifier Confusion Matrix, without normalization')
    plt.show()

##########################################################################################################################

# Gaussian Process Classifier

def Gaussian_Process_Classifier(x_train, x_test, y_train, y_test):
    gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
    gpc.fit(x_train, y_train)

    predicted = gpc.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    gpc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(gpc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nGaussian Process Classifier Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = gpc_cm.sum(axis=None)
    correct = gpc_cm.diagonal().sum()
    print("Gaussian Process Classifier Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(gpc_cm, classes=class_names, title='Gaussian Process Classifier Confusion Matrix, without normalization')
    plt.show()

##########################################################################################################################

# Support Vector Machine classifier

def Support_Vector_Machine(x_train, x_test, y_train, y_test):
    svm = SVC(gamma=0.001)
    svm.fit(x_train, y_train)

    predicted = svm.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    svm_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(svm_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nSupport Vector Machine Classifier Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = svm_cm.sum(axis=None)
    correct = svm_cm.diagonal().sum()
    print("Support Vector Machine Classifier Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(svm_cm, classes=class_names, title='Support Vector MachineClassifier Confusion Matrix, without normalization')
    plt.show()



##########################################################################################################################

# Stochastic Gradient Descent

def Stochastic_Gradient_Descent(x_train, x_test, y_train, y_test):

    gpc = SGDClassifier()
    gpc.fit(x_train, y_train)

    predicted = gpc.predict(x_test)

    print(metrics.classification_report(y_test, predicted))
    plt.figure()
    gpc_cm = confusion_matrix(y_test, predicted)
    sns.heatmap(gpc_cm, square=True, cmap='inferno')
    title('Confusion Matrix-Color Map:\nStochastic Gradient Descent Classifier Classifier')
    ylabel('True')
    xlabel('Predicted label')
    total = gpc_cm.sum(axis=None)
    correct = gpc_cm.diagonal().sum()
    print("Stochastic Gradient Descent Classifier  Accuracy: %0.2f %%" % (100.0 * correct / total))

    plt.figure()
    le = preprocessing.LabelEncoder()
    enc = le.fit(y_test)
    class_names = enc.classes_
    plot_confusion_matrix(gpc_cm, classes=class_names, title='Stochastic Gradient Descent Classifier\n Confusion Matrix, without normalization')
    plt.show()

##########################################################################################################################

def Neural_Network(x_train, x_test, y_train, y_test):
    '''Neural network with 3 hidden layers'''
    model = Sequential()
    model.add(Dense(256, input_dim=300, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='relu', kernel_initializer='normal'))
    model.add(Dense(2, activation="softmax", kernel_initializer='normal'))

    # gradient descent
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # configure the learning process of the model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


    model = Neural_Network()
    model.summary()
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    encoded_y = np_utils.to_categorical((label_encoder.transform(y_train)))
    label_encoder.fit(y_test)
    encoded_y_test = np_utils.to_categorical((label_encoder.transform(y_test)))
    estimator = model.fit(x_train, encoded_y, epochs=20, batch_size=64)
    print("Model Trained!")
    score = model.evaluate(x_test, encoded_y_test)
    print("")
    print("Accuracy = " + format(score[1] * 100, '.2f') + "%")  # 92.69%

    probabs = model.predict_proba(x_test)
    y_pred = np.argmax(probabs, axis=1)

    plot_cmat(y_test, y_pred)
##########################################################################################################################



# Main fnction

def PRProject_Main(n):


    if n == 1:
        Knn_Classifier(x_train, x_test, y_train, y_test)

    elif n==2:
        Bayesian_Classifier(x_train, x_test, y_train, y_test)

    elif n==3:
        Random_Forest_Classifier(x_train, x_test, y_train, y_test)

    elif n==4:
        Multilayer_Prceptron_Classifier(x_train, x_test, y_train, y_test)

    elif n==5:
        Ada_Boost_Classifier(x_train, x_test, y_train, y_test)

    elif n==6:
        Decision_Tree_Classifier(x_train, x_test, y_train, y_test)

    elif n==7:
        Gaussian_Process_Classifier(x_train, x_test, y_train, y_test)

    elif n==8:
        Support_Vector_Machine(x_train, x_test, y_train, y_test)

    elif n==9:
        Stochastic_Gradient_Descent(x_train, x_test, y_train, y_test)

    elif n==10:
        Neural_Network(x_train, x_test, y_train, y_test)
if __name__ == "__main__":


    # Reading train data for preprocessing...

    # x_train, x_test, y_train, y_test = getEmbeddings("datasets/train.csv")
    # np.save('./xtr', x_train)
    # np.save('./xte', x_test)
    # np.save('./ytr', y_train)
    # np.save('./yte', y_test)

    x_train = np.load('./xtr.npy')
    x_test = np.load('./xte.npy')
    y_train = np.load('./ytr.npy')
    y_test = np.load('./yte.npy')
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train,
                                                        stratify=y_train,
                                                        test_size=0.2)
    print("Handwritten Digit Recognition Project-Pattern Recognition\n\nTo Choose your Method  Run, First You Need To Install "
          "Required Libraries\n\n")
    print("1.Knn Classifier\n"
          "2.Bayesian Classifier\n"
          "3.Random Forest Classifier\n"
          "4.Multilayer Prceptron Classifier\n"
          "5.Ada Boost Classifier\n"
          "6.Decision Tree Classifier\n"
          "7.Gaussian Process Classifier\n"
          "8.Support Vector Machine\n"
          "9.Stochastic Gradient Descent\n"
          "10.Neural Network (TensorFlow)\n"
          "Choosing The Number Of Method : ")
    n = int(input())
    PRProject_Main(n)
