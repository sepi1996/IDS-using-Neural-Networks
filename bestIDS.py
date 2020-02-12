import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import pandas as pd

def draw_confusion_matrix(model, X, y):
    XPrediction = model.predict(X, batch_size=32, verbose=1)
    Xpredicted = np.argmax(XPrediction, axis=1)

    confusionMatrix = confusion_matrix(np.argmax(y, axis=1), Xpredicted)

    matrix = pd.DataFrame(confusionMatrix, range(3), range(3))
    plt.figure(figsize = (3,3))
    sn.set(font_scale=1)
    ax = sn.heatmap(matrix, annot=True, annot_kws={"size":10})
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    print("\n")
    FP = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix)  
    FN = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
    TP = np.diag(confusionMatrix)
    TN = confusionMatrix.sum() - (FP + FN + TP)

    #true positive rate
    TPR = TP/(TP+FN)
    #true negative rate
    TNR = TN/(TN+FP) 
    #false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    print('######Benign ',' FTP Attack ', 'SSH Attack' )
    print('TPR', TPR)
    print('TNR', TNR)
    print('FPR', FPR)
    print('FNR', FNR)

def evaluate_test(model, X_test, y_test):
    results = model.evaluate(X_test, y_test, verbose=1)
    print()
    print('Test Accuracy: %1.2f%%' % (results[1]*100))

def def_model(X, y, X_val, y_val):
    dimX = len(X[0])
    dimy = y.shape[1]
    
    #Design neural network architecture
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(dimX,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(dimy, activation='softmax'))

    #Configure Model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #Configure TensorBoard
    tensorbrd = TensorBoard('logs/IDS')
    #Train and Test
    model.fit(X, y, batch_size=128, epochs=10, verbose=2, validation_data=(X_val, y_val), callbacks=[tensorbrd])

    return model

def start(XPickle, yPickle):
    #Load Data
    np.random.seed(42)
    pickle_in = open(XPickle, "rb")
    X = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(yPickle, "rb")
    y = pickle.load(pickle_in)
    pickle_in.close() 

    #Split data train(81%), validation(10%), test(9%)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = def_model(X, y, X_val, y_val)

    draw_confusion_matrix(model, X, y)

    evaluate_test(model, X_test, y_test)
    
if __name__ == "__main__":
    if len(sys.argv) == 3:
        start(sys.argv[1], sys.argv[2])
    else:
        print("This program reproduce the behaivour of a NIDS based on the two Pickle objects produced before")
        print('Usage: python bestIDS.py X.pickle, y.pickle')
