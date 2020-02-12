import os
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import keras
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from sklearn.model_selection import train_test_split
import talos
from datetime import datetime


p = {'input_neuron':[32, 64, 128, 256],
     'first_neuron':[32, 64, 128, 256],
     'hidden_layers':[1, 2, 3],
     'shapes': [0.0],
     'dropout': [0.0, 0.2],
     'activation':['relu'],
     'batch_size': [10, 64],
     'optimizer': ['adam', 'nadam']}


def def_model(X, y, X_val, y_val, params):
    dimX = len(X[0])
    dimy = y.shape[1]

    model = Sequential()
    model.add(Dense(params['input_neuron'], activation='relu', input_shape=(dimX,)))
    model.add(Dropout(params['dropout']))
    
    talos.utils.hidden_layers(model, params, 1)
    
    model.add(Dense(dimy, activation='softmax'))
    NAME = "IDS-ML-InputNeurons={}-HiddenLayers={}-HiddenNeurons={}-Dropout={}=Optimizer{}".format(params['input_neuron'], params['hidden_layers'], params['first_neuron'], params['dropout'], params['optimizer'])
    tensorboard = TensorBoard('logsTalos1/{}'.format(NAME))
   
    #Configure Model
    model.compile(loss='categorical_crossentropy', 
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    
    out = model.fit(X, y,
                    validation_data=[X_val, y_val],
                    batch_size=params['batch_size'],
                    epochs=10,
                    verbose=2,
                    callbacks=[tensorboard])
    return out, model

def start(XPickle, yPickle):
    np.random.seed(42)
    pickle_in = open(XPickle, "rb")
    X = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(yPickle, "rb")
    y = pickle.load(pickle_in)
    pickle_in.close() 

    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


    print("Starting training at: ", datetime.utcnow())
    talos.Scan(x=X, y=y, x_val=X_val, y_val=y_val, params=p, model=def_model, experiment_name='talosIDS')
    print("Train finished at: ", datetime.utcnow())




if __name__ == "__main__":
    if len(sys.argv) == 3:
        start(sys.argv[1], sys.argv[2])
    else:
        print("This program will be used to find out the best model architecture")
        print('Usage: python kerasTuner.py X.pickle, y.pickle')