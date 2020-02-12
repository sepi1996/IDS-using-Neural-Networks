import sys
import os
import pandas as pd
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical, normalize
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn import preprocessing
from random import randrange

def createCSVCleaned(inFile, outFile):
    bening = beningDeleted = ftp = ssh = totalRowsDeleted = 0
    with open(inFile, 'r') as csvfile:
        data = csvfile.readlines()
        totalRows = len(data) - 1
        print('total rows read = {}'.format(totalRows))
        with open(outFile, 'w') as csvoutfile:
            firstRow = data[0]
            csvoutfile.write('{}\n'.format(firstRow))
            for line in data[1:]:
                line = line.strip()
                if "Infinity" not in line and "NaN" not in line:
                    if "FTP-BruteForce" in line:
                        ftp = ftp + 1
                        csvoutfile.write('{}\n'.format(line))
                    elif "SSH-Bruteforce" in line:
                        ssh = ssh + 1
                        csvoutfile.write('{}\n'.format(line))
                    else: 
                        if randrange(3) == 1:
                            bening = bening + 1
                            csvoutfile.write('{}\n'.format(line))
                        else:
                            beningDeleted = beningDeleted + 1
                else:
                    totalRowsDeleted = totalRowsDeleted + 1
    
    print('total rows procesed = {}'.format((totalRows - totalRowsDeleted)))
    print('total rows deleted = {}'.format(totalRowsDeleted))
    print('total bening rows deleted = {}'.format(beningDeleted))
    print('total ssh rows = {}'.format(ssh))
    print('total ftp rows  = {}'.format(ftp))
    print('total bening rows = {}'.format(bening - 1))
        
def createPickles(outFile):
    dataFrame = pd.read_csv(outFile)
    del dataFrame['Timestamp']
    del dataFrame['Dst Port']
    del dataFrame['Protocol']
    dataFrame = shuffle(dataFrame) # mezclar los elementos

    en = LabelEncoder()
    yData = dataFrame.pop('Label')
    en.fit(yData)
    yDataEncoded = en.transform(yData)
    y = to_categorical(yDataEncoded)

    X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(dataFrame.values)).values

    #Create pickle objects
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def importData(inFile):
    outFile = inFile.replace(".csv", "cleaned.csv")
    createCSVCleaned(inFile, outFile)
    createPickles(outFile)
    

if __name__ == "__main__":
    if len(sys.argv) == 2:
        importData(sys.argv[1])
    else:
        print("This program will generate a cleaned csv and two pickle objects (X.pickle and y.pickle) from the input dataset")
        print('Usage: python import.py inputFile.csv')

        