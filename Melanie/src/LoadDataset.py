from PIL import Image
import os
from os import listdir
from os.path import isfile, isdir, join, exists, getsize
import numpy as np
import Prep_dataset as prep
import pandas as pd
import constants

imgDir = constants.small_img_path
#imgDir = 'C:\\Users\\Melanie\\Desktop\\impress_data_small\\'

def load_image(file_path):
    with Image.open(file_path) as image:
        im_arr = np.fromstring(image.tobytes(), dtype = np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))

    return im_arr

def getDataset(testDataSuffixList = [1, 2], predictDataSuffix = [0]):
    artist_data = pd.read_csv(prep.csv)
    artist_class = artist_data['class']
    
    trainingImages = []
    trainingResult = []
    testImages = []
    testResult = []
    trainSize = 0
    testSize = 0
    predictSize = 0
    i=0
    imageNumber = 0
    
    imageList = [f for f in os.listdir(imgDir) if isfile(join(imgDir, f))]  
    while imageNumber < constants.img_count:
        artistClass = artist_class[i]
        imageFilename = imgDir + imageList[imageNumber]
        #print(imageFilename)
        if exists(imageFilename) and isfile(imageFilename):
            imageSize = getsize(imageFilename)
            if imageSize > 0:
                imageArray = load_image(imageFilename)
                if imageNumber % 9 in testDataSuffixList:
                    testImages.append(imageArray)
                    testResult.append(artistClass)
                    testSize += 1
                elif imageNumber % 3 not in predictDataSuffix:
                    trainingImages.append(imageArray)
                    trainingResult.append(artistClass)
                    trainSize += 1
                else:
                    predictSize += 1

        imageNumber += 1
        i += 1
        
    trainingSet = (np.asarray(trainingImages), np.asarray(trainingResult))
    testSet = (np.asarray(testImages), np.asarray(testResult))
    print(trainingResult)
    print(testResult)

    print("Loaded all images. Training size = " + str(trainSize) + ", Test size = " + str(testSize) + ", Predict size = " + str(predictSize))

    return trainingSet, testSet

def getDatasetForPrediction(predictDataSuffix = [0]):
    predictImages = []
    predictResult = []
    predictFilepaths = []

    artist_data = pd.read_csv(prep.csv)
    artist_class = artist_data['class']
    i=0

    imageNumber = 0
    imageList = [f for f in os.listdir(imgDir) if isfile(join(imgDir, f))]
    while imageNumber < constants.img_count:
        artistClass = artist_class[i]
        imageFilename = imgDir + imageList[imageNumber]
        #print(imageFilename)
        if exists(imageFilename) and isfile(imageFilename):
            imageSize = getsize(imageFilename)
            if imageSize > 0:
                imageArray = load_image(imageFilename)
                if imageNumber % 3 in predictDataSuffix:
                    predictImages.append(imageArray)
                    predictResult.append(artistClass)
                    predictFilepaths.append(imageFilename)

        imageNumber += 1
        i += 1


    predictSize = len(predictImages)
    predictSet = (np.asarray(predictImages), np.asarray(predictResult), predictFilepaths)
    print(predictResult)

    print("Loaded all prediction images. Size = " + str(predictSize))

    return predictSet