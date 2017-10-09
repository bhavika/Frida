from PIL import Image
import os
from os import listdir
from os.path import isfile, isdir, join, exists, getsize
import numpy as np

def load_image(file_path):
    with Image.open(file_path) as image:
        im_arr = np.fromstring(image.tobytes(), dtype = np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))

    return im_arr

def getDataset(testDataSuffixList = [1, 2], predictDataSuffix = [0]):
    trainingImages = []
    trainingResult = []
    testImages = []
    testResult = []
    trainSize = 0
    testSize = 0
    predictSize = 0
            
    annotationFilePath = 'C:\\Users\\Melanie\\Desktop\\art_anno\\annotation.txt'
    print("Annotation file = " + annotationFilePath)

    annotationFile = open(annotationFilePath)
    fileContents = annotationFile.readlines();

    #searchTerm = fileContents[0].rstrip()
    artistClass = int(fileContents[1].rstrip())
    #imageCount = int(fileContents[2].rstrip())

    #print("result = " + str(artistClass)) #1-baltatu, 2-sisley, 3-blanchard

    imgDir = 'C:\\Users\\Melanie\\Desktop\\impress_data_small\\'
    imageNumber = 0
    imageList = [f for f in os.listdir(imgDir) if isfile(join(imgDir, f))]  
    while imageNumber < 120:
        imageFilename = imgDir + imageList[imageNumber]
        print(imageFilename)
        if exists(imageFilename) and isfile(imageFilename):
            imageSize = getsize(imageFilename)
            if imageSize > 0:
                imageArray = load_image(imageFilename)
                if imageNumber % 10 in testDataSuffixList:
                    testImages.append(imageArray)
                    testResult.append(artistClass)
                    testSize += 1
                elif imageNumber % 10 not in predictDataSuffix:
                    trainingImages.append(imageArray)
                    trainingResult.append(artistClass)
                    trainSize += 1
                else:
                    predictSize += 1

        imageNumber += 1


    trainingSet = (np.asarray(trainingImages), np.asarray(trainingResult))
    testSet = (np.asarray(testImages), np.asarray(testResult))

    print("Loaded all images. Training size = " + str(trainSize) + ", Test size = " + str(testSize) + ", Predict size = " + str(predictSize))

    return trainingSet, testSet

def getDatasetForPrediction(predictDataSuffix = [0]):
    predictImages = []
    predictResult = []
    predictFilepaths = []


    annotationFilePath = 'C:\\Users\\Melanie\\Desktop\\art_anno\\annotation.txt'
    print("Annotation file = " + annotationFilePath)

    annotationFile = open(annotationFilePath)
    fileContents = annotationFile.readlines();

    searchTerm = fileContents[0].rstrip()
    result = int(fileContents[1].rstrip())
    imageCount = int(fileContents[2].rstrip())

    print("search term = " + str(searchTerm) + ", result = " + str(result) + ", count = " + str(imageCount))

    subjectImageFolder = 'C:\\Users\\Melanie\\Desktop\\impress_data_small\\'
    imageNumber = 1

    while imageNumber <= imageCount:
        imageFilename = subjectImageFolder + str(imageNumber).zfill(4) + '.jpg'
        if exists(imageFilename) and isfile(imageFilename):
            imageSize = getsize(imageFilename)
            if imageSize > 0:
                imageArray = load_image(imageFilename)
                if imageNumber % 10 in predictDataSuffix:
                    predictImages.append(imageArray)
                    predictResult.append(result)
                    predictFilepaths.append(imageFilename)

        imageNumber += 1


    predictSize = len(predictImages)
    predictSet = (np.asarray(predictImages), np.asarray(predictResult), predictFilepaths)

    print("Loaded all prediction images. Size = " + str(predictSize))

    return predictSet