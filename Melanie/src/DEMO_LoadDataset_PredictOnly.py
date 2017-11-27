# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:42:05 2017

@author: Melanie
"""
from PIL import Image
import os
from os import listdir
from os.path import isfile, isdir, join, exists, getsize
import numpy as np
import Prep_dataset as prep
import pandas as pd
import constants

imgDir = constants.demo_path

def load_image(file_path):
    with Image.open(file_path) as image:
        im_arr = np.fromstring(image.tobytes(), dtype = np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))

    return im_arr

def getDatasetForPrediction(predictDataSuffix = [0]):
    predictImages = []
    predictResult = []
    predictFilepaths = []

    #artist_data = pd.read_csv(prep.csv)
    #artist_class = artist_data['class']
    
    imageNumber = 0
    imageList = [f for f in os.listdir(imgDir) if isfile(join(imgDir, f))]
    while imageNumber < 1:
        artistClass = -1
        imageFilename = imgDir + imageList[imageNumber]
        #print(imageFilename)
        if exists(imageFilename) and isfile(imageFilename):
            imageSize = getsize(imageFilename)
            if imageSize > 0:
                imageArray = load_image(imageFilename)
                
                predictImages.append(imageArray)
                predictResult.append(artistClass)
                predictFilepaths.append(imageFilename)

        imageNumber += 1


    predictSize = len(predictImages)
    predictSet = (np.asarray(predictImages), np.asarray(predictResult), predictFilepaths)
    print(predictResult)

    print("Loaded all prediction images. Size = " + str(predictSize))

    return predictSet