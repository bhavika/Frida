# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 23:49:34 2017

@author: Melanie
"""

import DEMO_LoadDataset_PredictOnly
from keras.models import load_model
import constants

# returns a compiled model
# identical to the previous one

model = load_model(constants.model_path)
print("Model has been reloaded.")


(x_predict, y_predict, z_predict) = DEMO_LoadDataset_PredictOnly.getDatasetForPrediction([0])

print("Dataset is loaded.")

modelPredictions = model.predict(x_predict)

print("Got predictions.")

baltatuCorrect = 0
sisleyCorrect = 0
blanchardCorrect = 0
kuindzhiCorrect = 0
guillauminCorrect = 0
rodinCorrect = 0
morisotCorrect = 0
pissarroCorrect = 0
hassamCorrect = 0
monetCorrect = 0
artachinoCorrect = 0
vreedenburghCorrect = 0
degasCorrect = 0
manetCorrect = 0
boudinCorrect = 0

totalCorrect = 0


imgNum = 0
totalImages = y_predict.shape[0]

while imgNum < totalImages:
    actual = y_predict[imgNum]
    print(actual)
    print(modelPredictions[imgNum][0], modelPredictions[imgNum][1], modelPredictions[imgNum][2],
      modelPredictions[imgNum][3], modelPredictions[imgNum][4], modelPredictions[imgNum][5],
      modelPredictions[imgNum][6], modelPredictions[imgNum][7], modelPredictions[imgNum][8],
      modelPredictions[imgNum][9], modelPredictions[imgNum][10], modelPredictions[imgNum][11],
      modelPredictions[imgNum][12], modelPredictions[imgNum][13], modelPredictions[imgNum][14])
    imgNum += 1
    
print('Done.')