# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:55:27 2017

@author: Melanie
"""

import DEMO_LoadDataset
from keras.models import load_model

# returns a compiled model
# identical to the previous one
model = load_model('C:\KerasModels\Model 2017-11-06 11-40-48\ArtistIdentificationModel.h5')
print("Model has been reloaded.")

(x_predict, y_predict, z_predict) = DEMO_LoadDataset.getDatasetForPrediction([0])
print("Dataset is loaded.")

modelPredictions = model.predict(x_predict)
print("Got predictions.")

baltatuCorrect = 0
baltatuIncorrect = 0
sisleyCorrect = 0
sisleyIncorrect = 0
blanchardCorrect = 0
blanchardIncorrect = 0

imgNum = 0
totalImages = y_predict.shape[0]

while imgNum < totalImages:
    actual = y_predict[imgNum]
    print(actual)
    prediction = 1
    print(modelPredictions[imgNum][0], modelPredictions[imgNum][1], modelPredictions[imgNum][2])

    imgNum += 1

print("Done.")