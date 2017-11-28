# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 23:49:34 2017

@author: Melanie
"""

import DEMO_LoadDataset_PredictOnly
from keras.models import load_model
import constants
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# returns a compiled model
# identical to the previous one

model = load_model(constants.model_path)
print("Model has been reloaded.")

(x_predict, y_predict, z_predict) = DEMO_LoadDataset_PredictOnly.getDatasetForPrediction([0])

print("Dataset is loaded.")

modelPredictions = model.predict(x_predict)

print("Got predictions.")

pissarro_Img1 = mpimg.imread(
    constants.small_img_path + 'camille-pissarro_apple-trees-and-poplars-in-the-setting-sun-1901.jpg')
pissarro_Img2 = mpimg.imread(constants.small_img_path + 'camille-pissarro_avenue-de-l-opera-effect-of-snow-1898.jpg')

hassam_Img1 = mpimg.imread(constants.small_img_path + 'childe-hassam_across-the-avenue-in-sunlight-june.jpg')
hassam_Img2 = mpimg.imread(constants.small_img_path + 'childe-hassam_an-evening-street-scene-pont-aven.jpg')

monet_Img1 = mpimg.imread(constants.small_img_path + 'claude-monet_andre-lauvray(1).jpg')
monet_Img2 = mpimg.imread(constants.small_img_path + 'claude-monet_apple-trees-in-bloom-at-giverny-1901(1).jpg')

plt.subplot(3, 3, 1), plt.imshow(x_predict[0])
plt.subplot(3, 3, 2), plt.imshow(x_predict[1])
plt.subplot(3, 3, 3), plt.imshow(x_predict[2])

plt.subplot(3, 3, 4), plt.imshow(pissarro_Img1)
plt.subplot(3, 3, 5), plt.imshow(hassam_Img1)
plt.subplot(3, 3, 6), plt.imshow(monet_Img1)

plt.subplot(3, 3, 7), plt.imshow(pissarro_Img2)
plt.subplot(3, 3, 8), plt.imshow(hassam_Img2)
plt.subplot(3, 3, 9), plt.imshow(monet_Img2)

plt.show()

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