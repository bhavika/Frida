'''
Created on Nov 5, 2017
@author: Melanie
'''
import LoadDataset
from keras.models import load_model
import constants

# returns a compiled model
# identical to the previous one

model = load_model(constants.model_path)
print("Model has been reloaded.")

(x_predict, y_predict, z_predict) = LoadDataset.getDatasetForPrediction([0])

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
    print(modelPredictions[imgNum][0], modelPredictions[imgNum][1], modelPredictions[imgNum][2],
          modelPredictions[imgNum][3], modelPredictions[imgNum][4], modelPredictions[imgNum][5],
          modelPredictions[imgNum][6], modelPredictions[imgNum][7], modelPredictions[imgNum][8],
          modelPredictions[imgNum][9])
#    if modelPredictions[imgNum][0] > modelPredictions[imgNum][1]:
#        prediction = 0
#
#    if prediction == 1:
#        if actual == 1:
#            predictCarCorrect += 1
#            if predictCarCorrect <= 10:
#                print("TP: " + str(z_predict[imgNum]))
#        else:
#            predictCarNotCorrect += 1
#            if predictCarNotCorrect <= 10:
#                print("FP: " + str(z_predict[imgNum]))
#
#    elif prediction == 0:
#        if actual == 0:
#            predictNotCarCorrect += 1
#            if predictNotCarCorrect <= 10:
#                print("TN: " + str(z_predict[imgNum]))
#        else:
#            predictNotCarNotCorrect += 1
#            if predictNotCarNotCorrect <= 10:
#                print("FN: " + str(z_predict[imgNum]))
#    else:
#        print("Ooops - bad prediction value")

    imgNum += 1

print("Done.")
#print("predictCarCorrect = " + str(predictCarCorrect))
#print("predictCarNotCorrect = " + str(predictCarNotCorrect))
#print("predictNotCarCorrect = " + str(predictNotCarCorrect))
#print("predictNotCarNotCorrect = " + str(predictNotCarNotCorrect))
