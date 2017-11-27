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
    i=0
    max_index = 0
    while i < 14:
        #print(modelPredictions[imgNum][i])
        i += 1
        if modelPredictions[imgNum][i] > modelPredictions[imgNum][i-1]:
            max_index = i
    #print(max_index)
    
    if actual == max_index:
        totalCorrect += 1
        if actual == 0:
            baltatuCorrect += 1
        if actual == 1:
            sisleyCorrect += 1
        if actual == 2:
            blanchardCorrect += 1
        if actual == 3:
            kuindzhiCorrect += 1
        if actual == 4:
            guillauminCorrect += 1
        if actual == 5:
            rodinCorrect += 1
        if actual == 6:
            morisotCorrect += 1
        if actual == 7:
            pissarroCorrect += 1
        if actual == 8:
            hassamCorrect += 1
        if actual == 9:
            monetCorrect += 1
        if actual == 10:
            artachinoCorrect += 1
        if actual == 11:
            vreedenburghCorrect += 1
        if actual == 12:
            degasCorrect += 1
        if actual == 13:
            manetCorrect += 1
        if actual == 14:
            boudinCorrect += 1

    imgNum += 1

print("Done.")
print("Accuracy by Artist:")
print("Baltatu:")
print(baltatuCorrect/constants.test_samples)
print("Sisley:")
print(sisleyCorrect/constants.test_samples)
print("Blanchard:")
print(blanchardCorrect/constants.test_samples)
print("Kuindzhi:")
print(kuindzhiCorrect/constants.test_samples)
print("Guillaumin:")
print(guillauminCorrect/constants.test_samples)
print("Rodin:")
print(rodinCorrect/constants.test_samples)
print("Morisot:")
print(morisotCorrect/constants.test_samples)
print("Pissarro:")
print(pissarroCorrect/constants.test_samples)
print("Hassam:")
print(hassamCorrect/constants.test_samples)
print("Monet:")
print(monetCorrect/constants.test_samples)
print("Artachino:")
print(artachinoCorrect/constants.test_samples)
print("Vreendenburgh:")
print(vreedenburghCorrect/constants.test_samples)
print("Degas:")
print(degasCorrect/constants.test_samples)
print("Manet:")
print(manetCorrect/constants.test_samples)
print("Boudin:")
print(boudinCorrect/constants.test_samples)
print("Total Correct:")
print(totalCorrect)
print("Number of Images:")
print(totalImages)
print("Overall Accuracy:")
print (totalCorrect/totalImages)
