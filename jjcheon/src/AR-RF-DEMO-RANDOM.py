from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from PIL import Image, ImageFilter, ImageFile, ImageStat
import pandas as pd
import time
#import pickle
import matplotlib.pyplot as plt

from util_image import *
from scipy.stats import randint as sp_randint
from Constant import *
import numpy as np

filepath = base_address
unique_artists = set()
unique_link = list()
mapping = dict()
print("")
print("Load the dataset from csv file")
print("")

#file categories are ids,location,artist,class
#paintings_by_artist = pd.read_csv(filepath+train_file, names=['ids','location','artist', 'class'], header=0)
#paintings_by_artist['absolute_location'] = base_address +random_style+ paintings_by_artist['location']

mapping_artists= pd.read_csv(filepath+mapping_file, names=['class','artist'], header=0)

for i in range(mapping_artists.shape[0]):
    label= mapping_artists.iloc[i]['class']
    artist = mapping_artists.iloc[i]['artist']
    #print(label,artist)
    mapping.update({label:artist})
#
##paintings_by_artist = paintings_by_artist.sort_values('class')
#ImageFile.LOAD_TRUNCATED_IMAGES = True
#
#rows_count = paintings_by_artist.shape[0]
cols_count = 10
actual_image_count = 1
#
##investigate the actual number of images
#for i in range(paintings_by_artist.shape[0]):
#    link = paintings_by_artist.iloc[i]['absolute_location']
#    #paintings_by_artist = paintings_by_artist.sort_values('class')
#    my_file = Path(link)
#    if my_file.is_file():
#        actual_image_count += 1
#        unique_link.append(link)
#
#
#
feature= [[0 for j in range(cols_count)] for i in range(actual_image_count)]
label_data = [0 for i in range(10)]
image_data= []
#

print("Calculate the feature values of each paintings ")
print()
print("Features are brightness,blur,edge,contour,emboss,smooth, and so on")
print()
print ("Split arrays or matrices into random train and test subsets")
print ("X_train, X_test, y_train, y_test, image_train, image_test ")

print ("Calculate feature importance")
print("# Tuning hyper-parameters by GridSearchCV")
print("Train with X_train, y_train data by using StratifiedKFold Cross Validation.")
print("Test with X_test, y_test data by using StratifiedKFold Cross Validation.")
print("Predict class for X_test.")


#absolute_link  = base_address+'Random/tree.jpg'
img = Image.open(absolute_link,'r')
image = img.resize((224,224))
image = np.array(image).astype(np.float32)
image_data.append(img)
#    else :
#        continue

    #transform the image value to statistical value
stat = ImageStat.Stat(img)
i=0
feature[i][0] = find_brightness(img)
feature[i][1] = find_blur_value(absolute_link)
feature[i][2] = find_edge(absolute_link)
feature[i][3] = find_edge_enhance(img)
feature[i][4] = find_edge_enhance_more(img)
feature[i][5] = find_contour(img)
feature[i][6] = find_emboss(img)
feature[i][7] = find_detail(img)
feature[i][8] = find_smooth(img)
feature[i][9] = find_smooth_more(img)

start_time= time.time()
#loaded_model_filename = './wikiart/finalized_model_random_forest.sav'
#loaded_model = pickle.load(open(filename, 'rb'))
from sklearn.externals import joblib
loaded_model = joblib.load(loaded_model_filename)
print("feature",feature)
#x_predict = loaded_model.predict(np.array(feature).reshape(-1,1))
x_predict = loaded_model.predict(feature)



# Zip together the `images_test` and `predicted` values in `images_and_predictions`
images_and_predictions = list(zip(image, x_predict))

#f, axarr = plt.subplots(2,2)
#for index, (image, prediction) in enumerate(images_and_predictions[:1]):
#    axarr[index, 0].imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
#    axarr[index, 0].set_title('Predicted: ' + str(mapping.get(prediction)))
#    axarr[index, 1].imshow(find_similiar_image(prediction,img), cmap=plt.cm.gray_r, interpolation='nearest')
#    axarr[index, 1].set_title(str(mapping.get(prediction))+' similar image ')
#plt.show()

f1, axarr1 = plt.subplots(1,2)
plt.axis('off')
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
for index, (image, prediction) in enumerate(images_and_predictions[:1]):
    axarr1[0].imshow(img)
    axarr1[0].set_title('Picture chosen from Google without a label ')
    axarr1[1].text(0.0,0.5,'Author Prediction :'+str(mapping.get(prediction)),
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
plt.show()

image_data = find_author_image(prediction)
f, axarr = plt.subplots(1,5,squeeze=False)
plt.axis('off')
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
for index, (image, prediction) in enumerate(images_and_predictions[:1]):
    axarr[index, 0].imshow(image_data[0])
    axarr[index, 1].imshow(image_data[1])
    axarr[index, 2].imshow(image_data[2])
    axarr[index, 3].imshow(image_data[3])
    axarr[index, 4].imshow(image_data[4])
plt.show()
