from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from PIL import Image, ImageFilter, ImageFile, ImageStat
import pandas as pd
import time
#import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from pathlib import Path

from util_image import find_blur_value, find_edge_enhance, find_smooth_more, find_brightness, find_edge, find_contour, \
    find_emboss, find_edge_enhance_more, find_detail, find_smooth, find_similiar_image
from scipy.stats import randint as sp_randint
from Constant import base_address, style,normal_demo_csv_file,mapping_file, base_csv_file_location, saved_trained_file
import numpy as np

filepath = base_address
unique_artists = set()
unique_link = list()
mapping = dict()
print("")
print("Load the dataset from csv file")
print("")

#file categories are ids,location,artist,class
paintings_by_artist = pd.read_csv(filepath+normal_demo_csv_file, names=['ids','location','artist', 'class'], header=0)
paintings_by_artist['absolute_location'] = base_address +style+ paintings_by_artist['location']

mapping_artists= pd.read_csv(filepath+mapping_file, names=['class','artist'], header=0)

for i in range(mapping_artists.shape[0]):
    label= mapping_artists.iloc[i]['class']
    artist = mapping_artists.iloc[i]['artist']
    #print(label,artist)
    mapping.update({label:artist})

#paintings_by_artist = paintings_by_artist.sort_values('class')
ImageFile.LOAD_TRUNCATED_IMAGES = True

rows_count = paintings_by_artist.shape[0]
cols_count = 10
actual_image_count = 0

#investigate the actual number of images
for i in range(paintings_by_artist.shape[0]):
    link = paintings_by_artist.iloc[i]['absolute_location']
    #paintings_by_artist = paintings_by_artist.sort_values('class')
    my_file = Path(link)
    if my_file.is_file():
        actual_image_count += 1
        unique_link.append(link)



feature= [[0 for j in range(cols_count)] for i in range(actual_image_count)]
label_data = [0 for i in range(actual_image_count)]
image_data= []


print("Calculate the feature values of each paintings ")
print()
print("Features are brightness,blur,edge,contour,emboss,smooth, and so on")
print()

#read records one by one
#for i in range(paintings_by_artist.shape[0]):
for i in range(len(list(unique_link))):
    #link = paintings_by_artist.iloc[i]['absolute_location']
    link = list(unique_link)[i]
    #paintings_by_artist = paintings_by_artist.sort_values('class')
    global img
    #print label_data
    my_file = Path(link)
    if my_file.is_file():
        img = Image.open(link,'r')
        image = img.resize((224,224))
        image = np.array(image).astype(np.float32)
        image_data.append(img)

        #img.show()
    else :
        continue

    #transform the image value to statistical value
    stat = ImageStat.Stat(img)

    #generate the class data
    label_data[i] = paintings_by_artist.iloc[i]['class']

    ##before feature selection
    #feature values : blur, brightness, edge, edge_enhance, edge_enhance_more, contour,emboss, detail, smooth, smooth_more
    feature[i][0] = find_brightness(img)
    feature[i][1] = find_blur_value(link)
    feature[i][2] = find_edge(link)
    feature[i][3] = find_edge_enhance(img)
    feature[i][4] = find_edge_enhance_more(img)
    feature[i][5] = find_contour(img)
    feature[i][6] = find_emboss(img)
    feature[i][7] = find_detail(img)
    feature[i][8] = find_smooth(img)
    feature[i][9] = find_smooth_more(img)

    unique_artists.add(label_data[i])



#X_train, X_test, y_train, y_test , image_train, image_test = train_test_split(feature, label_data,image_data, test_size=0.25)
#print("x_test",X_test)
print()
#print("y_test",mapping.get(y_test))

#for i in range(len(y_test)) :
#    print("y_test",mapping.get(y_test[i]))

start_time= time.time()
filename = './wikiart/finalized_model_random_forest.sav'
filename = base_csv_file_location+saved_trained_file
#loaded_model = pickle.load(open(filename, 'rb'))
from sklearn.externals import joblib
loaded_model = joblib.load(filename)

x_predict = loaded_model.predict(feature)
#result = loaded_model.score(X_test, y_test)
#result = loaded_model.score(numpy.array(x_predict).reshape(-1,1), numpy.array( y_test).reshape(-1,1))
#print(result)


data = precision_recall_fscore_support(label_data, x_predict, average='micro')
print("precision_recall_fscore_support result:",data)
print ("Confusion Matrix:\n", confusion_matrix(label_data, x_predict))
print ("Random Accuracy :", accuracy_score(label_data, x_predict)*100)
print ("precision :", data[0])
print ("recall:", data[1])
print("--- %s seconds ---" % (time.time() - start_time))


# Import matplotlib

# Assign the predicted values to `predicted`


# Zip together the `images_test` and `predicted` values in `images_and_predictions`
images_and_predictions = list(zip(image_data, x_predict))

f, axarr = plt.subplots(2,2)
for index, (image, prediction) in enumerate(images_and_predictions[:2]):
    axarr[index, 0].imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    axarr[index, 0].set_title('Predicted: ' + str(mapping.get(prediction)))
    axarr[index, 1].imshow(find_similiar_image(prediction,image), cmap=plt.cm.gray_r, interpolation='nearest')
    axarr[index, 1].set_title(str(mapping.get(prediction))+' similar image ')
plt.show()

#axarr[1,0].imshow(image_datas[2])
#axarr[1,1].imshow(image_datas[3])