from PIL import Image, ImageFilter, ImageFile, ImageStat
import pandas as pd
import time
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, make_scorer, explained_variance_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from util_image import find_blur_value, find_edge_enhance, find_smooth_more, find_brightness, find_edge, find_contour, \
    find_emboss, find_edge_enhance_more, find_detail, find_smooth
from scipy.stats import randint as sp_randint
from util_file import make_author_mapping_file, make_author_image_mapping
from Constant import style, base_address, train_file

#load the impressionist csv and images file
#base_address = '/home/jay/PycharmProjects/688-project/wikiart/'
#style = 'Impressionism-5a-10p/'
#train_file = 'impressionists-5a-10p.csv'

filepath = base_address
unique_artists = set()
unique_link = list()

print("")
print("Load the dataset from csv file ",train_file)
print("")

#file categories are ids,location,artist,class
paintings_by_artist = pd.read_csv(filepath+train_file, names=['ids','location','artist', 'class'], header=0)
paintings_by_artist['absolute_location'] = base_address +style+ paintings_by_artist['location']

#make the mapping between class and the author
make_author_mapping_file(filepath,train_file)


#paintings_by_artist = paintings_by_artist.sort_values('class')
ImageFile.LOAD_TRUNCATED_IMAGES = True

rows_count = paintings_by_artist.shape[0]
cols_count = 10
actual_image_count = 0

#investigate the actual number of images
for i in range(paintings_by_artist.shape[0]):
    link = paintings_by_artist.iloc[i]['absolute_location']
    my_file = Path(link)
    if my_file.is_file():
        actual_image_count += 1
        unique_link.append(link)


feature= [[0 for j in range(cols_count)] for i in range(actual_image_count)]
label_data = [0 for i in range(actual_image_count)]
image_stat_data = [0.0 for i in range(actual_image_count)]
image_data= []


print("Calculate the feature values of each paintings ")
print("Features are brightness,blur,edge,contour,emboss,smooth, and so on")
print()


#read records one by one
for i in range(len(list(unique_link))):
    #link = paintings_by_artist.iloc[i]['absolute_location']
    link = list(unique_link)[i]
    #paintings_by_artist = paintings_by_artist.sort_values('class')
    global img
    #print label_data
    my_file = Path(link)
    if my_file.is_file():
        img = Image.open(link,'r')
        image_data.append(img)
    else :
        continue

    #transform the image value to statistical value
    stat = ImageStat.Stat(img)

    #generate the class data
    #label_data[i]= paintings_by_artist.iloc[i]['class']
    label_data[i] =paintings_by_artist.iloc[i]['class']


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

make_author_image_mapping(label_data, image_data)

print ("")
print ("Split arrays or matrices into random train and test subsets")
print ("X_train, X_test, y_train, y_test, image_train, image_test ")
X_train, X_test, y_train, y_test , image_train, image_test = train_test_split(feature, label_data,image_data, test_size=0.25)

print("X_train length", len(X_train),"X_test len",len(X_test),"image_train len",len(image_train), "image_test len",len(image_test))

# Zip together the `images_test` and `predicted` values in `images_and_predictions`
train_and_images = list(zip(X_train, image_train))

print ("")
print ("Calculate feature importance")

clf = RandomForestClassifier(n_jobs=10, random_state=0)
#clf = clf.fit(X_train, y_train)

rfecv = RFECV(estimator=clf, step=1,
              scoring='accuracy')
rfecv.fit(X_train, y_train)

mask = rfecv.get_support(indices=True)
print("Selected feature indexies", mask)
print ("")

# Create a random forest Classifier. By convention, clf means 'Classifier'
num_feature = sp_randint(1,16)
tuned_parameters = {"max_depth": [3,None],
              #"max_features": [1,10,13,15],
              "min_samples_split": [2,5,10,16],
              "min_samples_leaf": [1,3,9],
              "bootstrap":[True, False],
              "criterion":["gini","entropy"]}

#scores = ['precision']
scores= {'accuracy': make_scorer(accuracy_score)}

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf2 = GridSearchCV(ExtraTreesClassifier(), tuned_parameters, cv=3,
    #clf2=GridSearchCV(clf, tuned_parameters, cv=5,
                                          #scoring = '%s_macro' % score, verbose=True)
                       #scoring = '%s' % score, verbose = True)
                       scoring = None)
    clf2.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf2.best_params_)
    print()

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
print()
# Apply the Classifier we trained to the test data (which, remember, it has never seen before)

start_time = time.time()
clf4 = ExtraTreesClassifier(max_depth=clf2.best_params_['max_depth'],
                            #max_features=clf2.best_params_['max_features'],
                            min_samples_split=clf2.best_params_['min_samples_split'],
                            min_samples_leaf =clf2.best_params_['min_samples_leaf'],
                            bootstrap= clf2.best_params_['bootstrap'],
                            criterion= clf2.best_params_['criterion'], n_jobs=10)


print()
print("Train with X_train, y_train data by using StratifiedKFold Cross Validation.")
print()

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X_train, y_train):
    clf4.fit(X_train, y_train)

print("Test with X_test, y_test data by using StratifiedKFold Cross Validation.")
print()

#from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X_test, y_test):
    clf4.fit(X_test, y_test)
filename = 'finalized_model_random_forest.sav'
pickle.dump(clf4, open(filename, 'wb'))
print("y_test",y_test)
print("Predict class for X_test.")
x_predict = clf4.predict(X_test)
print("x_predict",x_predict)

print()
print("Show the estimation result .")
print()

from sklearn.metrics import confusion_matrix
print ("Confusion Matrix:\n", confusion_matrix(y_test, x_predict))
print ("Random Accuracy :", accuracy_score(y_test, x_predict)*100)
print("--- %s seconds ---" % (time.time() - start_time))


