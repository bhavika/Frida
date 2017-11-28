import pandas as pd
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

base_address = 'C:\\Users\\Melanie\\Desktop\\'


def read_artist_data(base_address, filepath):
    """
    Read artist_train.csv to view each image. 
    :param base_address: str, absolute location for the wikiart dataset
    :param filepath: str, absolute location for the train/eval file to be read. 
    """
    filepath = base_address + filepath
    paintings_by_artist = pd.read_csv(filepath,names=['location', 'class'], header=0)
    paintings_by_artist['absolute_location'] = base_address+paintings_by_artist['location']
    paintings_by_artist = paintings_by_artist.sort_values('class')
    for i in range(paintings_by_artist.shape[0]):
        link = paintings_by_artist.iloc[i]['absolute_location']
        print(link)
    img = Image.open(link)
    img.show()


def select_impressionist_artists(filepath):
    """
    We want to find out how many impressionist artists are present in the dataset 
    to analyze data needs and availability. 
    Required: Atleast 15 artists with 40 paintings each. 
    :param filepath: absolute location of Impressionism folder. 
    """
    paintings = os.listdir(filepath)
    unique_artists = set()
    artist_paintings_count = {}

    for filename in paintings:
        s = filename.find('_')
        artist_name = filename[:s]
        unique_artists.add(artist_name)
        if artist_name in artist_paintings_count.keys():
            artist_paintings_count[artist_name] += 1
        else:
            artist_paintings_count[artist_name] = 1
    print("Total Unique Artists in the Impressionism style:", len(unique_artists))
    print("Paintings by each artist:", artist_paintings_count)

    more_than_40 = {k: v for k,v in artist_paintings_count.items() if v >= 40}
    more_than_300 = {k: v for k, v in artist_paintings_count.items() if v >= 300}

    print("{} artists with 40 paintings or more: {}".format(len(more_than_40), more_than_40))
    print("{} artists with 300 paintings or more: {}".format(len(more_than_300), more_than_300))


def create_traintest(filepath, target_suffix):
    data = pd.read_csv(filepath, sep=',')
    # Filter artists that appear only once - since we don't have enough train/test images for them
    counts = data['class'].value_counts()
    data = data[data['class'].isin(counts[counts > 1].index)]
    train, test = train_test_split(data, train_size=0.6, shuffle=True, stratify=data['class'])
    train.to_csv('../data/train_{}.csv'.format(target_suffix), sep=',', index=False)
    test.to_csv('../data/test_{}.csv'.format(target_suffix), sep=',', index=False)

    print(train.shape)
    print(test.shape)
    # print(train.sort_values(by='class'))
    # print(test.sort_values(by='class'))
    # labels_train = train['class'].unique()
    # labels_test = test['class'].unique()
    #
    # print(set(labels_train) ^ set(labels_test))


def create_dataset(filepath, artist_count, image_count):
    """
    We separate out the Impressionist era data and artists from that period. Our train, test & validation sets will 
    be based on this subset only. 
    Creates a CSV file with the image location, artist and class label (encoded for artists). 
    :param filepath: str, the base location where Impressionist paintings are stored. 
    """
    paintings = os.listdir(filepath)
    frida_data = {}
    for filename in paintings:
        s = filename.find('_')
        artist_name = filename[:s]
        frida_data[filename] = artist_name

    frida = pd.DataFrame(list(frida_data.items()), columns=['location', 'artist'])
    le = LabelEncoder()
    labels = le.fit(frida['artist'])
    frida['class'] = le.transform(frida['artist'])

    counts = frida['class'].value_counts()
    filter = list(counts[counts > image_count].index)

    # filter for artists that appear image_count times or more
    more_than_img_count = (frida[frida['class'].isin(filter)])
    topn = more_than_img_count.groupby('class').head(image_count)
    topn.sort_values(by='class', inplace=True, ascending=True)
    topn = topn.head(artist_count*image_count)

    # Reindex these class labels to start with 0

    labels = le.fit(topn['artist'])
    topn['class'] = le.transform(topn['artist'])

    mappings = dict(zip(le.classes_, le.transform(le.classes_)))
    mappings_df = pd.DataFrame(list(mappings.items()), columns=['artist', 'class'])

    mappings_df.to_csv('C:/Users/Melanie/Desktop/csvFiles/data/top{}_mappings.csv'.format(artist_count), sep=',')

    print("Selected {} paintings for each of the {} artists".format(image_count, artist_count))
    print(topn['class'].value_counts())

    topn.to_csv('C:/Users/Melanie/Desktop/csvFiles/data/top{}.csv'.format(artist_count), sep=',', index=True, index_label='ids')
    frida.to_csv('C:/Users/Melanie/Desktop/csvFiles/data/impressionists.csv', sep=',', index=True, index_label='ids')


def validate_traintest(train_path, test_path):
    """
    Validate train test datasets for Pytorch  
    - class indices should start at 0
    - no test classes should be absent in train classes
    :return: 
    """

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_classes = train['class'].unique()
    test_classes = test['class'].unique()

    print("Class matches in train and test: ", sorted(train_classes) == sorted(test_classes))
    print("Train index starts with 0: ", sorted(train_classes)[0] == 0)
    print("Test index starts with 0: ", sorted(test_classes)[0] == 0)


if __name__ == '__main__':
    train_file = '/artist_train.csv'
    style = 'Impressionism'
    # read_artist_data(base_address=base_address,filepath=train_file)
    select_impressionist_artists(base_address+style)
    create_dataset(base_address+style, artist_count=15, image_count=50)
    dsname = 'top15'
    #create_traintest('../data/{}.csv'.format(dsname), dsname)
    #validate_traintest('../data/train_{}.csv'.format(dsname), '../data/test_{}.csv'.format(dsname))