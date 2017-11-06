import pandas as pd
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

base_address = '/home/bhavika/wikiart/'


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
    more_than_30 = {k: v for k, v in artist_paintings_count.items() if v >= 30}

    print("{} artists with 40 paintings or more: {}".format(len(more_than_40), more_than_40))
    print("{} artists with 30 paintings or more: {}".format(len(more_than_30), more_than_30))


def create_traintest(filepath):
    data = pd.read_csv(filepath, sep=',')
    # Filter artists that appear only once - since we don't have enough train/test images for them
    counts = data['class'].value_counts()
    data = data[data['class'].isin(counts[counts > 1].index)]
    train, test = train_test_split(data, train_size=0.6, shuffle=True, stratify=data['class'])
    train.to_csv('../data/train_full.csv', sep=',', index=False)
    test.to_csv('../data/test_full.csv', sep=',', index=False)

    print(train.shape)
    print(test.shape)
    # print(train.sort_values(by='class'))
    # print(test.sort_values(by='class'))
    # labels_train = train['class'].unique()
    # labels_test = test['class'].unique()
    #
    # print(set(labels_train) ^ set(labels_test))


def create_dataset(filepath):
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
    frida.to_csv('../data/impressionists.csv', sep=',', index=True, index_label='ids')


if __name__ == '__main__':
    train_file = '/artist_train.csv'
    style = 'Impressionism'
    # read_artist_data(base_address=base_address,filepath=train_file)
    # select_impressionist_artists(base_address+style)
    # create_dataset(base_address+style)
    create_traintest('../data/impressionists.csv')