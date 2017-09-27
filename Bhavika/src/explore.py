import pandas as pd
from PIL import Image
import os

base_address = '/home/bhavika/wikiart/'


def read_artist_data(base_address, filepath):
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

if __name__ == '__main__':
    train_file = '/artist_train.csv'
    read_artist_data(base_address=base_address,filepath=train_file)
    select_impressionist_artists(base_address+'Impressionism')