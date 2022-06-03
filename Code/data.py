import glob
import pandas as pd

TRAIN_DATA_PATH = r'/Users/moritzdeecke/Desktop/Spotify_Dataset/training_set'
TEST_DATA_PATH = r'/Volumes/ExtremeSSD/Spotify_Dataset/extracted/Extraction/test/testset/test_set1.16'
TRACK_FEATURE_PATH = r'/Users/moritzdeecke/Desktop/Spotify_Dataset/track_features'


def load_track_features():
    feature_files = glob.glob(TRACK_FEATURE_PATH + "/*.csv")

    lis1 = []
    for filename in feature_files:
        df1 = pd.read_csv(filename, index_col=None, header=0)
        lis1.append(df1)
    track_features = pd.concat(lis1, axis=0, ignore_index=True)

    return track_features


def load_train_data():
    test_files = glob.glob(TRAIN_DATA_PATH + "/*.csv")

    l = []
    for filename in test_files:
        print('Reading file:', filename)
        df = pd.read_csv(filename, usecols=['track_id_clean', 'not_skipped'], index_col=None, header=0)
        l.append(df)

    data = pd.concat(l, axis=0, ignore_index=True)
    data = data.rename(columns={'track_id_clean': 'track_id'})

    return data


def load_test_data():
    test_files = glob.glob(TEST_DATA_PATH + "/*.csv")

    lis2 = []

    for filename in test_files:
        print('Reading file:', filename)
        df2 = pd.read_csv(filename, usecols=['track_id_clean', 'not_skipped'], index_col=None, header=0)
        lis2.append(df2)

    test_data = pd.concat(lis2, axis=0, ignore_index=True)
    test_data = test_data.rename(columns={'track_id_clean': 'track_id'})

    return test_data
