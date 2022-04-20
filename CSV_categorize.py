import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

USER_COLUMNS = ["session_id", "session_position", "session_length", "no_pause_before_play",
                "short_pause_before_play",
                "long_pause_before_play", "hist_user_behavior_n_seekfwd",
                "hist_user_behavior_n_seekback",
                "hist_user_behavior_is_shuffle", "hour_of_day", "date", "premium",
                "hist_user_behavior_reason_start",
                "hist_user_behavior_reason_end"]

def sns_plot_count(ds_clean):
    assert type(ds_clean) == pd.DataFrame, "wrong type"
    for col in ds_clean:
        print(col)
        if col in ['hist_user_behavior_is_shuffle', 'premium', 'track_id', 'skip_1', 'skip_2', 'skip_3', 'not_skipped']:
            continue
        sns.histplot(data=ds_clean, x=col, stat="count", common_norm=False, bins=40, multiple="dodge")
        plt.show()

def sns_plot(ds_clean):
    assert 'not_skipped' in ds_clean, "forgot not_skipped"
    for col in ds_clean:
        print(col)
        if col in ['track_id', 'skip_1', 'skip_2', 'skip_3', 'not_skipped']:
            continue
        sns.histplot(data=ds_clean, x=col, hue="not_skipped", stat="probability", common_norm=False, bins=40)
        plt.show()


def filter_df(df, columns):
    assert columns in ['user', 'music', 'both']
    if columns == 'both':
        return df
    elif columns == 'user':
        userdata = df[USER_COLUMNS]
        return userdata
    elif columns == 'music':
        musicdata = df.drop(columns=USER_COLUMNS + ['not_skipped', 'skip_1', 'skip_2', 'skip_3'])
        return musicdata


def convert_df_to_numeric(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = pd.factorize(df[col])[0]
    return df


def convert_df_to_array(df):
    array = np.array(df).astype(np.float32)
    return array


def convert_array_to_torch(array):
    return torch.from_numpy(array).type(torch.FloatTensor)