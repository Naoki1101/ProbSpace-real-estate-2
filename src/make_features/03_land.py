import numpy as np
import pandas as pd

from feature_utils import save_features


def get_features(df):
    features_df = pd.DataFrame()

    df['延床面積（㎡）'] = df['延床面積（㎡）'].replace({'2000㎡以上': '2000', '10m^2未満': '9'})
    df['面積（㎡）'] = df['面積（㎡）'].replace({'2000㎡以上': 2000, '5000㎡以上': 5000})

    features_df['land_unit_price'] = df['Ｈ３１価格']
    features_df['estimated_land_price'] = df['Ｈ３１価格'] * df['延床面積（㎡）'].astype(float)
    features_df['estimated_land_price2'] = df['Ｈ３１価格'] * df['面積（㎡）'].astype(float)

    features_df['distance_from_tokyo_station'] = np.sqrt((df['経度'] - 503172.86)  ** 2 + (df['緯度'] - 128451.924)  ** 2)

    return features_df


def main():
    train_df = pd.read_feather('../data/input/train_data.feather')
    test_df = pd.read_feather('../data/input/test_data.feather')

    land_usecols = ['経度', '緯度', 'Ｈ３１価格']
    land_df = pd.read_csv('../data/input/published_land_price.csv', usecols=land_usecols + ['駅名'])

    station_df = land_df.groupby('駅名')[land_usecols].median().reset_index()

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_df = whole_df.merge(station_df, left_on=['最寄駅：名称'], right_on=['駅名'], how='left')

    whole_features_df = get_features(whole_df)

    train_features_df = whole_features_df.iloc[:len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df):]

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()
