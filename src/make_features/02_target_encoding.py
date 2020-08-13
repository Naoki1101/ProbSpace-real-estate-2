import sys
import pandas as pd
from easydict import EasyDict as edict

sys.path.append('../src')
from utils import DataHandler
from factory import get_fold
from feature_utils import save_features, TargetEncoding


def get_features(train, test):
    train_features_df = pd.DataFrame()
    test_features_df = pd.DataFrame()

    # TargetEncoding
    cfg = edict({
        'name': 'KFold',
        'params': {
            'n_splits': 5,
            'shuffle': True,
            'random_state': 0,
        },
        'split': {
            'y': 'y',
            'groups': None
        },
        'weight': [1.0]
        })
    fold_df = get_fold(cfg, train, train[['y']])

    te = TargetEncoding(fold_df)
    train_features_df['te_land_type'] = te.fit_transform(train['種類'], train['y'])
    test_features_df['te_land_type'] = te.transform(test['種類'])

    te = TargetEncoding(fold_df)
    train_features_df['te_region'] = te.fit_transform(train['地域'], train['y'])
    test_features_df['te_region'] = te.transform(test['地域'])

    te = TargetEncoding(fold_df)
    train_features_df['te_region_code'] = te.fit_transform(train['市区町村コード'], train['y'])
    test_features_df['te_region_code'] = te.transform(test['市区町村コード'])

    te = TargetEncoding(fold_df)
    train_features_df['te_city_name'] = te.fit_transform(train['市区町村名'], train['y'])
    test_features_df['te_city_name'] = te.transform(test['市区町村名'])

    te = TargetEncoding(fold_df)
    train_features_df['te_town_name'] = te.fit_transform(train['地区名'], train['y'])
    test_features_df['te_town_name'] = te.transform(test['地区名'])

    te = TargetEncoding(fold_df)
    train_features_df['te_station_name'] = te.fit_transform(train['最寄駅：名称'], train['y'])
    test_features_df['te_station_name'] = te.transform(test['最寄駅：名称'])

    te = TargetEncoding(fold_df)
    train_features_df['te_land_shape'] = te.fit_transform(train['土地の形状'], train['y'])
    test_features_df['te_land_shape'] = te.transform(test['土地の形状'])

    te = TargetEncoding(fold_df)
    train_features_df['te_structure'] = te.fit_transform(train['建物の構造'], train['y'])
    test_features_df['te_structure'] = te.transform(test['建物の構造'])

    te = TargetEncoding(fold_df)
    train_features_df['te_feature_purpose'] = te.fit_transform(train['今後の利用目的'], train['y'])
    test_features_df['te_feature_purpose'] = te.transform(test['今後の利用目的'])

    te = TargetEncoding(fold_df)
    train_features_df['te_renovation'] = te.fit_transform(train['改装'], train['y'])
    test_features_df['te_renovation'] = te.transform(test['改装'])

    te = TargetEncoding(fold_df)
    train_features_df['te_city_planning'] = te.fit_transform(train['都市計画'], train['y'])
    test_features_df['te_city_planning'] = te.transform(test['都市計画'])

    te = TargetEncoding(fold_df)
    train_features_df['te_direction'] = te.fit_transform(train['前面道路：方位'], train['y'])
    test_features_df['te_direction'] = te.transform(test['前面道路：方位'])

    te = TargetEncoding(fold_df)
    train_features_df['te_load_type'] = te.fit_transform(train['前面道路：種類'], train['y'])
    test_features_df['te_load_type'] = te.transform(test['前面道路：種類'])

    return train_features_df, test_features_df


def main():
    train_df = pd.read_feather('../data/input/train_data.feather')
    test_df = pd.read_feather('../data/input/test_data.feather')

    train_features_df, test_features_df = get_features(train_df, test_df)

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()