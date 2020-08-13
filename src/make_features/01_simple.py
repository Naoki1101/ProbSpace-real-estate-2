import pandas as pd

from feature_utils import save_features


def label_encode(v):
    le = {k: i for i, k in enumerate(v.unique())}
    v_encoded = v.map(le)
    return v_encoded


def extract_year(x):
    year = int(x[2: -1])
    if '昭和' in x:
        return year + 1925
    elif '平成' in x:
        return year + 1988


def get_features(df):
    features_df = pd.DataFrame()

    df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].replace({'30分?60分': '45', '1H?1H30': '75', '1H30?2H': '105', '2H?': '120'})
    df['面積（㎡）'] = df['面積（㎡）'].replace({'2000㎡以上': 2000, '5000㎡以上': 5000})
    df['間口'] = df['間口'].replace({'50.0m以上': '50'})
    df['延床面積（㎡）'] = df['延床面積（㎡）'].replace({'2000㎡以上': '2000', '10m^2未満': '9'})
    df['建築年'] = df['建築年'].replace({'戦前': '昭和20年'}).fillna('昭和-1年')

    features_df['land_type'] = label_encode(df['種類'])
    features_df['region'] = label_encode(df['地域'])
    features_df['region_code'] = label_encode(df['市区町村コード'])
    features_df['city_name'] = label_encode(df['市区町村名'])
    features_df['town_name'] = label_encode(df['地区名'])
    features_df['station_name'] = label_encode(df['最寄駅：名称'])
    features_df['distance_minutes'] = df['最寄駅：距離（分）'].astype(float)
    features_df['layout'] = label_encode(df['間取り'])
    features_df['layout_str_length'] = df['間取り'].apply(lambda x: len(x) if type(x) == str else None)
    features_df['layout_contain_L'] = df['間取り'].apply(lambda x: 1 if type(x) == str and 'Ｌ' in x else 0)
    features_df['layout_contain_S'] = df['間取り'].apply(lambda x: 1 if type(x) == str and 'Ｓ' in x else 0)
    features_df['layout_num'] = df['間取り'].apply(lambda x: int(x[0]) if str(x)[0] in ['１', '２', '３', '４', '５', '６', '７'] else None)
    features_df['area'] = df['面積（㎡）'].astype(float)
    features_df['land_shape'] = label_encode(df['土地の形状'])
    features_df['frontage'] = df['間口'].astype(float)
    features_df['all_area'] = df['延床面積（㎡）'].astype(float)
    features_df['year_name'] = label_encode(df['建築年'].apply(lambda x: str(x)[:2]))
    features_df['build_year'] = df['建築年'].apply(extract_year)
    features_df['structure'] = label_encode(df['建物の構造'])
    features_df['purpose'] = label_encode(df['用途'])
    features_df['purpose_contain_residence'] = df['用途'].apply(lambda x: 1 if type(x) == str and '住宅' in x else 0)
    features_df['purpose_contain_apartment'] = df['用途'].apply(lambda x: 1 if type(x) == str and '共同住宅' in x else 0)
    features_df['purpose_contain_store'] = df['用途'].apply(lambda x: 1 if type(x) == str and '店舗' in x else 0)
    features_df['purpose_contain_parking'] = df['用途'].apply(lambda x: 1 if type(x) == str and '駐車場' in x else 0)
    features_df['purpose_contain_factory'] = df['用途'].apply(lambda x: 1 if type(x) == str and '工場' in x else 0)
    features_df['purpose_contain_office'] = df['用途'].apply(lambda x: 1 if type(x) == str and '事務所' in x else 0)
    features_df['purpose_contain_workshop'] = df['用途'].apply(lambda x: 1 if type(x) == str and '作業場' in x else 0)
    features_df['purpose_contain_warehouse'] = df['用途'].apply(lambda x: 1 if type(x) == str and '倉庫' in x else 0)
    features_df['purpose_contain_else'] = df['用途'].apply(lambda x: 1 if type(x) == str and 'その他' in x else 0)
    features_df['feature_purpose'] = label_encode(df['今後の利用目的'])
    features_df['direction'] = label_encode(df['前面道路：方位'])
    features_df['load_type'] = label_encode(df['前面道路：種類'])
    features_df['load_width'] = df['前面道路：幅員（ｍ）']
    features_df['city_planning'] = label_encode(df['都市計画'])
    features_df['building_to_land_ratio'] = df['建ぺい率（％）']
    features_df['capacity_ratio'] = df['容積率（％）']
    features_df['trade_year'] = df['取引時点'].apply(lambda x: int(x[:4]))
    features_df['trade_quater'] = df['取引時点'].apply(lambda x: int(x[:4]) + int(x[6]) / 10)
    features_df['renovation'] = label_encode(df['改装'])
    features_df['circumstances'] = label_encode(df['取引の事情等'])

    features_df['null_count'] = df.T.isnull().sum().values

    features_df['trade_year_diff_build_year'] = features_df['trade_year'] - features_df['build_year']
    features_df['trade_quater_diff_build_year'] = features_df['trade_quater'] - features_df['build_year']

    features_df['building_to_land_ratio_div_capacity_ratio'] = features_df['building_to_land_ratio'] / (features_df['capacity_ratio'] + 0.01)
    features_df['building_to_land_ratio_diff_capacity_ratio'] = features_df['building_to_land_ratio'] - features_df['capacity_ratio']

    features_df['area_div_all_area'] = features_df['area'] / (features_df['all_area'] + 0.01)
    features_df['area_diff_all_area'] = features_df['area'] - features_df['all_area']

    return features_df


def main():
    train_df = pd.read_feather('../data/input/train_data.feather')
    test_df = pd.read_feather('../data/input/test_data.feather')

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_features_df = get_features(whole_df)

    train_features_df = whole_features_df.iloc[:len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df):]

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()
