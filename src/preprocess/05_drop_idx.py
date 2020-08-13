import numpy as np
import pandas as pd


def main():
    train_df = pd.read_feather('../data/input/train_data.feather')
    train_df['estimated_land_price2'] = pd.read_feather('../features/estimated_land_price2_train.feather')['estimated_land_price2']

    large_diff_idx = train_df[train_df['estimated_land_price2'] >= 1e+7][train_df['y'] < 1].index.values
    np.save('../pickle/large_diff.npy', large_diff_idx)

    large_diff_idx2 = train_df[train_df['estimated_land_price2'] >= 1e+7][train_df['y'] < 2].index.values
    np.save('../pickle/large_diff2.npy', large_diff_idx2)


if __name__ == '__main__':
    main()
