import numpy as np
import pandas as pd


def main():
    train_df = pd.read_feather('../data/input/train_data.feather')
    oof = np.load('../logs/lgbm_014_20200810213754_0.271/oof.npy')

    train_df['residual']  = train_df['y'] - oof
    train_df[['residual']].to_feather('../features/residual.feather')


if __name__ == '__main__':
    main()
