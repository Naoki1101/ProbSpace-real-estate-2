import pandas as pd


def main():
    train_df = pd.read_feather('../data/input/train_data.feather')
    train_df[['y']].to_feather('../features/y.feather')


if __name__ == '__main__':
    main()
