import numpy as np
import pandas as pd


def main():
    test_df = pd.read_feather('../data/input/test_data.feather')
    sample_df = pd.DataFrame({
        'id': test_df['id'],
        'y': np.zeros(len(test_df))
        })

    sample_df.to_feather('../data/input/sample_submission.feather')


if __name__ == '__main__':
    main()
