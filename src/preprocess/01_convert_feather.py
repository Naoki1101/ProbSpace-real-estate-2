import os
import pandas as pd
import glob
import zipfile

extension = 'csv'


def zip_extract(path):
    with zipfile.ZipFile(path) as existing_zip:
        existing_zip.extractall('../data/input/')
        os.system(f'rm -rf {path}')


def main():
    zfiles = glob.glob(f'../data/input/*.zip')
    if len(zfiles) >= 1:
        for zfile in zfiles:
            zip_extract(zfile)

    path_list = glob.glob(f'../data/input/*.{extension}')

    for path in path_list:
        (pd.read_csv(path, encoding="utf-8"))\
            .to_feather(path.replace(extension, 'feather'))


if __name__ == '__main__':
    main()
