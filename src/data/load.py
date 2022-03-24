import os
import shutil
from urllib import request
from zipfile import ZipFile


if __name__ == '__main__':
    # change current dir to project's main dir
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.abspath(os.pardir))
    url = 'https://madm.dfki.de/files/sentinel/EuroSAT.zip'

    # download zip file from url
    request.urlretrieve(url, 'EuroSAT.zip')
    # extract zip file to current dir
    with ZipFile('EuroSAT.zip') as f:
        f.extractall(os.path.abspath(os.curdir))
    # delete zip file after extraction
    os.remove('EuroSAT.zip')
    # rename folder to raw
    os.rename('2750', 'raw')
    # make data dir inside current dir
    os.mkdir('data')
    # move folder to data dir
    shutil.move('raw', os.path.join('data', 'raw'))
