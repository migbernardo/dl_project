import os
import shutil
from urllib import request
from zipfile import ZipFile
from path import get_path

src_path = os.path.abspath(os.curdir)
main_path = get_path()
url = 'https://madm.dfki.de/files/sentinel/EuroSAT.zip'

# download zip file from url
request.urlretrieve(url, 'EuroSAT.zip')
# extract zip file inside path
with ZipFile(os.path.join(src_path, 'EuroSAT.zip')) as f:
    f.extractall(src_path)
# delete zip file after extraction
os.remove(os.path.join(src_path, 'EuroSAT.zip'))
# rename folder to raw
os.rename(os.path.join(src_path, '2750'), os.path.join(src_path, 'raw'))
# make data dir
os.mkdir(os.path.join(main_path, 'data'))
data_path = os.path.join(main_path, 'data')
# move folder to destination
shutil.move(os.path.join(src_path, 'raw'), os.path.join(data_path, 'raw'))