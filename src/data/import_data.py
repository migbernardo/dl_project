import os
import shutil
from urllib import request
from zipfile import ZipFile


# returns the absolute path to one of the folders in the project's main dir
def get_path(folder_name):
    # get current dir abs path
    cur_dir = os.path.abspath(os.curdir)
    # go back to parent dir
    cur_dir2 = os.path.abspath(os.path.join(cur_dir, os.pardir))
    # go back to subsequent parent dir
    cur_dir3 = os.path.abspath(os.path.join(cur_dir2, os.pardir))
    # go to target dir
    target_dir = os.path.join(cur_dir3, folder_name)
    return target_dir


src_path = os.path.abspath(os.curdir)
dst_path = get_path('data')
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
# move folder to destination
shutil.move(os.path.join(src_path, 'raw'), os.path.join(dst_path, 'raw'))
