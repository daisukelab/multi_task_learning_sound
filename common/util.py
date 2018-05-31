import warnings
warnings.simplefilter('ignore')
import os
import re
from glob import glob
import numpy as np
np.warnings.filterwarnings('ignore')

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def list_files(pattern):
    _files = glob(pattern)
    return _files

def load_txt_list(txtfile):
    with open(txtfile, 'r') as f:
        return f.read().splitlines()
    return None

def save_txt_list(txtfile):
    with open(txtfile, 'w') as f:
        f.write('\n'.join(trainlist)+'\n')

from itertools import chain
def flatten_list(lists):
    return list(chain.from_iterable(lists))

def unify_2d_length(data, dest_length):
    d_len = data.shape[1]
    if d_len < dest_length:
        L = abs(d_len - dest_length)
        unified  = np.pad(data, ((0, 0), (0, L)), 'symmetric')
    elif dest_length < d_len:
        unified = data[:, :dest_length]
    else:
        unified = data
    return unified

def random_unify_3d_mels(mels, length):
    l_mels = mels.shape[1]
    if l_mels < length:
        l = abs(l_mels - length)
        start = np.random.choice(l)
        mels  = np.pad(mels, ((0, 0), (start, l-start), (0, 0)), 'symmetric')
    elif l_mels > length:
        l = abs(l_mels - length)
        start = np.random.choice(l)
        mels  = mels[:, start: start + length, :]
    return mels

from scipy import stats
def get_2d_mode_length(mels_set):
    _length = stats.mode([x.shape[1] for x in mels_set])[0][0] # mode value
    return _length

def all_elements_are_identical(iterator):
    # https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)
