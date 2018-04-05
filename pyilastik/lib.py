import os
import re
import itertools

import h5py
import numpy as np

from pyilastik.ilastik_version_05 import IlastikVersion05
from pyilastik.ilastik_storage_version_01 import IlastikStorageVersion01 # for ilastik 1.2

class UnknownIlastikVersion(Exception):
    pass

# image shape: (?,?,H,W,C), e.g. (1, 1, 2098, 2611, 3)
# labels shape: (?,?,H,W,1), e.g. (1, 1, 2098, 2611, 1), 0 == unlabeled
# prediction shape: (?,?,H,W,L), e.g. (1, 1, 2098, 2611, 3)
def read_project(ilastik_file, image_path=None, prediction=False, skip_image=False):
    f = h5py.File(ilastik_file, 'r')

    version = f.get('/PixelClassification/StorageVersion')
    if version is not None:
        version = version[()].decode()

        # newer ilastik version 1.2 (storage version 0.1)
        if version == '0.1':
            return IlastikStorageVersion01(f, image_path, prediction, skip_image)

    if version is None:
        # old ilastik version 0.5
        version = f.get('ilastikVersion')[()]
        if version == 0.5:
            return IlastikVersion05(f, image_path, prediction, skip_image)

    raise UnknownIlastikVersion('Unknown ilastik version v{}. I only understand ilastik v0.5 and ilastik storage version v0.1 (ilastik v1.2.0)!'.format(format))
