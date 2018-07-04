import h5py
from pyilastik.ilastik_version_05 import IlastikVersion05
# for ilastik 1.2
from pyilastik.ilastik_storage_version_01 import IlastikStorageVersion01


class UnknownIlastikVersion(Exception):
    pass

# image shape: (?,?,H,W,C), e.g. (1, 1, 2098, 2611, 3)
# labels shape: (?,?,H,W,1), e.g. (1, 1, 2098, 2611, 1), 0 == unlabeled
# prediction shape: (?,?,H,W,L), e.g. (1, 1, 2098, 2611, 3)


def read_project(ilastik_file, image_path=None, prediction=False,
                 skip_image=False):
    f = h5py.File(ilastik_file, 'r')
    version = f.get('/PixelClassification/StorageVersion')
    if version is not None:
        try:
            version = version[()].decode()
        except AttributeError:
            # for ilastik releases >1.3.0
            version = version[()]

        # for ilastik release >1.2 (storage version 0.1)
        if version == '0.1':
            return IlastikStorageVersion01(f, image_path, prediction,
                                           skip_image)

    if version is None:
        # old ilastik version 0.5
        version = f.get('ilastikVersion')[()]
        if version == 0.5:
            return IlastikVersion05(f, image_path, prediction, skip_image)
    msg = ('Unknown ilastik version v{}. I only understand ilastik v0.5 and' +
           'ilastik storage version v0.1 (ilastik v1.2.0)!').format(format)
    raise UnknownIlastikVersion(msg)
