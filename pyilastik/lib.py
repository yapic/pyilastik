import os
import re
import itertools

import h5py
import numpy as np
import skimage.io

class UnknownIlastikVersion(Exception):
    pass

# image shape: (?,?,H,W,C), e.g. (1, 1, 2098, 2611, 3)
# labels shape: (?,?,H,W,1), e.g. (1, 1, 2098, 2611, 1), 0 == unlabeled
# prediction shape: (?,?,H,W,L), e.g. (1, 1, 2098, 2611, 3)
class read_project(object):
    def __init__(self, ilastik_file, image_path=None, prediction=False):
        self.prediction = prediction
        f = h5py.File(ilastik_file, 'r')

        self.version = f.get('/PixelClassification/StorageVersion')
        if self.version is not None:
            self.version = self.version[()].decode()

            if self.version != '0.1':
                self.version = None

        if self.version is None:
            self.version = f.get('ilastikVersion')[()] + '_ilastik'
            if self.version != 0.5:
                raise UnknownIlastikVersion('Unknown ilastik version v{}. I only understand ilastik v0.5 and ilastik storage version v0.1 (ilastik v1.2.0)!'.format(format))

        self.f = f
        self.image_path = image_path


    def __iter__(self):
        '''
        Returns `filename, (img, labels, prediction)`
        prediction is None if no prediction was made
        '''
        print('ilasik version', self.version)
        if self.version == '0.5_ilastik':
            return self.iter_v05_ilastik()

        if self.version == '0.1':
            return self.iter_v01()


    def iter_v01(self, skip_image=False):
        f = self.f
        assert f.get('/Input Data/StorageVersion')[()].decode() == '0.2'

        for dset_name, dset in f.get('/PixelClassification/LabelSets').items():
            i = re.search('[0-9]+$', dset_name).group(0)
            lane = 'lane0' + i

            path = f.get('/Input Data/infos/{lane}/Raw Data/filePath'.format(lane=lane))
            path = path[()].decode()

            if self.image_path is not None:
                _, fname = os.path.split(path)
                path = os.path.join(self.image_path, fname)
            img = None if skip_image else skimage.io.imread(path)
            prediction = None # TODO

            if skip_image:
                # 1st get the (approximate) labeled image size
                slice_list = []
                for block in f.get('/PixelClassification/LabelSets/{}'.format(dset_name)).values():
                    slices = re.findall('([0-9]+):([0-9]+)', block.attrs['blockSlice'].decode('ascii'))
                    slice_list.append(slices)

                slice_list = np.array(slice_list)
                X = np.amax(slice_list[:,0,1])
                Y = np.amax(slice_list[:,1,1])
                Z = np.amax(slice_list[:,2,1])
                C = np.amax(slice_list[:,3,1])

                labels = np.zeros([X, Y, Z, C])
            else:
                labels = np.zeros_like(img)
                while labels.ndim < 4:
                    labels = labels[..., np.newaxis]

            for block in f.get('/PixelClassification/LabelSets/{}'.format(dset_name)).values():
                slices = re.findall('([0-9]+):([0-9]+)', block.attrs['blockSlice'].decode('ascii'))
                slices = [slice(int(start), int(end)) for start, end in slices]
                if len(slices) == 4:
                    labels[slices[0], slices[1], slices[2], slices[3]] = block[()]
                elif len(slices) == 3:
                    labels[slices[0], slices[1], slices[2], 0] = block[()]
                elif len(slices) == 2:
                    labels[slices[0], slices[1], 0, 0] = block[()]
                else:
                    assert NotImplemented

            yield path, (img, labels, None)


    def iter_v05_ilastik(self):
        f = self.f
        for dset_name, dset in f.get('/DataSets').items():
            path = dset.attrs['fileName']
            img = np.array(dset.get('data'))
            labels = np.array(dset.get('labels/data'))

            if self.prediction:
                prediction = np.array(dset.get('prediction'))
                yield path, (img, labels, prediction)
            else:
                yield path, (img, labels, None)


    def __len__(self):
        if self.version == '0.5_ilastik':
            return len(self.f.get('DataSets'))
