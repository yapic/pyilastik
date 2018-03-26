import os
import re
import functools
import warnings
import numpy as np
import skimage.io

import pyilastik.utils as utils

# image shape: (?,?,H,W,C), e.g. (1, 1, 2098, 2611, 3)
# labels shape: (?,?,H,W,1), e.g. (1, 1, 2098, 2611, 1), 0 == unlabeled
# prediction shape: (?,?,H,W,L), e.g. (1, 1, 2098, 2611, 3)


class IlastikStorageVersion01(object):

    def __init__(self, h5_handle, image_path=None, prediction=False,
                 skip_image=False):
        self.prediction = prediction
        self.f = h5_handle
        self.image_path = image_path
        self.skip_image = skip_image

        assert self.f.get('/Input Data/StorageVersion')[()].decode() == '0.2'

    def __iter__(self):
        '''
        Returns `filename, (img, labels, prediction)`
        prediction is None if no prediction was made
        '''

        for dset_name in self.f.get('/PixelClassification/LabelSets').keys():
            i = re.search('[0-9]+$', dset_name).group(0)

            res = self[int(i)]
            if res is not None:
                yield res

    @functools.lru_cache(1)
    def image_path_list(self):
        '''
        Returns the list of path to the image files on the original file
        system.
        '''
        path_list = []
        for dset_name in self.f.get('/PixelClassification/LabelSets').keys():
            i = re.search('[0-9]+$', dset_name).group(0)
            i = int(i)

            lane = 'lane{:04}'.format(i)
            dset_name = 'labels{:03}'.format(i)

            path = self.f.get(
                '/Input Data/infos/{lane}/Raw Data/filePath'.format(lane=lane))
            path = path[()].decode()

            path_list.append(path)

        return path_list

    def __getitem__(self, i):
        '''
        Returns `filename, (img, labels, prediction)` for the i'th image
        prediction is None if no prediction was made
        '''
        if type(i) == str:
            idx = self.image_path_list().index(i)
            return self[idx]

        f = self.f

        lane = 'lane{:04}'.format(i)

        path = f.get(
            '/Input Data/infos/{lane}/Raw Data/filePath'.format(lane=lane))
        path = path[()].decode()
        original_path = path

        fname = utils.basename(path)
        if self.image_path is not None:
            path = os.path.join(self.image_path, fname)

        ilp_path, _ = os.path.split(f.filename)

        if self.skip_image:
            img = None
        else:
            if os.path.isfile(path):
                path = path
            elif os.path.isfile(os.path.join(ilp_path, fname)):
                path = os.path.join(ilp_path, fname)
                warnings.warn(
                    'Loading file from ilp file path {}'.format(path))
            else:
                warnings.warn(
                    '!!! File {} not found. Skipping...'.format(path))
                return None

            img = skimage.io.imread(path)

        prediction = None # TODO

        if self.skip_image:
            # 1st get the (approximate) labeled image size
            labels = self._init_labelmat_from_labels(i)

        else:
            labels = np.zeros_like(img)

        while img is not None and img.ndim < 4:
            img = img[..., np.newaxis]
        while labels.ndim < 4:
            labels = labels[..., np.newaxis]

        for block in self._get_blocks(i):
            slices = re.findall('([0-9]+):([0-9]+)',
                                block.attrs['blockSlice'].decode('ascii'))
            slices = [slice(int(start), int(end)) for start, end in slices]
            if len(slices) == 4:
                labels[slices[0], slices[1], slices[2], slices[3]] = block[()]
            elif len(slices) == 3:
                labels[slices[0], slices[1], slices[2], 0] = block[()]
            elif len(slices) == 2:
                labels[slices[0], slices[1], 0, 0] = block[()]
            else:
                raise NotImplemented

        return original_path, (img, labels, None)

    def _init_labelmat_from_labels(self, item_index):
        '''
        image dimensions are retrieved from label data
        empty 4D label matrix of retrieved dimensions (xyzc)is returned
        '''
        slice_list = []

        for block in self._get_blocks(item_index):
            print(block.attrs['blockSlice'].decode('ascii'))
            slices = re.findall('([0-9]+):([0-9]+)',
                                block.attrs['blockSlice'].decode('ascii'))
            slice_list.append(slices)

        if len(slice_list) == 0:
            msg = 'No labels found in Ilastik file - ' +\
                  'cannot approximate image size'
            warnings.warn(msg)
            labels = np.zeros([0, 0, 0, 0])

        else:
            slice_list = np.array(slice_list).astype('int')
            n_regions, n_dims, _ = slice_list.shape

            # if z dimension is missing (if images are 2d), add z dimension of
            # size 1
            if n_dims != 4:
                channel_slice = np.zeros((n_regions, 1, 2), dtype='int')
                channel_slice[:, :, 1] = 1
                # add z dimension
                slice_list = np.concatenate((slice_list, channel_slice),
                                            axis=1)

            X = np.amax(slice_list[:, 0, 1])
            Y = np.amax(slice_list[:, 1, 1])
            Z = np.amax(slice_list[:, 2, 1])
            C = np.amax(slice_list[:, 3, 1])

            labels = np.zeros([X, Y, Z, C])

        return labels

    def _get_blocks(self, item_index):
        dset_name = 'labels{:03}'.format(item_index)
        labelset_str = '/PixelClassification/LabelSets/{}'.format(dset_name)
        return self.f.get(labelset_str).values()

    def __len__(self):
        return len(self.f.get('/PixelClassification/LabelSets').keys())
