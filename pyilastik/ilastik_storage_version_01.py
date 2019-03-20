import os
import re
import functools
import warnings
import numpy as np
from bigtiff import Tiff
import pyilastik.utils as utils
from functools import lru_cache

# image shape: (?,?,H,W,C), e.g. (1, 1, 2098, 2611, 3)
# labels shape: (?,?,H,W,1), e.g. (1, 1, 2098, 2611, 1), 0 == unlabeled
# prediction shape: (?,?,H,W,L), e.g. (1, 1, 2098, 2611, 3)


def imread(path):
    '''
    reads tiff image in dimension order zyxc
    '''

    slices = Tiff.memmap_tcz(path)

    img = []
    for C in range(slices.shape[1]):
        img.append(np.stack([s for s in slices[0, C, :]]))
    img = np.stack(img)
    img = np.moveaxis(img, (0, 1, 2, 3), (3, 0, 1, 2))
    return img


def is_overlap(tile_pos, block_pos):
    '''
    checks if plock position overlaps tile position
    '''

    return np.array([block_pos[:, 0] <= tile_pos[:, 1]-1,
                     block_pos[:, 1]-1 >= tile_pos[:, 0]]).all()


def tile_loc_from_slices(slice_list):
    '''
    get optimal tile position and shape for list of slices
    '''
    slice_list = np.array(slice_list)
    assert slice_list.ndim == 3

    lower = np.min(slice_list[:, :, 0], axis=0)
    upper = np.max(slice_list[:, :, 1], axis=0)

    shape = upper - lower
    pos = lower

    return pos, shape


def ndarray2slices(ndarray):

    return [slice(int(start), int(end)) for start, end in ndarray]


def _get_slices_for(p_pos, q_pos, p_shape, q_shape):
    q_lower = p_pos - q_pos
    q_upper = q_lower + p_shape
    q_slice = np.array([q_lower, q_upper]).transpose()
    q_slice[q_slice < 0] = 0
    q_shape_lookup = np.tile(q_shape, (2, 1)).transpose()
    q_slice[q_slice > q_shape_lookup] = q_shape_lookup[q_slice >
                                                       q_shape_lookup]

    return [slice(start, stop) for start, stop in q_slice]


class IlastikStorageVersion01(object):

    def __init__(self, h5_handle, image_path=None, prediction=False,
                 skip_image=False):
        self.prediction = prediction
        self.f = h5_handle
        self.image_path = image_path
        self.skip_image = skip_image

        try:
            version = self.f.get('/Input Data/StorageVersion')[()].decode()
        except AttributeError:
            # for ilastik release > 1.3.0
            version = self.f.get('/Input Data/StorageVersion')[()]
        assert version == '0.2'

    def __iter__(self):
        '''
        Returns `filename, (img, labels, prediction)`

        prediction is None if no prediction was made

        img is a 4D matrix of pixel values in order (Z, Y, X, C) or None

        labels is a 4D matrix in order (Z, Y, X, C) where C size of C dimension
        is always 1 (only one label channel implemented, i.e.
        currently no overlapping label regions supported)
        The shape of labels is equal or smaller to the shape of img
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

        prediction = None  # TODO

        # 1st get the (approximate) labeled image size
        shape = self.shape_of_labelmatrix(i)
        n_dims = len(shape)

        tile_slice = np.array([[0, s] for s in shape])

        labels = self.tile(i, tile_slice)

        msg = 'dimensions of labelmatrix should be 4 (zyxc) or 3 (yxc)'
        assert n_dims in [3, 4], msg

        if n_dims == 3:
            # add z dimension if missing
            labels = np.expand_dims(labels, axis=0)

        if self.skip_image:
            return original_path, (None, labels, prediction)

        fname = utils.basename(path)
        if self.image_path is not None:
            path = os.path.join(self.image_path, fname)

        ilp_path, _ = os.path.split(f.filename)
        if os.path.isfile(path):
            path = path
        elif os.path.isfile(os.path.join(ilp_path, fname)):
            path = os.path.join(ilp_path, fname)
            warnings.warn(
                'Loading file from ilp file path {}'.format(path))
        else:
            warnings.warn(
                '!!! File {} not found. Skipping pixelimage...'.format(path))
            return original_path, (None, labels, prediction)

        img = imread(path)

        # pad labelmatrix to same size as image
        padding = [(0, d2-d1) for d2, d1 in zip(img.shape, labels.shape)]
        padding[-1] = (0, 0)  # set padding for channel axis to 0
        labels = np.pad(labels, padding, mode='constant',
                        constant_values=0)

        return original_path, (img, labels, prediction)

    def n_dims(self, item_index):
        '''
        get nr of dimensions
        3 for yxc
        4 for zyxc
        0 if no labels available
        '''
        slices = self._get_block_slices(item_index)

        if slices.size > 0:
            return slices.shape[1]
        else:
            return 0

    def shape_of_labelmatrix(self, item_index):
        '''
        Label matrix shape is retrieved from label data.

        Order is (Z, Y, X, C) or (Y, X, C) where C size of C dimension
        is always 1 (only one label channel implemented, i.e.
        currently no overlapping label regions supported)

        Label matrix shape does not always equal corresponding image shape.
        Label matrix shape is always smaller or equal to corresponding
        image shape.

        If no labels exist, label matrix shape is 0 in all dimensions.
        '''
        slice_list = self._get_block_slices(item_index)

        if len(slice_list) == 0:
            msg = 'No labels found in Ilastik file - ' +\
                  'cannot approximate image size'
            warnings.warn(msg)

            labelmat_shape = np.zeros((4,), dtype='int')

        else:
            labelmat_shape = np.amax(slice_list[:, :, 1], axis=0)
        return labelmat_shape

    @lru_cache(maxsize=None)
    def _get_blocks(self, item_index):
        dset_name = 'labels{:03}'.format(item_index)
        labelset_str = '/PixelClassification/LabelSets/{}'.format(dset_name)
        return self.f.get(labelset_str).values()

    def _blocks_in_tile(self, item_index, tile_slice):
        '''
        Checks which blocks overlap with the given tile slice.
        Returns a list of booleans (one for each block).
        '''
        block_slices = self._get_block_slices(item_index)

        # We do not check for overlap in the channel dimension (the last
        # dimension). The channel dimension is a dummy dimension with slice
        # values being always [0 1].
        return [is_overlap(tile_slice[:-1, :], block_slice[:-1, :])
                for block_slice in block_slices]

    def tile_for_selected_blocks(self, item_index, block_selection):
        '''
        returns a labelmatrix tile and a corresponding position
        '''
        slices = self._get_block_slices(item_index).copy()
        block_indices = np.nonzero(block_selection)[0]
        pos, shape = tile_loc_from_slices(slices[block_indices])
        labels = np.zeros(shape)

        for i in block_indices:

            s_shifted = slices[i]
            s_shifted[:, 0] = s_shifted[:, 0] - pos
            s_shifted[:, 1] = s_shifted[:, 1] - pos
            s_fmt = ndarray2slices(s_shifted)
            assert (s_shifted >= 0).all()

            labels[s_fmt] = self.load_block_data(item_index, i)

        return pos, labels

    @lru_cache(maxsize=None)
    def load_block_data(self, item_index, block_index):
        blocks = self._get_blocks(item_index)

        for counter, block in enumerate(blocks):
            if counter == block_index:
                return block[()]

    def tile(self, item_index, tile_slice):
        '''
        Order is (Z, Y, X, C) or (Y, X, C) where C size of C dimension
        is always 1 (only one label channel implemented, i.e.
        currently no overlapping label regions supported)
        '''

        sel = self._blocks_in_tile(item_index, tile_slice)

        pos_q = tile_slice[:, 0]
        shape_q = tile_slice[:, 1] - tile_slice[:, 0]
        labels_q = np.zeros(shape_q)

        # return empty label matrix if no blocks in tile
        if not sel:
            return labels_q
        if not np.any(sel):
            return labels_q

        pos_p, labels_p = self.tile_for_selected_blocks(item_index, sel)
        shape_p = labels_p.shape

        q_slice = _get_slices_for(pos_p, pos_q, shape_p, shape_q)
        p_slice = _get_slices_for(pos_q, pos_p, shape_q, shape_p)

        labels_q[q_slice] = labels_p[p_slice]

        return labels_q

    @lru_cache(maxsize=None)
    def _get_block_slices(self, item_index):
        slice_list = []
        for block in self._get_blocks(item_index):
            slices = re.findall('([0-9]+):([0-9]+)',
                                block.attrs['blockSlice'].decode('ascii'))
            slice_list.append(slices)

        return np.array(slice_list).astype('int')

    def __len__(self):
        return len(self.f.get('/PixelClassification/LabelSets').keys())
