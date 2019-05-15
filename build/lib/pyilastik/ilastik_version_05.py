import re
import pyilastik.utils as utils
import numpy as np

# image shape: (?,?,H,W,C), e.g. (1, 1, 2098, 2611, 3)
# labels shape: (?,?,H,W,1), e.g. (1, 1, 2098, 2611, 1), 0 == unlabeled
# prediction shape: (?,?,H,W,L), e.g. (1, 1, 2098, 2611, 3)
class IlastikVersion05(object):
    def __init__(self, h5_handle, image_path=None, prediction=False, skip_image=False):
        self.prediction = prediction
        self.f = h5_handle
        self.image_path = image_path
        self.skip_image = skip_image


    def __getitem__(self, i):
        '''
        Returns `filename, (img, labels, prediction)` for the i'th image
        prediction is None if no prediction was made
        '''
        f = self.f
        dset = f.get('/DataSets/dataItem{:02}'.format(i))

        path = utils.basename(dset.attrs['fileName'].decode('ascii'))
        img = np.array(dset.get('data')) if not self.skip_image else None
        labels = np.array(dset.get('labels/data'))

        if self.prediction:
            prediction = np.array(dset.get('prediction'))
            return path, (img, labels, prediction)
        else:
            return path, (img, labels, None)


    def __iter__(self):
        '''
        Returns `filename, (img, labels, prediction)`
        prediction is None if no prediction was made
        '''
        f = self.f
        for dset_name in f.get('/DataSets').keys():
            i = re.search('[0-9]+$', dset_name).group(0)
            yield self[int(i)]


    def __len__(self):
        return len(self.f.get('DataSets'))
