import os
import numpy as np

import pyilastik

path = os.path.join(os.path.dirname(__file__), 'data')

def test_ilastik_version05():
    print(path)
    p = os.path.join(path, 'ilastik-0.5.ilp')
    for fname, (img, labels, prediction) in pyilastik.read_project(p, image_path=path):
        np.testing.assert_equal(fname, 'ilastik-test-2-4-8.tif')
        np.testing.assert_array_equal(img.shape, (1,1,4,2,1))
        np.testing.assert_array_equal(img.shape, labels.shape)
        assert prediction is None


def test_ilastik_storage_version01():
    p = os.path.join(path, 'ilastik-1.2.ilp')
    for fname, (img, labels, prediction) in pyilastik.read_project(p, image_path=path):
        np.testing.assert_equal(fname, 'ilastik-test-2-4-8.tif')
        np.testing.assert_array_equal(img.shape, (8,4,2,1))
        np.testing.assert_array_equal(img.shape, labels.shape)
        assert prediction is None

