import os
import numpy as np
from numpy.testing import assert_array_equal
import pyilastik

path = os.path.join(os.path.dirname(__file__), 'data')


def test_ilastik_version05():
    print(path)
    p = os.path.join(path, 'ilastik-0.5.ilp')
    for fname, (img, labels, prediction) in \
            pyilastik.read_project(p, image_path=path):
        np.testing.assert_equal(fname, 'ilastik-test-2-4-8.tif')
        np.testing.assert_array_equal(img.shape, (1, 1, 4, 2, 1))
        np.testing.assert_array_equal(img.shape, labels.shape)
        assert prediction is None


def test_ilastik_storage_version01():
    p = os.path.join(path, 'ilastik-1.2.ilp')
    for fname, (img, labels, prediction) in \
            pyilastik.read_project(p, skip_image=True):
        np.testing.assert_equal(fname, 'ilastik-test-2-4-8.tif')
        # np.testing.assert_array_equal(img.shape, (8, 4, 2, 1))
        # np.testing.assert_array_equal(img.shape, labels.shape)
        assert prediction is None


def test_2channels_1z_2classes():
    localpath = os.path.join(path, '2channels_1z_2classes')
    p = os.path.join(localpath, 'ilastik-1.2.2post1_mac.ilp')
    img_path = os.path.join(localpath, 'images')
    ilp = pyilastik.read_project(p, image_path=img_path, skip_image=True)
    (_, (_, labels, _)) = ilp['57x54_2channels_1z_2classes.tif']
    print(labels.shape)
    assert set(np.unique(labels[0, :, :4, 0])) == {1.}
    assert set(np.unique(labels[0, :, 4:, 0])) == {2.}


def test_1channel_1z_2classes():
    localpath = os.path.join(path, '1channel_1z_2classes')
    p = os.path.join(localpath, 'ilastik_1.2.2post1_mac.ilp')
    img_path = os.path.join(localpath, 'images')
    ilp = pyilastik.read_project(p, image_path=img_path, skip_image=True)
    (_, (_, labels, _)) = ilp['57x54_1channel_1z_2classes.tif']

    assert set(np.unique(labels[0, :18, :, 0][:])) == {2.}
    assert set(np.unique(labels[0, 18:, :21, 0][:])) == {1.}
    assert set(np.unique(labels[0, 18:, 21:, 0][:])) == {0.}


def test_shape_of_labelmatrix():
    localpath = os.path.join(path, '1channel_1z_2classes')
    p = os.path.join(localpath, 'ilastik_1.2.2post1_mac.ilp')
    ilp = pyilastik.read_project(p, skip_image=True)
    mat_shape = ilp.shape_of_labelmatrix(0)
    print(mat_shape)
    assert_array_equal(mat_shape, np.array((54, 57, 1)))

    localpath = os.path.join(path, '2channels_1z_2classes')
    p = os.path.join(localpath, 'ilastik-1.2.2post1_mac.ilp')
    ilp = pyilastik.read_project(p, skip_image=True)
    mat_shape = ilp.shape_of_labelmatrix(0)
    assert_array_equal(mat_shape, np.array((54, 57, 1)))

    p = os.path.join(path, 'ilastik-1.2.ilp')
    ilp = pyilastik.read_project(p, skip_image=True)
    mat_shape = ilp.shape_of_labelmatrix(0)
    assert_array_equal(mat_shape, np.array((8, 4, 2, 1)))


def test_ilastik0122_mac():

    localpath = os.path.join(path, 'purkinjetest')
    p = os.path.join(localpath, 'ilastik-1.2.2post1mac.ilp')
    img_path = os.path.join(localpath, 'images')
    ilp = pyilastik.read_project(p, image_path=img_path, skip_image=True)

    (_, (_, labels, _)) = ilp['images/769_cerebellum_5M41_subset_1.tif']
    assert set(np.unique(labels[:])) == {0, 1, 2, 3, 4}  # all 4 labels

    (_, (_, labels, _)) = ilp['images/758_cerebellum_5M60_subset_3.tif']
    assert set(np.unique(labels[:])) == {0, 1, 2, 3, 4}  # all 4 labels

    (_, (_, labels, _)) = ilp['images/667_cerebellum_5M53_subset_3.tif']
    print(np.unique(labels[:]))
    assert set(np.unique(labels[:])) == set([])  # all no labels

    (_, (_, labels, _)) = ilp['images/628_cerebellum_5M46_subset_1.tif']
    assert set(np.unique(labels[:])) == {0, 2, 3, 4}  # without label 1
