from unittest import TestCase
import os
import numpy as np
from numpy.testing import assert_array_equal
import pyilastik
from pyilastik.ilastik_storage_version_01 import imread
data_path = path = os.path.join(os.path.dirname(__file__),
                                'data/dimensionstest')


val_multi_z = np.array(
    [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
     [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])

val_single_z = val_multi_z[0, :, :]

# swapped label values
val_multi_z_swapped = np.zeros(val_multi_z.shape)
val_multi_z_swapped[val_multi_z == 2] = 1
val_multi_z_swapped[val_multi_z == 1] = 2
val_single_z_swapped = val_multi_z_swapped[0, :, :]

class TestImgImport(TestCase):

    def test_imread(self):

        img_path = os.path.join(path, 'x15_y10_z1_c1.tif')
        img = imread(img_path)
        self.assertEqual(img.shape, (1, 10, 15, 1))

        img_path = os.path.join(path, 'x15_y10_z1_c2.tif')
        img = imread(img_path)
        self.assertEqual(img.shape, (1, 10, 15, 2))

        img_path = os.path.join(path, 'x15_y10_z2_c4.tif')
        img = imread(img_path)
        self.assertEqual(img.shape, (2, 10, 15, 4))

        img_path = os.path.join(path, 'x15_y10_z2_c1.tif')
        img = imread(img_path)
        self.assertEqual(img.shape, (2, 10, 15, 1))

        img_path = os.path.join(path, 'x15_y10_z1_rgb.tif')
        img = imread(img_path)
        self.assertEqual(img.shape, (1, 10, 15, 3))

        img_path = os.path.join(path, 'x15_y10_z2_rgb.tif')
        img = imread(img_path)
        self.assertEqual(img.shape, (2, 10, 15, 3))

    def test_single_channel_single_z(self):

        p = os.path.join(path, 'x15_y10_z1_c1_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=False)
        (_, (img, labels, _)) = ilp['x15_y10_z1_c1.tif']

        self.assertEqual(img.shape, (1, 10, 15, 1))
        self.assertEqual(labels.shape, (1, 10, 15, 1))
        assert_array_equal(labels[0, :, :, 0], val_single_z)

    def test_single_channel_multi_z(self):

        p = os.path.join(path, 'x15_y10_z2_c1_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=False)
        (_, (img, labels, _)) = ilp['x15_y10_z2_c1.tif']

        self.assertEqual(img.shape, (2, 10, 15, 1))
        self.assertEqual(labels.shape, (2, 10, 15, 1))
        assert_array_equal(labels[:, :, :, 0], val_multi_z)

    def test_multi_channel_multi_z(self):

        p = os.path.join(path, 'x15_y10_z2_c4_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=False)
        (_, (img, labels, _)) = ilp['x15_y10_z2_c4.tif']

        self.assertEqual(img.shape, (2, 10, 15, 4))
        self.assertEqual(labels.shape, (2, 10, 15, 1))
        assert_array_equal(labels[:, :, :, 0], val_multi_z)

    def test_rgb_single_z(self):

        p = os.path.join(path, 'x15_y10_z1_rgb_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=False)
        (_, (img, labels, _)) = ilp['x15_y10_z1_rgb.tif']

        self.assertEqual(img.shape, (1, 10, 15, 3))
        self.assertEqual(labels.shape, (1, 10, 15, 1))
        assert_array_equal(labels[0, :, :, 0], val_single_z_swapped)

    def test_rgb_multi_z(self):

        p = os.path.join(path, 'x15_y10_z2_rgb_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=False)
        (_, (img, labels, _)) = ilp['x15_y10_z2_rgb.tif']

        self.assertEqual(img.shape, (2, 10, 15, 3))
        self.assertEqual(labels.shape, (2, 10, 15, 1))
        assert_array_equal(labels[:, :, :, 0], val_multi_z)

    def test_large_dat(self):

        localpath = os.path.join(path, '../purkinjetest')
        p = os.path.join(localpath, 'ilastik-1.2.2post1mac.ilp')
        ilp = pyilastik.read_project(p, skip_image=False,
                                     image_path=os.path.join(localpath,
                                                             'images'))
        (_, (img, labels, _)) = ilp['images/769_cerebellum_5M41_subset_1.tif']

        self.assertEqual(img.shape, (1, 684, 1047, 3))
        self.assertEqual(labels.shape, (1, 684, 1047, 1))

        val = np.array([[2., 2., 2., 2.],
                        [0., 2., 2., 2.],
                        [0., 0., 2., 2.],
                        [0., 0., 4., 4.],
                        [4., 4., 4., 4.]])

        assert_array_equal(val, labels[0, 212:217, 309:313, 0])
