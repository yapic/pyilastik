from unittest import TestCase
import os
import numpy as np
from numpy.testing import assert_array_equal
import pyilastik

data_path = path = os.path.join(os.path.dirname(__file__), 'data')


val_multi_z = np.array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 2., 2., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 2., 2.]],
                        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])

val_single_z = val_multi_z[0, :, :]

# swapped label values
val_multi_z_swapped = np.zeros(val_multi_z.shape)
val_multi_z_swapped[val_multi_z == 2] = 1
val_multi_z_swapped[val_multi_z == 1] = 2

val_single_z_swapped = val_multi_z_swapped[0, :, :]


class TestLabelImportDimensions(TestCase):

    def test_single_channel_single_z(self):

        p = os.path.join(path, 'dimensionstest/x15_y10_z1_c1_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)
        (_, (_, labels, _)) = ilp['x15_y10_z1_c1.tif']

        self.assertEqual(labels.shape, (1, 8, 10, 1))
        assert_array_equal(labels[0, :, :, 0], val_single_z)

    def test_single_channel_multi_z(self):

        p = os.path.join(path, 'dimensionstest/x15_y10_z2_c1_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)
        (_, (_, labels, _)) = ilp['x15_y10_z2_c1.tif']

        self.assertEqual(labels.shape, (2, 8, 10, 1))
        assert_array_equal(labels[:, :, :, 0], val_multi_z)

    def test_single_channel_multi_z_2(self):

        p = os.path.join(path, 'ilastik-1.2.ilp')
        # 2x 4y 8z
        ilp = pyilastik.read_project(p, skip_image=True)
        (_, (_, labels, _)) = ilp['ilastik-test-2-4-8.tif']

        self.assertEqual(labels.shape, (8, 4, 2, 1))

    def test_multi_channel_multi_z(self):

        p = os.path.join(path, 'dimensionstest/x15_y10_z2_c4_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)
        (_, (_, labels, _)) = ilp['x15_y10_z2_c4.tif']

        self.assertEqual(labels.shape, (2, 8, 10, 1))
        assert_array_equal(labels[:, :, :, 0], val_multi_z)

    def test_multi_channel_single_z(self):

        p = os.path.join(path, 'dimensionstest/x15_y10_z1_c2_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)
        (_, (_, labels, _)) = ilp['x15_y10_z1_c2.tif']

        self.assertEqual(labels.shape, (1, 8, 10, 1))
        assert_array_equal(labels[0, :, :, 0], val_single_z)

    def test_rgb_single_z(self):

        p = os.path.join(path, 'dimensionstest/x15_y10_z1_rgb_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)
        (_, (_, labels, _)) = ilp['x15_y10_z1_rgb.tif']

        self.assertEqual(labels.shape, (1, 8, 10, 1))
        assert_array_equal(labels[0, :, :, 0], val_single_z_swapped)

    def test_rgb_mutli_z(self):

        p = os.path.join(path, 'dimensionstest/x15_y10_z2_rgb_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)
        (_, (_, labels, _)) = ilp['x15_y10_z2_rgb.tif']

        self.assertEqual(labels.shape, (2, 8, 10, 1))
        assert_array_equal(labels[:, :, :, 0], val_multi_z)

    def test_shape_of_labelmatrix_large_dat(self):

        localpath = os.path.join(path, 'purkinjetest')
        p = os.path.join(localpath, 'ilastik-1.2.2post1mac.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)

        # 758_cerebellum_5M60_subset_3.tif
        mat_shape = ilp.shape_of_labelmatrix(1)
        assert_array_equal(mat_shape, np.array((684, 861, 1)))

    def test_labels_large_dat(self):

        localpath = os.path.join(path, 'purkinjetest')
        p = os.path.join(localpath, 'ilastik-1.2.2post1mac.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)
        (_, (_, labels, _)) = ilp['images/769_cerebellum_5M41_subset_1.tif']

        val = np.array([[2., 2., 2., 2.],
                        [0., 2., 2., 2.],
                        [0., 0., 2., 2.],
                        [0., 0., 4., 4.],
                        [4., 4., 4., 4.]])

        assert_array_equal(val, labels[0, 212:217, 309:313, 0])
