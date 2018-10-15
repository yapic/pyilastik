from unittest import TestCase
import os
import numpy as np
from numpy.testing import assert_array_equal
import pyilastik
import pyilastik.ilastik_storage_version_01 as ils

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

        # print(np.transpose(labels, (3, 0, 2, 1)).astype(int))
        # assert False


    def test_caching_of_block_slices(self):

        p = os.path.join(path, 'dimensionstest/x15_y10_z2_c4_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)

        s = np.array([[0, 2], [0, 10], [0, 15], [0, 1]])

        bs_before = ilp._get_block_slices(0).copy()
        tile = ilp.tile(0, s)
        bs_after = ilp._get_block_slices(0)
        print(bs_before)
        assert_array_equal(bs_before, bs_after)







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

    def test_read_project_for_ilastik_version_1_3_0(self):
        p = os.path.join(path, 'ilastik-1.3.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)
        labels = ilp['ilastik-test-2-4-8.tif'][1][1]

        assert labels.shape == (8, 4, 2, 1)
        assert set(np.unique(labels)) == {0, 1, 2}

    def test_get_block_slices(self):

        localpath = os.path.join(path, 'purkinjetest')
        p = os.path.join(localpath, 'ilastik-1.2.2post1mac.ilp')

        #p = os.path.join(path, 'dimensionstest/x15_y10_z2_rgb_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)

        b = ilp._get_block_slices(0)
        val = np.array([[[253, 267],
                         [318, 484],
                         [0, 1]],
                        [[186, 253],
                         [289, 486],
                         [0, 1]]])
        assert_array_equal(b, val)


    def test_blocks_in_tile(self):


        localpath = os.path.join(path, 'purkinjetest')
        p = os.path.join(localpath, 'ilastik-1.2.2post1mac.ilp')

        tile_slices = np.array([[255, 300],
                                [312, 500],
                                [0, 1]])

        ilp = pyilastik.read_project(p, skip_image=True)
        b = ilp._blocks_in_tile(0, tile_slices)

        self.assertTrue(b[0])
        self.assertFalse(b[1])
        self.assertEqual(len(b), 2)


        p = os.path.join(path, 'dimensionstest/x502_y251_z5_c1_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)

        tile_slices = np.array([[0, 1],
                               [0, 50],
                               [0, 50],
                               [0, 1]])
        assert_array_equal(ilp._blocks_in_tile(0, tile_slices),
                           np.array([True, False, False]))






    def test_tile(self):

        p = os.path.join(path, 'dimensionstest/x502_y251_z5_c1_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)

        tile_slice = np.array([[0, 1],
                               [0, 10],
                               [0, 7],
                               [0, 1]])

        t = ilp.tile(0, tile_slice)

        val_1 = np.array([[0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 1., 1., 0., 0.],
                          [0., 0., 1., 1., 1., 0., 0.],
                          [0., 0., 1., 1., 1., 0., 0.],
                          [0., 0., 1., 1., 1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0.]])

        assert_array_equal(t[0, :, :, 0], val_1)

        tile_slice = np.array([[0, 1],
                               [92, 101],
                               [157, 164],
                               [0, 1]])

        t = ilp.tile(0, tile_slice)

        val_2 = np.array([[0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 1., 1., 1., 0.],
                          [0., 0., 1., 1., 1., 1., 0.],
                          [0., 0., 2., 2., 0., 0., 0.],
                          [0., 0., 2., 2., 0., 0., 0.],
                          [0., 0., 2., 2., 0., 0., 0.],
                          [0., 0., 2., 2., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0.]])

        assert_array_equal(t[0, :, :, 0], val_2)

        tile_slice = np.array([[0, 1],
                               [50, 55],
                               [50, 55],
                               [0, 1]])

        t = ilp.tile(0, tile_slice)

        val_3 = np.array([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])

        assert_array_equal(t[0, :, :, 0], val_3)




    def test_tile_for_selected_blocks(self):

        p = os.path.join(path, 'dimensionstest/x502_y251_z5_c1_classes2.ilp')
        ilp = pyilastik.read_project(p, skip_image=True)

        pos, tile = ilp.tile_for_selected_blocks(0, [True, False, True])

        tile_1 = tile[0, :10, :10, 0]
        tile_2 = tile[0, -10:, -10:, 0]

        self.assertEqual(tile.shape, (2, 96, 161, 1))

        val_1 = np.array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                          [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                          [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                          [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        val_2 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
                          [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
                          [0., 0., 0., 0., 0., 0., 2., 2., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 2., 2., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 2., 2., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 2., 2., 0., 0.]])

        assert_array_equal(tile_1, val_1)
        assert_array_equal(tile_2, val_2)






    def test_is_overlap(self):

        tpl = np.array([[2, 13],
                        [4, 15]])

        b1 = np.array([[3, 6],
                       [10, 14]])
        b2 = np.array([[9, 16],
                       [3, 7]])
        b3 = np.array([[13, 17],
                       [10, 13]])
        b4 = np.array([[0, 1],
                       [12, 13]])
        b5 = np.array([[1, 4],
                       [13, 17]])
        b6 = np.array([[3, 6],
                       [2, 5]])
        b7 = np.array([[12, 15],
                       [14, 17]])
        b8 = np.array([[0, 3],
                       [2, 5]])
        b9 = np.array([[0, 3],
                       [4, 7]])


        self.assertTrue(ils.is_overlap(tpl, b1))
        self.assertTrue(ils.is_overlap(tpl, b2))
        self.assertFalse(ils.is_overlap(tpl, b3))
        self.assertFalse(ils.is_overlap(tpl, b4))
        self.assertTrue(ils.is_overlap(tpl, b5))
        self.assertTrue(ils.is_overlap(tpl, b6))
        self.assertTrue(ils.is_overlap(tpl, b7))
        self.assertTrue(ils.is_overlap(tpl, b8))
        self.assertTrue(ils.is_overlap(tpl, b9))

        tpl = np.array([[2, 7],
                        [4, 10]])
        b1 = np.array([[2, 3],
                       [4, 10]])
        self.assertTrue(ils.is_overlap(tpl, b1))
        # assert False


    def test_tile_loc_from_slices(self):

        s1 = np.array([[3, 6],
                       [10, 14],
                       [0, 1]])

        s2 = np.array([[9, 16],
                       [3, 7],
                       [0, 1]])

        s3 = np.array([[12, 15],
                       [14, 17],
                       [0, 1]])

        pos, shape = ils.tile_loc_from_slices([s1, s2, s3])
        assert_array_equal(pos, np.array([3, 3, 0]))
        assert_array_equal(shape, np.array([13, 14, 1]))

    def test_get_slices_for(self):

        q_pos = np.array((5, 4))
        q_shape = np.array((3, 3))
        p_pos = np.array((4, 2))
        p_shape = np.array((5, 6))

        q_slice = ils._get_slices_for(p_pos, q_pos, p_shape, q_shape)
        p_slice = ils._get_slices_for(q_pos, p_pos, q_shape, p_shape)
        self.assertEqual(q_slice, [slice(0, 3), slice(0, 3)])
        self.assertEqual(p_slice, [slice(1, 4), slice(2, 5)])

        q_pos = np.array((5, 4))
        q_shape = np.array((7, 3))
        p_pos = np.array((4, 2))
        p_shape = np.array((5, 6))

        q_slice = ils._get_slices_for(p_pos, q_pos, p_shape, q_shape)
        p_slice = ils._get_slices_for(q_pos, p_pos, q_shape, p_shape)
        self.assertEqual(q_slice, [slice(0, 4), slice(0, 3)])
        self.assertEqual(p_slice, [slice(1, 5), slice(2, 5)])

        q_pos = np.array((1, 3))
        q_shape = np.array((7, 4))
        p_pos = np.array((4, 2))
        p_shape = np.array((5, 6))

        q_slice = ils._get_slices_for(p_pos, q_pos, p_shape, q_shape)
        p_slice = ils._get_slices_for(q_pos, p_pos, q_shape, p_shape)
        self.assertEqual(q_slice, [slice(3, 7), slice(0, 4)])
        self.assertEqual(p_slice, [slice(0, 4), slice(1, 5)])
