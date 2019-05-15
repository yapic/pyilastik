#!/usr/bin/env python3
'''
Convert Ilastik project files (ilp) to Tiff files.

Usage:
  convert_ilp_to_tif.py <ilp_file> [<image_path>]

'''
from docopt import docopt

import re
import os
import sys

import lib

import numpy as np
import skimage.io

def main(ilp_filepath, image_path):
    output_path = ilp_filepath+'-tiffs'
    try:
        os.mkdir(output_path)
    except:
        pass

    print('Writing output to {}...'.format(output_path))
    for path, (img, labels, _) in lib.read_project(ilp_filepath, image_path=image_path, prediction=False):
        fname = lib.basename(path)

        skimage.io.imsave(os.path.join(output_path, fname+'-img.tif'), img)
        skimage.io.imsave(os.path.join(output_path, fname+'-lbl.tif'), labels)


if __name__ == '__main__':
    args = docopt(__doc__, version='convert_ilp_to_tif 0.1')
    ilp_path = args['<ilp_file>']
    image_path = args['<image_path>']

    try:
        main(ilp_path, image_path)
    except FileNotFoundError as e:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!! Maybe you can fix this error specifying the path to the images using the <image_path> option !!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        raise e

