import re
import os

def basename(path):
    '''
    A version of os.path.basename() but it supports windows paths on *nix machines
    '''
    fname =  os.path.basename(path)

    if re.match(r'^[a-zA-Z]:\\', fname):
        # windows path on linux machine (starts with X:\)
        fname = fname.split('\\')[-1]

    return fname

