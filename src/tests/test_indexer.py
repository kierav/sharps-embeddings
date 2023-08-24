from src.indexer import Indexer
from datetime import datetime
import glob
import h5py
import numpy as np
import os
import pandas as pd

sharps_dir = 'data'
save_dir = 'src/tests'
sharp = 'sharp_1845'
out_file = 'src/tests/test_indexer.csv'

def test_indexer_init():
    indexer = Indexer(sharps_dir,save_dir,out_file)
    assert type(indexer.out_file) == str
    assert type(indexer.sharps) == list
    assert indexer.sharps[0] == sharp

def test_index_sharp():
    indexer = Indexer(sharps_dir,save_dir,out_file)
    files = indexer.index_sharp(sharp)
    assert len(files) > 0

def test_save_header():
    indexer = Indexer(sharps_dir,save_dir,out_file)
    Br_files = glob.glob(os.path.join(sharps_dir,sharp)+'/**Br.fits')
    file = Br_files[0]
    indexer.save_header(file)
    assert len(indexer.index['quality']) == 1 

def test_create_img_stack():
    indexer = Indexer(sharps_dir,save_dir,out_file)
    Br_files = glob.glob(os.path.join(sharps_dir,sharp)+'/**Br.fits')
    Br_file = Br_files[0]
    Bt_file = Br_file.replace('.Br.','.Bt.')
    Bp_file = Br_file.replace('.Br.','.Bp.')
    Blos_file = Br_file.replace('.Br.','.magnetogram.')
    indexer.create_img_stack([Br_file,Bt_file,Bp_file,Blos_file],
                             'src/tests/test_img_stack.h5')
    img_stack = np.array(h5py.File('src/tests/test_img_stack.h5','r')['hmi'])
    assert np.shape(img_stack)[0] == 4

def test_index_sharp():
    indexer = Indexer(sharps_dir,save_dir,out_file)
    indexer.index_sharp(sharp)
    new_files = glob.glob(save_dir+'/hmi*.h5')
    assert len(indexer.index['file']) == len(new_files)
    print(indexer.index)

def test_index_all():
    indexer = Indexer(sharps_dir,save_dir,out_file)
    indexer.index_all()
    new_files = glob.glob(save_dir+'/hmi*.h5')
    assert len(indexer.index['file']) == len(new_files)

def test_save_index():
    indexer = Indexer(sharps_dir,save_dir,out_file)
    indexer.index_all()
    indexer.save_index()
    index = pd.read_csv(out_file)
    assert len(index) == len(indexer.index['file'])
    assert index['sample_time'].dtype == datetime