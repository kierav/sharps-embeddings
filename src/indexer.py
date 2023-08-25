import glob
import h5py
import os
import pandas as pd
import argparse
from astropy.io import fits
from sunpy.map import Map

META_KEYS_TO_KEEP = ['naxis1','naxis2','t_obs','t_rec','origin','date','telescop','instrume',
                     'usflux','meangam','meangbt','meangbz','meangbh','meanjzd','totusjz',
                     'meanalp','absnjzh','savncpp','meanpot','totpot','meanshr','shrgt45',
                     'r_value','ctype1','ctype2','crpix1','crpix2','crval1','crval2','cdelt1',
                     'cdelt2','imcrpix1','imcrpix2','dsun_obs','dsun_ref','rsun_ref',
                     'crln_obs','crlt_obs','car_rot','obs_vr','obs_vw','obs_vn',
                     'rsun_obs','quality','harpnum','latdtmin','londtmin','latdtmax',
                     'londtmax','omega_dt','size','area','nacr','size_acr','area_acr',
                     'mtot','mnet','mpos_tot','mneg_tot','mmean','mstdev','mskew',
                     'mkurt','lat_min','lon_min', 'lat_max','lon_max','noaa_ar',
                     'noaa_ars','usfluxl','meangbl','missvals','datamin','datamax',
                     'datamedn','datamean']

class Indexer():
    """
    Indexes SHARPs fits files, saving header to csv and cleaning bad quality files

    Parameters:
        sharps_dir (str):       path to sharps files
        save_dir (str):         path to save new files at
        out_file (str):         path to save index at
        start (int, optional):  sharp number to start indexing at
        end (int, optional):    sharp number to end indexing at
    """
    def __init__(self,sharps_dir:str,save_dir:str,out_file:str,
                 start:int=None,end:int=None):
        self.sharps_dir = sharps_dir
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.out_file = out_file

        # find sharps in dir and filter
        self.sharps = os.listdir(sharps_dir)
        self.sharps = sorted([int(sharp.lstrip('sharp_')) for sharp in self.sharps])
        if not start == None:
            self.sharps = self.sharps[self.sharps>=start]
        if not end == None:
            self.sharps = self.sharps[self.sharps<end]

        self.index = {}
        self.index['file'] = []
        self.index['br_file'] = []
        self.index['blos_file'] = []
        for key in META_KEYS_TO_KEEP:
            self.index[key] = []

    def index_all(self):
        """
        Index all sharps in directory and create pandas dataframe index
        """
        for sharp in self.sharps:
            sharp_dir = 'sharp_'+str(sharp)
            self.index_sharp(sharp_dir)

        self.index_df = pd.DataFrame(self.index)
        sample_time = self.index_df['t_obs'].str.rstrip('_TAI')
        sample_time = pd.to_datetime(sample_time,format='%Y.%m.%d_%H:%M:%S.%f')
        self.index_df.insert(1,'sample_time',sample_time)

    def clean_index(self):
        """
        Remove all samples with bad quality flags from index
        """
        bad_samples = self.index_df['quality']>65536
        self.index_df = self.index_df[~bad_samples]

    def save_index(self,out_file:str=None):
        """
        Save all indexed data to csv

        Parameters:
            out_file (str):     optional path to save index at, else saves at init out_file path
        """
        if out_file == None:
            out_file = self.out_file

        self.index_df.to_csv(out_file,index=False)
    
    def index_sharp(self,sharp):
        """
        Index files for a given sharp. Create stacks [Br,Bp,Bt,Blos] for each 
        timestamp and save as a new hdf5 file. Add metadata to index.

        Parameters:
            sharp (str):    given sharp in format 'sharp_100'
        
        Returns:
            Br_files (list):    list of Br files for sharp
        """
        Br_files = glob.glob(os.path.join(self.sharps_dir,sharp)+'/**Br.fits')

        for Br_file in Br_files:
            Bt_file = Br_file.replace('.Br.','.Bt.')
            Bp_file = Br_file.replace('.Br.','.Bp.')
            Blos_file = Br_file.replace('.Br.','.magnetogram.')
            new_file = Br_file.replace('.Br.fits','.h5').replace(self.sharps_dir+'/'+sharp,self.save_dir)
            try:
                self.create_img_stack([Br_file,Bp_file,Bt_file,Blos_file],
                                      new_file)
            except ValueError:
                continue

            self.index['file'].append(new_file)
            self.index['br_file'].append(Br_file)
            self.index['blos_file'].append(Blos_file)

            self.save_header(Br_file)

        return Br_files        

    def create_img_stack(self,files,new_file):
        """
        Opens stack of files and saves them as one stacked hdf5 file.
        
        Parameters:
            files (list):   files to open and stack
            new_file (str): path to save new file at
        
        Raises ValueError exception if can't open a file.
        """
        img_stack = []
        for file in files:
            try:
                with fits.open(file,cache=False) as data_fits:
                    data_fits.verify('fix')
                    img = data_fits[1].data
            except ValueError:
                raise ValueError
            img_stack.append(img)

        with h5py.File(new_file,'w') as h5:
            h5.create_dataset('hmi', data=img_stack,compression='gzip')

    def save_header(self,file):
        """
        Read header and save desired keys

        Parameters:
            file (str):     path to fits file
        """
        sharp_map = Map(file)
        for key in META_KEYS_TO_KEEP:
            try:
                self.index[key].append(sharp_map.meta[key])
            except Exception as e:
                print('Adding None for ',key,'for',file)
                self.index[key].append(None)



def parse_args(args=None):
    """
    Parses command line arguments to script. Sets up argument for which 
    dataset to index.

    Parameters:
        args (list):    defaults to parsing any command line arguments
    
    Returns:
        parser args:    Namespace from argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir',
                        type=str,
                        default='data',
                        help='sharps directory to index'
                        )
    parser.add_argument('-s','--savedir',
                        type=str,
                        default='data/hdf5',
                        help='directory to save stacked hdf5 files'
                        )
    parser.add_argument('-i','--outfile',
                        type=str,
                        default='data/index.csv',
                        help='path to save index file'
                        )
    parser.add_argument('--start',
                    type=int,
                    default=0,
                    help='sharp to start indexing at'
                    )
    parser.add_argument('--end',
                    type=int,
                    default=10000,
                    help='sharp to end indexing at'
                    )
    return parser.parse_args(args)


def main():
    parser = parse_args()
    
    indexer = Indexer(parser.dir,parser.savedir,parser.outfile,parser.start,parser.end)
    indexer.index_all()
    indexer.clean_index()
    indexer.save_index()