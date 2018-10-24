import os, sys
import argparse

import config
import utils

import h5py
import numpy as np
import stempeg

from google.cloud import storage
from google.cloud.storage import Blob


def prepData(filenames):
    
    for fi in filenames:
        fi = fi.split('/')[-1]
        print('processing {}'.format(fi))
        stems, fs = stempeg.read_stems("/home/grauj/sigsep/data/train/"+fi)
        mix, drums, bass, comp, voice = map(utils.stft_stereo, stems)
        hdf5_file = h5py.File("stft/train/"+fi[:-9]+".hdf5", mode='w')
        hdf5_file.create_dataset("mix_stft", list(mix.shape), np.float32)
        hdf5_file.create_dataset("tar_stft", [8, mix.shape[1], 513], np.float32)
        hdf5_file["mix_stft"][:,:,:]= mix
        hdf5_file["tar_stft"][:,:,:] = np.concatenate((voice, drums, bass, comp),axis = 0)
        hdf5_file.close()

def normalizeData():
    print('Normalizing data')
    maximus = np.zeros((10,1,513))
    minimus = np.zeros((10,1,513))
    file_list = [x for x in os.listdir('stft/train/') if x.endswith('.hdf5')]
    for fi in file_list:
        print('Normalizing file {}'.format(str(fi)))
        hdf5_file = h5py.File('stft/train/'+fi, "r")
        tar_stft = np.array(hdf5_file["tar_stft"])
        tar_stft_max = tar_stft.max(axis = 1).reshape(8,1,513)
        tar_stft_min = tar_stft.min(axis = 1).reshape(8,1,513)

        mix_stft = np.array(hdf5_file["mix_stft"])
        mix_stft_max = mix_stft.max(axis = 1).reshape(2,1,513)
        mix_stft_min = mix_stft.min(axis = 1).reshape(2,1,513)

        loc_max = np.concatenate((tar_stft_max,mix_stft_max),axis=0)
        loc_min = np.concatenate((tar_stft_min,mix_stft_min),axis=0)
        
        maximus = np.concatenate((maximus,loc_max),axis=1).max(axis=1).reshape(10,1,513)
        minimus = np.concatenate((minimus,loc_min),axis=1).min(axis=1).reshape(10,1,513)

    hdf5_file = h5py.File('stft/train/stats.hdf5', mode='w')

    hdf5_file.create_dataset("feats_maximus", [10,513], np.float32) 
    hdf5_file.create_dataset("feats_minimus", [10,513], np.float32)   
    hdf5_file["feats_maximus"][:] = maximus.reshape(10,513)
    hdf5_file["feats_minimus"][:] = minimus.reshape(10,513)



def listBucketFiles(bucket):
    
    file_list = bucket.list_blobs()
    filenames = [ str(fi).split(',')[1][:-1] for fi in file_list if str(fi).split(',')[1].endswith('.stem.mp4>')]
    
    return filenames

def saveData(file_list):
     
    for fi in file_list:
        name = fi.split('/')[-1]
        blob = Blob(fi[1:], bucket)
        with open('data/train/' + name,'wb') as file_obj:
            blob.download_to_file(file_obj)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Process dataset from bucket')
    parser.add_argument('-db','--database', type=str,
                        help='Optional: argument to indicate optional bucket name. Default: musdb')
    parser.add_argument('-m','--maxmin', type=str,
                        help='Optional: Compute max and mins across dataset')
    args = parser.parse_args()
    if args.maxmin:
        normalizeData()
    if args.database:
        bucket_name = args.database
    else:
        bucket_name = 'musdb'
    
    client = storage.Client()
    try:
        bucket = client.get_bucket(bucket_name)   
    except:
        print('Can\'t find/access bucket')

    file_list = listBucketFiles(bucket)
    prepData(file_list)
