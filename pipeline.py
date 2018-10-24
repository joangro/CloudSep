import numpy as np
import os
import time
import h5py
import random

NUM_EPOCH=500
FILES_PER_BATCH=20
SAMPLES_PER_FILE=5
MAX_TIME_CONTEXT=30


def dataGen():
    
    filenames = [x for x in os.listdir('stft/train/') if x.endswith('.hdf5')]
    num_files = len(filenames)
    for i in range(NUM_EPOCH):
        inputs = []
        targets = []
        for i in range(FILES_PER_BATCH):
            rand_file = filenames[random.randint(0,num_files)]
            hdf5_file = h5py.File('stft/train/'+rand_file)
            tar_stft = hdf5_file['tar_stft']
            mix_stft = hdf5_file['mix_stft']
            for i in range(SAMPLES_PER_FILE):
                rand_index = np.random.randint(0,mix_stft.shape[1]-MAX_TIME_CONTEXT)
                inputs.append(tar_stft[:,rand_index:rand_index+MAX_TIME_CONTEXT,:])
                targets.append(mix_stft[:,rand_index:rand_index+MAX_TIME_CONTEXT,:])
            hdf5_file.close()
            
        yield np.array(inputs), np.array(targets)

if __name__ == '__main__':
    gen = dataGen()
    for inp, tar in gen:
        print('Inp shape: {} \nTar shape: {}'.format(inp.shape, tar.shape))


