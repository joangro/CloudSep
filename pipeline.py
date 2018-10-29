import numpy as np
import os
import time
import h5py
import random

NUM_EPOCH=300
FILES_PER_BATCH=2
SAMPLES_PER_FILE=5
MAX_TIME_CONTEXT=30

class DataAugmentation():
    def __init__(self, status=True):
        self.status = status
   
    def swapChannels(self, inputs, targets):
        targets = np.array(targets)
        for i in range(4):
            aux_tar = targets[i*2:i*2+2,:,:]
            aux_tar[:,:,:] = aux_tar[::-1,:,:]
            if i == 0:
                swapped_tar = aux_tar[:,:,:]
            else:
                swapped_tar = np.append(swapped_tar, aux_tar, axis=0)
        return np.array(inputs)[::-1,:,:], swapped_tar

    def muteRandomSource(self, inputs, targets):
        rand_source =  random.randint(0,4)

    def createNewMix(self):
        pass

def dataGen():
    da = DataAugmentation(status=True)
    filenames = [x for x in os.listdir('stft/train/') if x.endswith('.hdf5') and not x.startswith('stats')]
    num_files = len(filenames)
    for i in range(NUM_EPOCH):
        targets = []
        inputs = []
        for i in range(FILES_PER_BATCH):
            rand_file = filenames[random.randint(0,num_files-1)]
            hdf5_file = h5py.File('stft/train/'+rand_file, "r+")
            tar_stft = hdf5_file['tar_stft']
            mix_stft = hdf5_file['mix_stft']
            for i in range(SAMPLES_PER_FILE):
                rand_index = np.random.randint(0,mix_stft.shape[1]-MAX_TIME_CONTEXT)
                if random.random()<0.2:
                    target_aux = tar_stft[:,rand_index:rand_index+MAX_TIME_CONTEXT,:]
                    input_aux = mix_stft[:,rand_index:rand_index+MAX_TIME_CONTEXT,:]
                    da_inputs, da_targets = da.swapChannels(input_aux, target_aux)
                    targets.append(da_targets[:,:,:])
                    inputs.append(da_inputs[:,:,:])
                else:
                    targets.append(tar_stft[:,rand_index:rand_index+MAX_TIME_CONTEXT,:])
                    inputs.append(mix_stft[:,rand_index:rand_index+MAX_TIME_CONTEXT,:])
            hdf5_file.close()
            
        yield np.array(inputs), np.array(targets)

if __name__ == '__main__':
    gen = dataGen()
    for inp, tar in gen:
        print('Inp shape: {} \nTar shape: {}'.format(inp.shape, tar.shape))


