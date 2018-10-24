from __future__ import print_function, division
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import pipeline
import config


NUM_EPOCH_TRAIN = 2000
MAX_TIME_CONTEXT = pipeline.MAX_TIME_CONTEXT

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.conv_hor = (1, 513)
        self.conv_ver = (15, 1)
        
        self.ch_out_hor = 50
        self.ch_out_ver = 30

        # Principal encoder
        self.encoder = nn.Sequential(
                nn.Conv2d(2, self.ch_out_hor, self.conv_hor, stride = 1, padding = 0, bias = True, groups = 1),
                nn.Conv2d(self.ch_out_hor, self.ch_out_ver, self.conv_ver, stride = 1, padding = 0, bias = True, groups = 1)
        )

        # Source decoders
        self.drums_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.ch_out_ver, self.ch_out_hor, self.conv_ver, stride = 1, padding = 0, bias = True, groups = 1),
                nn.ConvTranspose2d(self.ch_out_hor, 2, self.conv_hor, stride = 1, padding = 0, bias = True, groups = 1)
        )
        self.voice_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.ch_out_ver, self.ch_out_hor, self.conv_ver, stride = 1, padding = 0, bias = True, groups = 1),
                nn.ConvTranspose2d(self.ch_out_hor, 2, self.conv_hor, stride = 1, padding = 0, bias = True, groups = 1)
        )
        self.bass_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.ch_out_ver, self.ch_out_hor, self.conv_ver, stride = 1, padding = 0, bias = True, groups = 1),
                nn.ConvTranspose2d(self.ch_out_hor, 2, self.conv_hor, stride = 1, padding = 0, bias = True, groups = 1)
        )
        self.other_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.ch_out_ver, self.ch_out_hor, self.conv_ver, stride = 1, padding = 0, bias = True, groups = 1),
                nn.ConvTranspose2d(self.ch_out_hor, 2, self.conv_hor, stride = 1, padding = 0, bias = True, groups = 1)
        )

        # FCL
        self.layer_first = nn.Sequential(
                nn.Linear(self.ch_out_ver*16, 128),
                nn.ReLU()
        )
        self.layer_drums = nn.Sequential(
                nn.Linear(128, self.ch_out_ver*16),
                nn.ReLU()
        )
        self.layer_voice = nn.Sequential(
                nn.Linear(128, self.ch_out_ver*16),
                nn.ReLU()
        )
        self.layer_bass = nn.Sequential(
                nn.Linear(128, self.ch_out_ver*16),
                nn.ReLU()
        )	
        self.layer_other = nn.Sequential(
                nn.Linear(128, self.ch_out_ver*16),
                nn.ReLU()
        )

        self.final_output = nn.ReLU()


    def forward(self, x):
        encode = self.encoder(x)
        encode_reshaped = encode.view(pipeline.FILES_PER_BATCH*5, -1)
        
        layer_output = self.layer_first(encode_reshaped)
        
        layer_output_voice = self.layer_voice(layer_output) 
        layer_output_voice = layer_output_voice.view(-1,self.ch_out_ver,16,1)
        output_voice = self.voice_decoder(layer_output_voice)

        layer_output_drums = self.layer_drums(layer_output)
        layer_output_drums = layer_output_drums.view(-1,self.ch_out_ver,16,1)
        output_drums = self.drums_decoder(layer_output_drums)

        layer_output_bass = self.layer_bass(layer_output)
        layer_output_bass = layer_output_bass.view(-1,self.ch_out_ver,16,1)
        output_bass = self.bass_decoder(layer_output_bass)

        layer_output_other = self.layer_other(layer_output)
        layer_output_other = layer_output_other.view(-1,self.ch_out_ver,16,1)
        output_other = self.other_decoder(layer_output_other)
        
        reshape_output = torch.cat((output_voice, output_drums, output_bass, output_other), 1)
        output_final = self.final_output(reshape_output)

        return output_final 

def loss_calc(inputs, targets, loss_func, autoencoder):
    targets = targets *np.linspace(1.0,0.5,513)
    targets_cuda = Variable(torch.FloatTensor(targets))
    inputs = Variable(torch.FloatTensor(inputs))
    output = autoencoder(inputs)

    vocals = output[:,:2,:,:]
    drums = output[:,2:4,:,:]
    bass = output[:,4:6,:,:]
    others = output[:,6:,:,:]

    total_sources = vocals + bass + drums + others

    mask_vocals = vocals/total_sources
    mask_drums = drums/total_sources
    mask_bass = bass/total_sources
    mask_others = others/total_sources

    out_vocals = inputs * mask_vocals
    out_drums = inputs * mask_drums
    out_bass = inputs * mask_bass
    out_others = inputs * mask_others

    targets_vocals = targets_cuda[:,:2,:,:]
    targets_drums = targets_cuda[:,2:4,:,:]
    targets_bass = targets_cuda[:,4:6,:,:]
    targets_others = targets_cuda[:,6:,:,:]

    step_loss_vocals = loss_func(out_vocals, targets_vocals)
    alpha_diff = config.alpha * loss_func(out_vocals, targets_bass)
    alpha_diff += config.alpha * loss_func(out_vocals, targets_drums)
    beta_other_voc = config.beta_voc * loss_func(out_vocals, targets_others)

    step_loss_drums = loss_func(out_drums, targets_drums)
    alpha_diff += config.alpha * loss_func(out_drums, targets_vocals)
    alpha_diff += config.alpha * loss_func(out_drums, targets_bass)
    beta_other = config.beta * loss_func(out_drums, targets_others)

    step_loss_bass = loss_func(out_bass, targets_bass)
    alpha_diff +=  config.alpha *  loss_func(out_bass, targets_vocals)
    alpha_diff +=  config.alpha *  loss_func(out_bass, targets_drums)
    beta_other = config.beta * loss_func(out_bass, targets_others)

    return step_loss_vocals, step_loss_drums, step_loss_bass, alpha_diff, beta_other, beta_other_voc




def trainNet():
    autoencoder = AutoEncoder()
    optimizer = torch.optim.Adadelta(autoencoder.parameters(), lr = 1, rho=0.95)
    loss_func = nn.MSELoss( size_average=False )
    for epoch in range(NUM_EPOCH_TRAIN):
        start_time = time.time()

        generator = pipeline.dataGen()
        optimizer.zero_grad()

        train_loss = 0
        train_loss_vocals = 0
        train_loss_drums = 0
        train_loss_bass = 0
        train_alpha_diff = 0 
        train_beta_other = 0
        train_beta_other_voc = 0

        for inputs, targets in generator:
            step_loss_vocals, step_loss_drums, step_loss_bass, alpha_diff, beta_other, beta_other_voc = loss_calc(inputs, targets, loss_func, autoencoder)
            step_loss = abs(step_loss_vocals + step_loss_drums + step_loss_bass - beta_other - alpha_diff - beta_other_voc)
            train_loss += step_loss.item()
            train_loss_vocals +=step_loss_vocals.item()
            train_loss_drums +=step_loss_drums.item()
            train_loss_bass +=step_loss_bass.item()
            train_alpha_diff += alpha_diff.item()
            train_beta_other += beta_other.item() 
            train_beta_other_voc+=beta_other_voc.item() 

            step_loss.backward()

            optimizer.step()
        train_loss = train_loss/(200*NUM_EPOCH_TRAIN*MAX_TIME_CONTEXT*513)
        train_loss_vocals = train_loss_vocals/(200*NUM_EPOCH_TRAIN*MAX_TIME_CONTEXT*513)
        train_loss_drums = train_loss_drums/(200*NUM_EPOCH_TRAIN*MAX_TIME_CONTEXT*513)
        train_loss_bass = train_loss_bass/(200*NUM_EPOCH_TRAIN*MAX_TIME_CONTEXT*513)
        duration = time.time()-start_time
        print('Epoch number {} took {} seconds, epoch total loss:'.format(epoch, duration, train_loss))
        print('                                  epoch vocal loss: %.7f' % (train_loss_vocals))
        print('                                  epoch drums loss: %.7f' % (train_loss_drums))
        print('                                  epoch bass  loss: %.7f' % (train_loss_bass))



if __name__ == '__main__':
    trainNet()
