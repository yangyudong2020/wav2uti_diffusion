# -*- coding: utf-8 -*-


import argparse
import glob
import os
from shutil import copy

import h5py
import numpy as np
import soundfile as sf
import torch
import tqdm
import fairseq
from torch import nn
import transformers

transformers.logging.set_verbosity_error()

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# config
MAX_LENGTH = 256
DEFAULT_WAV2_NAME = 'pretraining/xlsr_53_56k.pt'
WAV2_CONFIGS = {}

def list_files_in_directory(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return []

    # Get all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files

def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    assert sr == 16e3

    return wav, 16e3


class PretrainedWav2VecModel(nn.Module):
    def __init__(self, fname):
        super().__init__()
        checkpoint = torch.load(fname)
        wav2vec2_encoder = fairseq.models.wav2vec.Wav2Vec2Model.build_model(checkpoint['cfg']['model'])
        #wav2vec2_encoder.load_state_dict(checkpoint['model'])
        #model = model[0]
        #model.eval()

        self.model = wav2vec2_encoder

    def forward(self, x):
        with torch.no_grad():
            features = wav2vec2_encoder(audio, features_only=True, mask=False)['x']
        return features
        
    
    
class Prediction:
    """ Lightweight wrapper around a fairspeech embedding model """

    def __init__(self, fname, gpu=0): 
        self.gpu = gpu
        self.model = PretrainedWav2VecModel(fname).cuda(gpu)

    def __call__(self, x):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        with torch.no_grad():
            z, c = self.model(x.unsqueeze(0))

        return z.squeeze(0).cpu().numpy(), c.squeeze(0).cpu().numpy()


class H5Writer:
    """ Write features as hdf5 file in flashlight compatible format """

    def __init__(self, fname):
        self.fname = fname
        os.makedirs(os.path.dirname(self.fname), exist_ok=True)

    def write(self, data):
        channel, T = data.shape

        with h5py.File(self.fname, "w") as out_ds:
            data = data.T.flatten()
            out_ds["features"] = data
            out_ds["info"] = np.array([16e3 // 160, T, channel])
            
            
if __name__ == "__main__":

    # Defind wav path
    wav_file_path = '/mnt/shareEEx/yangyudong/utlnet/video-pytorch/wavDir/speaker00012_M_s1_stn00096.wav'  # 替换为实际的 WAV 文件路径
    
    audio_data, sample_rate = read_audio(wav_file_path)
  
    model = PretrainedWav2VecModel(DEFAULT_WAV2_NAME)
    
    audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)  # 添加批次维度
    

    with torch.no_grad():
        z= model(audio_tensor)
        
    print(z.shape)
