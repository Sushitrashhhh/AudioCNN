import pathlib as Path
import sys 
import pandas as pd, numpy as np

import modal
import torch, torchaudio, torch.nn as nn, torchaudio.transforms as T, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import AudioCNN


app = modal.App("audio-cnn")

image = (modal.Image.debian_slim().pip_install_from_requirements("requirements.txt").apt_install(['wget','unzip','ffmpeg', 'libsndfile1']).run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ]).add_local_python_source('model'))

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
modal_volume = modal.Volume.from_name('esc-model', create_if_missing=True)

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_files, split = True, transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_files)
        self.split = split
        self.transform = transform

        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold']!=5]
        else:
            self.metadata = self.metadata[self.metadata['fold']==5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform
        return spectrogram, row['label']
    
    def mixup_data(x,y):
        lam = np.random.beta(0.2,0.2)

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

@app.function(image=image, gpu='A10G', volume={'/data':volume, '/models': modal_volume}, timeout= 60*60*3)
def train():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)


    esc50_dir = Path('/opt/esc50-data')
    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=220550,
            n_fft = 1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )
    


@app.local_entrypoint()
def main():
    print('lala')