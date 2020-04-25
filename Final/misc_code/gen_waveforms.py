import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def generate_waveform(filename,filepath):
    
    if(filename.split('.')[-1] != 'wav'):
        print('Please use a wav format file...')
        return -1
    
    y,sr = librosa.load(filepath)

    output_filename = (filename[:-4])+'_waveform.png'
    print(f'Output file:\t{output_filename}')
    plt.figure(figsize=(16,8))
    librosa.display.waveplot(y,sr=sr,x_axis='s')
    plt.title(f'Waveform of {filename}')
    plt.ylabel('Amplitude')
    plt.savefig('../pictures/' + output_filename)


    output_filename = (filename[:-4])+'_spectrum.png'
    print(f'Output file:\t{output_filename}')
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    frequency = np.linspace(0,sr,len(magnitude))
    left_frequency = frequency[:int(len(frequency)/2)]
    left_magnitude = magnitude[:int(len(frequency)/2)]
    plt.clf()
    plt.figure(figsize=(16,8))
    plt.plot(left_frequency,left_magnitude)
    plt.title(f'Power spctrum of {filename}')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.savefig('../pictures/' + output_filename)

    
    output_filename = (filename[:-4])+'_spectrogram.png'
    print(f'Output file:\t{output_filename}')
    n_fft = 2048
    hop_length = 512
    stft = librosa.core.stft(y,hop_length=hop_length,n_fft=n_fft)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    plt.clf()
    plt.figure(figsize=(16,8))
    librosa.display.specshow(log_spectrogram,sr=sr,hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Power spectrogram of {filename}')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig('../pictures/' + output_filename)


    output_filename = (filename[:-4])+'_mel-frequency_spectrogram.png'
    print(f'Output file:\t{output_filename}')
    mfs = librosa.feature.melspectrogram(y,n_fft=n_fft,hop_length=hop_length,n_mels=128)
    mfs_dB = librosa.power_to_db(mfs)
    plt.clf()
    plt.figure(figsize=(16,8))
    librosa.display.specshow(mfs_dB,x_axis='time',y_axis='mel',sr=sr,hop_length=hop_length,fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-frequency spectrogram of {filename}')
    plt.tight_layout()
    plt.savefig('../pictures/' + output_filename)
    

    output_filename = (filename[:-4])+'_MFCC.png'
    print(f'Output file:\t{output_filename}')
    mfccs = librosa.feature.mfcc(y,n_fft=n_fft,hop_length=hop_length,n_mfcc=13)
    plt.clf()
    plt.figure(figsize=(16,8))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.ylabel('MFCC Coefficients')
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig('../pictures/' + output_filename)


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    filepath = filedialog.askopenfilename()
    filename = filepath.split('/')[-1]
    print(f'File to read:\t{filepath}')

    generate_waveform(filename,filepath)

    root.destroy()