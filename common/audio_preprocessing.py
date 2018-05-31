import librosa
import librosa.display
#%matplotlib inline
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
np.warnings.filterwarnings('ignore')

default_sampling_rate = 16000

def librosa_load_audio_as_mono(pathname, sr=default_sampling_rate, ms_duration=None):
    _path = pathname
    if ms_duration != None:
        y, sr = librosa.load(_path, sr=sr, duration=ms_duration/1e3)
    else:
        y, sr = librosa.load(_path, sr=sr)

    y_mono = librosa.to_mono(y)
    raw = np.array(list(y_mono), dtype=np.float32)
    return raw

def read_audio(pathname, sr=default_sampling_rate):
    return librosa.load(pathname, sr=sr)

def write_audio(pathname, audio, sr):
    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(pathname, (audio * maxv).astype(np.int16), sr)


def tf_wave_to_melspectrogram(wave, sr):
    spectrogram = librosa.feature.melspectrogram(wave, sr=sr, n_mels=40, hop_length=160, n_fft=400, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram
            
def show_melspectrogram(mels):
    librosa.display.specshow(mels, y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.show()
                                
def wave_padding(data, sr, minimum_sec):
    min_data_size = int(np.ceil(sr * minimum_sec))
    if len(data) < min_data_size:
        L = abs(len(data) - min_data_size)
        start = L // 2
        data  = np.pad(data, (start, L-start), 'constant')
    return data

def wavfile_to_melspectrogram(filename, minimum_sec=1., debug_display=False):
    x, sr = librosa.load(filename)
    x = wave_padding(x, sr, minimum_sec)
    mels = tf_wave_to_melspectrogram(x, sr)
    if debug_display:
        display.display(display.Audio(x, rate=sr))
        show_melspectrogram(mels)
        plt.title(filename)
        plt.show()
    return mels

